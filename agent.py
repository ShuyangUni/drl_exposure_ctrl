'''
Author: Shuyang Zhang
Date: 2024-05-27 15:30:55
LastEditors: ShuyangUni shuyang.zhang1995@gmail.com
LastEditTime: 2024-08-25 20:03:59
Description: 

Copyright (c) 2024 by Shuyang Zhang, All Rights Reserved. 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from torch.distributions import Normal
from log import Log


class CNNEncoder(nn.Module):
    def __init__(self, n_input):
        super(CNNEncoder, self).__init__()
        self.n_input = n_input
        self.conv1 = nn.Conv2d(self.n_input, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.feature_size = 3136

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        out = self.flatten(out)
        return out


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_width):
        super(Actor, self).__init__()
        self.cnn = CNNEncoder(state_dim[0])
        self.l1 = nn.Linear(self.cnn.feature_size, hidden_width)
        self.l2 = nn.Linear(hidden_width, hidden_width)
        self.mean_layer = nn.Linear(hidden_width, action_dim)
        self.log_std_layer = nn.Linear(hidden_width, action_dim)

    def forward(self, x, deterministic=False, with_logprob=True):
        x = self.cnn(x)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)

        pi_dist = Normal(mean, std)
        if deterministic:
            pi_a = mean
        else:
            pi_a = pi_dist.rsample()

        if with_logprob:
            log_pi = pi_dist.log_prob(pi_a).sum(dim=1, keepdim=True)
            log_pi -= (2 * (np.log(2) - pi_a - F.softplus(-2 * pi_a))
                       ).sum(dim=1, keepdim=True)
        else:
            log_pi = None

        a = torch.tanh(pi_a)

        return a, log_pi


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_width):
        super(Critic, self).__init__()
        self.cnn_1 = CNNEncoder(state_dim[0])
        self.l1_1 = nn.Linear(self.cnn_1.feature_size +
                              action_dim, hidden_width)
        self.l1_2 = nn.Linear(hidden_width, hidden_width)
        self.l1_3 = nn.Linear(hidden_width, hidden_width)
        self.l1_4 = nn.Linear(hidden_width, 1)

        self.cnn_2 = CNNEncoder(state_dim[0])
        self.l2_1 = nn.Linear(self.cnn_2.feature_size +
                              action_dim, hidden_width)
        self.l2_2 = nn.Linear(hidden_width, hidden_width)
        self.l2_3 = nn.Linear(hidden_width, hidden_width)
        self.l2_4 = nn.Linear(hidden_width, 1)

    def forward(self, s, a):
        f1 = self.cnn_1(s)
        s_a_1 = torch.cat([f1, a], 1)
        q1 = F.relu(self.l1_1(s_a_1))
        q1 = F.relu(self.l1_2(q1))
        q1 = F.relu(self.l1_3(q1))
        q1 = self.l1_4(q1)

        f2 = self.cnn_2(s)
        s_a_2 = torch.cat([f2, a], 1)
        q2 = F.relu(self.l2_1(s_a_2))
        q2 = F.relu(self.l2_2(q2))
        q2 = F.relu(self.l2_3(q2))
        q2 = self.l2_4(q2)
        return q1, q2


class SAC(object):
    def __init__(self, log: Log, params):
        self.log = log
        self.device = params['device']
        self.hidden_width = params['sac_hidden_dim']
        self.batch_size = params['sac_batch_size']
        self.gamma = params['sac_gamma']
        self.tau = params['sac_tau']
        self.lr = params['sac_lr']
        self.alpha_lr = params['sac_alpha_lr']
        self.adaptive_log_alpha_init = params['sac_adaptive_log_alpha_init']
        self.adaptive_alpha = params['sac_adaptive_alpha']
        if self.adaptive_alpha:
            self.target_entropy = -params['action_dim']
            self.log_alpha = torch.tensor(
                [self.adaptive_log_alpha_init], requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = torch.optim.Adam(
                [self.log_alpha], lr=self.alpha_lr)
        else:
            self.alpha = params['sac_alpha_const']

        self.actor = Actor(
            params['state_dim'], params['action_dim'], self.hidden_width).to(self.device)
        self.critic = Critic(
            params['state_dim'], params['action_dim'], self.hidden_width).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.lr)

    def take_action(self, s, deterministic=False):
        s_in = torch.unsqueeze(torch.tensor(
            s, dtype=torch.float), 0).to(self.device)
        a, _ = self.actor(s_in, deterministic, False)
        return a.data.cpu().numpy().flatten()

    def learn(self, replay_buffer):
        self.log.add_log("[SAC]: learn")
        batch_s, batch_a, batch_r, batch_s_, batch_dw = replay_buffer.sample(
            self.batch_size)

        batch_s = batch_s.to(self.device)
        batch_a = batch_a.to(self.device)
        batch_r = batch_r.to(self.device)
        batch_s_ = batch_s_.to(self.device)
        batch_dw = batch_dw.to(self.device)

        with torch.no_grad():
            batch_a_, log_pi_ = self.actor(batch_s_)
            target_Q1, target_Q2 = self.critic_target(batch_s_, batch_a_)
            target_Q = batch_r + self.gamma * \
                (1 - batch_dw) * \
                (torch.min(target_Q1, target_Q2) - self.alpha * log_pi_)

        current_Q1, current_Q2 = self.critic(batch_s, batch_a)
        critic_loss = F.mse_loss(current_Q1, target_Q) + \
            F.mse_loss(current_Q2, target_Q)
        self.log.add_log(f"[SAC]: critic_loss: {critic_loss.item()}")
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        for params in self.critic.parameters():
            params.requires_grad = False

        a, log_pi = self.actor(batch_s)
        Q1, Q2 = self.critic(batch_s, a)
        Q = torch.min(Q1, Q2)
        actor_loss = (self.alpha * log_pi - Q).mean()
        self.log.add_log(f"[SAC]: Q1: {Q1.detach().mean().item()}")
        self.log.add_log(f"[SAC]: Q2: {Q2.detach().mean().item()}")
        self.log.add_log(f"[SAC]: actor_loss: {actor_loss.item()}")

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for params in self.critic.parameters():
            params.requires_grad = True

        if self.adaptive_alpha:
            alpha_loss = -(self.log_alpha.exp() * (log_pi +
                           self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
        self.log.add_log(f"[SAC]: alpha: {self.alpha.item()}")
        self.log.add_log(f"[SAC]: log_alpha: {self.log_alpha.item()}")
        self.log.add_log(f"[SAC]: alpha_loss: {alpha_loss.item()}")

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filepath):
        torch.save(self.actor.state_dict(), filepath)
        self.log.add_log(f"[SAC]: save actor params to {filepath}")
        self.log.save_buffer_to_file()
