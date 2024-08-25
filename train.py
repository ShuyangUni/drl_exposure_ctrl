'''
Author: Shuyang Zhang
Date: 2024-06-06 18:13:51
LastEditors: ShuyangUni shuyang.zhang1995@gmail.com
LastEditTime: 2024-08-25 20:46:47
Description: 

Copyright (c) 2024 by Shuyang Zhang, All Rights Reserved. 
'''

from log import Log
from agent import SAC
from env import ReplayBuffer, ExposureEnv
import env as env
from datetime import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch
import random
import numpy as np

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


# init tensorboard
writer_train = SummaryWriter(f"log/")

# init log
current_time = datetime.now()
formatted_time = current_time.strftime('%Y-%m-%d-%H-%M-%S')
save_path = f"log/log_{formatted_time}.txt"
log = Log(save_path)

# parameters
params = {'device': torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
          'state_dim': (4, 84, 84),
          'action_dim': 1,
          'len_episode': 500,

          'rb_buffer_size': 50000,

          'tr_warm_size': 5000,
          'tr_n_episodes': 1000000,
          'tr_n_update': 50,
          'tr_n_save': 1000,

          'env_mode_test': False,
          'env_data_argumentation': True,
          'env_seq_filepath': "config/train.yaml",
          'env_crf_filepath': "config/crf.yaml",
          'env_img_ori_h': 192,
          'env_img_ori_w': 256,
          'env_expo_lb': 50,
          'env_expo_ub': 2000000,
          'env_expo_init': 10000,

          #   'rwd_mode': "stat",
          'rwd_mode': "feat",
          'rwd_mean_target': 0.5,
          'rwd_w_flk': 0.2,
          'rwd_w_detect': 0.005,
          'rwd_w_match': 0.005,

          'sac_hidden_dim': 512,
          'sac_batch_size': 256,
          'sac_gamma': 0.99,
          'sac_tau': 0.005,
          'sac_lr': 1e-4,
          'sac_alpha_lr': 1e-4,
          'sac_adaptive_alpha': True,
          'sac_alpha_const': 0.2,
          'sac_adaptive_log_alpha_init': -1.0
          }

for key, value in params.items():
    log.add_log(f"[train]: {key}: {value}")
    log.save_buffer_to_file()

# setup env
env = ExposureEnv(
    log, params, params['env_crf_filepath'], params['len_episode'])

# buffer
replay_buffer = ReplayBuffer(params)

# agent
agent = SAC(log, params)

# warm up
print("Replay buffer warm up")
while replay_buffer.len() < params['tr_warm_size']:
    s, _ = env.reset()
    done = False
    while not done:
        a = env.random_action()
        s_, r, done = env.step(a)
        replay_buffer.store(s, a, r, s_, done)
        s = s_

# training
return_list = []
episode_len_list = []
count = 0
with tqdm(range(params['tr_n_episodes'])) as pbar:
    for i in pbar:
        log.add_log(f"[train]: episode: {i}")
        log.save_buffer_to_file()
        episode_return = 0
        episode_len = 0
        s, _ = env.reset()
        done = False
        while not done:
            a = agent.take_action(s)
            s_, r, done = env.step(a)
            replay_buffer.store(s, a, r, s_, done)
            s = s_

            episode_return += r
            episode_len += 1
            count = count + 1

            # update
            if count % params['tr_n_update'] == 0:
                agent.learn(replay_buffer)

        return_list.append(episode_return)
        episode_len_list.append(episode_len)

        writer_train.add_scalar('return_episode', episode_return, i)
        writer_train.add_scalar('reward_avg', episode_return / episode_len, i)

        pbar.set_postfix(
            {'episode': f"{i}", 'return': f"{return_list[-1]:.2f}", 'len_epi': f"{episode_len_list[-1]:.1f}"})
        pbar.update(1)

        # save parameters
        if i % params['tr_n_save'] == 0 and i != 0:
            agent.save(f"model/actor_{params['rwd_mode']}_{i:09d}.pth")
