'''
Author: Shuyang Zhang
Date: 2024-06-06 21:20:32
LastEditors: ShuyangUni shuyang.zhang1995@gmail.com
LastEditTime: 2024-08-25 20:43:51
Description: 

Copyright (c) 2024 by Shuyang Zhang, All Rights Reserved. 
'''
from env import ExposureEnv
from agent import Actor
import torch
import time

params = {'device': torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
          'state_dim': (4, 84, 84),
          'action_dim': 1,
          'len_episode': 100000,

          'env_mode_test': True,
          'env_data_argumentation': True,
          'env_seq_filepath': "config/infer.yaml",
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

          'sac_hidden_dim': 512
          }


agent = Actor(params['state_dim'], params['action_dim'], params['sac_hidden_dim'])
agent.load_state_dict(torch.load('model/actor_drl_feat_10000.pth'))
agent.eval()

# setup env
env = ExposureEnv(None, params, params['env_crf_filepath'], params['len_episode'])

t_count = 0.0
count = 0

s, _ = env.reset(frame_id=0)
done = False
while not done:
    env.render(1)
    s_in = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
    t0 = time.time()
    a, _ = agent(s_in, True, False)
    t1 = time.time()
    count = count + 1
    t_count = t_count + t1 - t0
    a = a.data.numpy().flatten()[0]
    # a = env.random_action()
    s_, r, done = env.step(a)
    env.show_info()
    s = s_

print(f"avg time comsumption: {t_count/count}s")