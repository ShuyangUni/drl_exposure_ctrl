'''
Author: Shuyang Zhang
Date: 2024-05-27 15:30:55
LastEditors: ShuyangUni shuyang.zhang1995@gmail.com
LastEditTime: 2024-08-25 20:46:35
Description: 

Copyright (c) 2024 by Shuyang Zhang, All Rights Reserved. 
'''

import numpy as np
import os
import cv2
import yaml
import torch
from scipy.interpolate import UnivariateSpline


class PhotometricSimulator():
    def __init__(self, str_crf_filepath):
        self.str_crf_filepath = str_crf_filepath
        self.g_lookup = self.generate_crf_lookup()
        self.img_lb = 0
        self.img_ub = 65535
        self.irr_lb = self.g_lookup[self.img_lb]
        self.irr_ub = self.g_lookup[self.img_ub]

    def img2irr(self, img: np.uint16):
        irr = self.g_lookup[img]
        return irr

    def irr2img(self, irr: np.float64):
        indices = np.searchsorted(self.g_lookup, irr, side='right') - 1
        img_n = indices / 65535.0
        return img_n

    def img_synthesis(self, img0: np.uint16, expo0, expo1):
        irr0 = self.img2irr(img0)
        irr1 = irr0 - np.log(expo0) + np.log(expo1)
        irr1 = np.clip(irr1, self.irr_lb, self.irr_ub)
        img1_n_syn = self.irr2img(irr1)
        return img1_n_syn

    def generate_crf_lookup(self):
        with open(self.str_crf_filepath, 'r') as f:
            data = yaml.safe_load(f.read())
            g_func_y = data["g_func"]
            g_func_y = np.array(g_func_y, np.float64)

        g_func_x = np.arange(0, 256, 1) / 255
        g_func = UnivariateSpline(
            g_func_x[:-1], g_func_y[:-1], s=0.001, k=5)

        int_x_n = np.arange(0, 65536, 1).astype(np.float64) / 65535.0
        g_lookup = g_func(int_x_n)
        return g_lookup


class Sequence():
    def __init__(self, str_cam_filepath, str_expo_filepath, seq_state, img_w, img_h):
        print(f"loading sequence {str_cam_filepath}")
        data = np.loadtxt(str_cam_filepath)
        self.seq_state = seq_state
        self.image_id = data[:, 0].astype(np.uint32)

        expo_t = data[:, 2].astype(np.float64)
        gain = data[:, 3].astype(np.float64)
        self.expo = expo_t * np.power(10.0, gain / 20.0)

        self.state_id = self.__add_seq_state_label(
            expo_t, gain, self.seq_state)

        assert np.unique(
            self.state_id).shape[0] == seq_state.shape[0], "images has unlabelled states"

        # filter first and last images by state id
        idx_s = 0
        idx_e = self.state_id.shape[0] - 1
        while (self.state_id[idx_s] != 0):
            idx_s = idx_s + 1
        while (self.state_id[idx_e] != self.seq_state.shape[0] - 1):
            idx_e = idx_e - 1

        self.image_id = self.image_id[idx_s:(idx_e + 1)]
        self.expo = self.expo[idx_s:(idx_e + 1)]
        self.state_id = self.state_id[idx_s:(idx_e + 1)]

        # map from bracket_id 2 image_id of the first image
        self.bracket_map = np.where(self.state_id == 0)[0]

        # image name list
        self.img_names = self.__generate_image_path(str_cam_filepath)
        self.expo_base = self.__load_expo_from_file(str_expo_filepath)

        self.img_h = img_h
        self.img_w = img_w
        self.images = np.zeros(
            [self.image_id.shape[0], self.img_h, self.img_w], np.uint16)
        for i, img_name in enumerate(self.img_names):
            img = cv2.imread(img_name, cv2.CV_16UC1)
            img_resize = cv2.resize(img, (self.img_w, self.img_h))
            self.images[i, :, :] = img_resize

    def get_bracket_size(self):
        return self.bracket_map.shape[0]

    def get_base_image_by_target_expo(self, bracket_id, expo):
        if bracket_id < 0 or bracket_id >= self.get_bracket_size():
            return True, None, None

        idx_s = self.bracket_map[bracket_id]

        if bracket_id + 1 >= self.get_bracket_size():
            expo_tmp = self.expo[idx_s:]
        else:
            idx_e = self.bracket_map[bracket_id + 1]
            expo_tmp = self.expo[idx_s:idx_e]

        idx_in_bracket = np.where(expo >= expo_tmp)[0][-1]
        idx = self.bracket_map[bracket_id] + idx_in_bracket
        expo_base = expo_tmp[idx_in_bracket]

        return False, self.images[idx, :, :], expo_base

    def __add_seq_state_label(self, seq_expo_t, seq_gain, seq_state):
        seq_state_id = -np.ones(seq_expo_t.shape[0], np.uint8)
        for i, state in enumerate(seq_state):
            expo_t, gain = state[0], state[1]
            idx = np.where((seq_expo_t == expo_t) & (seq_gain == gain))
            seq_state_id[idx] = i
        return seq_state_id

    def __generate_image_path(self, str_cam_filepath):
        folder_name, _ = os.path.splitext(str_cam_filepath)
        img_names = []
        for img_id in self.image_id:
            img_name = os.path.join(folder_name, f"{img_id:08d}.tif")
            img_names.append(img_name)
        return img_names

    def __load_expo_from_file(self, str_expo_filepath):
        return np.loadtxt(str_expo_filepath)


class RewardStat():
    def __init__(self, params):
        self.mean_target = params['rwd_mean_target']
        self.w_flk = params['rwd_w_flk']

    def calc_reward(self, state):
        img1 = np.squeeze(state[-1, :, :]).astype(np.float64)
        img0 = np.squeeze(state[-2, :, :]).astype(np.float64)
        r_mean = self.__calc_reward_mean(img1)
        r_flk = self.w_flk * self.__calc_reward_flk(img0, img1)
        return r_mean, r_flk

    def __calc_reward_mean(self, img1):
        return - np.power(np.abs(np.mean(img1) - self.mean_target), 0.5)

    def __calc_reward_flk(self, img0, img1):
        return - np.power(np.abs(np.mean(img1) - np.mean(img0)), 0.5)


class RewardFeat():
    def __init__(self, params):
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.w_match = params['rwd_w_match']
        self.w_detect = params['rwd_w_detect']

    def calc_reward(self, state):
        # get images
        img1 = (np.squeeze(state[-1, :, :]) * 255).astype(np.uint8)
        img0 = (np.squeeze(state[-2, :, :]) * 255).astype(np.uint8)
        # calculate orb
        kp1, des1 = self.orb.detectAndCompute(img1, None)
        kp0, des0 = self.orb.detectAndCompute(img0, None)
        # calculate reward feature detection
        n_detect = len(kp1)
        # print(f"len(kp1): {len(kp1)}, len(kp0): {len(kp0)}")
        # calculate matches
        if len(kp1) == 0 or len(kp0) == 0:
            n_match = 0
        else:
            matches = self.bf.match(des1, des0)
            if len(matches) <= 4:
                n_match = len(matches)
            else:
                pts1 = np.zeros((len(matches), 2), dtype=np.float32)
                pts0 = np.zeros((len(matches), 2), dtype=np.float32)
                for i, match in enumerate(matches):
                    pts1[i, :] = kp1[match.queryIdx].pt
                    pts0[i, :] = kp0[match.trainIdx].pt
                _, mask = cv2.findHomography(pts1, pts0, cv2.RANSAC)
                matches_mask = mask.ravel().tolist()
                n_match = np.sum(matches_mask)

        r_detect = self.w_detect * n_detect
        r_match = self.w_match * n_match
        # print(r_detect, r_match)

        # # feature matching
        # image_matches = cv2.drawMatches(img1, kp1, img0, kp0, inliers_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # plt.figure(figsize=(15, 10))
        # plt.imshow(image_matches)
        # plt.title('ORB Feature Matches After RANSAC')
        # plt.axis('off')

        # # 显示结果
        # image_with_keypoints1 = cv2.drawKeypoints(img1, kp1, None, color=(0, 255, 0))
        # plt.figure(figsize=(10, 6))
        # plt.imshow(image_with_keypoints1, cmap='gray')
        # plt.title('ORB Keypoints')
        # plt.axis('off')

        # #
        # image_with_keypoints0 = cv2.drawKeypoints(img0, kp0, None, color=(0, 255, 0))
        # plt.figure(figsize=(10, 6))
        # plt.imshow(image_with_keypoints0, cmap='gray')
        # plt.title('ORB Keypoints')
        # plt.axis('off')
        # plt.show()

        return r_detect, r_match


class ExposureEnv():
    def __init__(self, log, params, str_crf_filepath, len_episode):
        # parameters
        self.log = log
        self.len_episode = len_episode
        self.n_img_input = params['state_dim'][0]
        self.b_test = params['env_mode_test']
        self.img_w = params['state_dim'][2]
        self.img_h = params['state_dim'][1]
        self.seqs = []
        self.cur_seq = -1
        self.cur_frame = -1
        self.episode_count = 0
        self.expo_lb = params['env_expo_lb']
        self.expo_ub = params['env_expo_ub']
        self.expo = params['env_expo_init']
        self.state_ori = None
        self.state_out = None
        self.b_data_argumentation = params['env_data_argumentation']
        self.b_time_reverse = False
        self.b_flip_horizontal = False
        self.b_flip_vertical = False
        self.b_time_acc = False
        self.time_acc_ratio = 1

        # simulator
        self.simulator = PhotometricSimulator(str_crf_filepath)

        # reward function
        if params['rwd_mode'] == "stat":
            self.reward_cal = RewardStat(params)
        if params['rwd_mode'] == "feat":
            self.reward_cal = RewardFeat(params)

        # add sequences and states from yaml file
        with open(params['env_seq_filepath'], 'r') as f:
            config = yaml.safe_load(f)
        # add states
        states = []
        for state in config['states']:
            states.append([state['expo_t'], state['gain']])
            if not params['env_mode_test']:
                log.add_log(
                    f"[Env]: \texpo time: {state['expo_t']}, gain: {state['gain']}")
        states = np.array(states, np.float64)
        # add sequences
        seqs_root = config['seqs_root']
        for seq_config in config['sequences']:
            filepath_img = os.path.join(
                seqs_root, seq_config['seq_name'], "cam0.txt")
            filepath_expo_ref = os.path.join(
                seqs_root, seq_config['seq_name'], "expo_ref.txt")
            seq = Sequence(filepath_img, filepath_expo_ref, states,
                           params['env_img_ori_w'], params['env_img_ori_h'])
            self.add_sequence(seq)
            if not params['env_mode_test']:
                log.add_log(
                    f"[Env]: add sequence: {os.path.join(seqs_root, seq_config['seq_name'])}")

        if not params['env_mode_test']:
            log.save_buffer_to_file()

    def reset(self, seq_id=None, frame_id=None, expo=None):
        # data augmentation
        if not self.b_test and self.b_data_argumentation:
            self.b_time_reverse = np.random.randint(0, 2, dtype=bool)
            self.b_flip_horizontal = np.random.randint(0, 2, dtype=bool)
            self.b_flip_vertical = np.random.randint(0, 2, dtype=bool)
            self.b_time_acc = np.random.randint(0, 2, dtype=bool)
            if self.b_time_acc:
                self.time_acc_ratio = np.random.randint(1, 4)
            else:
                self.time_acc_ratio = 1

        # random set & check
        if seq_id == None:
            seq_id = np.random.randint(0, len(self.seqs))
        else:
            assert seq_id >= 0 and seq_id < len(
                self.seqs), f"input seq_id out of range, range [0, {len(self.seqs)}) but get {seq_id}"

        episode_frame_size = self.len_episode * self.time_acc_ratio
        if frame_id == None:
            if not self.b_test:
                if self.b_time_reverse:
                    frame_id = np.random.randint(
                        episode_frame_size, self.seqs[seq_id].get_bracket_size() - 1)
                else:
                    frame_id = np.random.randint(
                        0, self.seqs[seq_id].get_bracket_size() - episode_frame_size)
            else:
                frame_id = np.random.randint(
                    0, self.seqs[seq_id].get_bracket_size())
        else:
            assert frame_id >= 0 and frame_id < self.seqs[seq_id].get_bracket_size(
            ), f"input frame_id out of range, range [0, {self.seqs[seq_id].get_bracket_size()}) but get {frame_id}"

        if expo == None:
            expo = self.seqs[seq_id].expo_base[frame_id]
            if not self.b_test:
                ratio = np.random.random() * 2 - 1.0
                expo = expo * ratio

            if expo < self.expo_lb:
                expo = self.expo_lb
            if expo > self.expo_ub:
                expo = self.expo_ub

        # init global variables
        self.cur_seq = seq_id
        self.cur_frame = frame_id
        self.expo = expo
        self.episode_count = 0

        # init update
        self.state_ori = None
        self.state_out = None
        self.update_state()

        self.episode_count = self.episode_count + 1
        if self.b_time_reverse:
            self.cur_frame = self.cur_frame - 1 * self.time_acc_ratio
        else:
            self.cur_frame = self.cur_frame + 1 * self.time_acc_ratio

        return self.state_out, self.expo

    def step(self, action):
        ev = action * 2
        self.expo = self.expo * np.power(2, ev)
        # print(self.expo)

        # check boundary
        if self.expo < self.expo_lb:
            self.expo = self.expo_lb
        if self.expo > self.expo_ub:
            self.expo = self.expo_ub

        # update state
        self.update_state()

        # update reward
        r_1, r_2 = self.reward_cal.calc_reward(self.state_ori)
        reward = r_1 + r_2

        if not self.b_test:
            self.log.add_log("[Env]: step")
            self.log.add_log(
                f"[Env]: episode count: {self.episode_count}, frame_id: {self.cur_frame}, expo: {self.expo}")
            self.log.add_log(f"[Env]: action: {action}")
            self.log.save_buffer_to_file()
        if self.b_test:
            print(
                f"[Env]: episode count: {self.episode_count}, frame_id: {self.cur_frame}, expo: {self.expo}")
            print(f"[Env]: action: {action}")
            print(f"[Env]: r: {reward}, r_1: {r_1}, r_2: {r_2}")
            int_avg = np.mean(self.state_ori[-1, :, :])
            print(
                f"[Env]: avg intensity: {int_avg * 255}, avg signal: {int_avg}")

        self.episode_count = self.episode_count + 1
        if self.b_time_reverse:
            self.cur_frame = self.cur_frame - 1 * self.time_acc_ratio
        else:
            self.cur_frame = self.cur_frame + 1 * self.time_acc_ratio

        if self.episode_count >= self.len_episode:
            return self.state_out, reward, True

        if self.b_time_reverse:
            if self.cur_frame < 0:
                return self.state_out, reward, True
        else:
            if self.cur_frame >= self.seqs[self.cur_seq].get_bracket_size():
                return self.state_out, reward, True

        return self.state_out, reward, False

    def update_state(self):
        # process first frame
        seq = self.seqs[self.cur_seq]
        out_of_range, img_base, expo_base = seq.get_base_image_by_target_expo(
            self.cur_frame, self.expo)
        if out_of_range:
            print(
                f"out_of_range: {self.cur_frame} of {self.seqs[self.cur_seq].get_bracket_size()}")
            return False

        # synthesis
        img_syn = self.simulator.img_synthesis(img_base, expo_base, self.expo)
        if self.b_flip_horizontal:
            img_syn = img_syn[:, ::-1]
        if self.b_flip_vertical:
            img_syn = img_syn[::-1, :]

        img_syn_resize = cv2.resize(img_syn, (self.img_w, self.img_h))

        if self.state_ori is None:
            self.state_ori = np.tile(img_syn, (self.n_img_input, 1, 1))
            self.state_out = np.tile(img_syn_resize, (self.n_img_input, 1, 1))
        else:
            self.state_ori = self.state_ori[1:, :, :]
            self.state_ori = np.concatenate(
                (self.state_ori, np.expand_dims(img_syn, axis=0)), axis=0)
            self.state_out = self.state_out[1:, :, :]
            self.state_out = np.concatenate(
                (self.state_out, np.expand_dims(img_syn_resize, axis=0)), axis=0)
        return True

    def random_action(self):
        return np.random.random() * 2 - 1.0

    def render(self, wait_ms=10):
        img_cur = (np.squeeze(self.state_ori[-1, :, :]) * 255).astype(np.uint8)
        cv2.imshow("render", img_cur)
        cv2.waitKey(wait_ms)

    def add_sequence(self, sequence):
        self.seqs.append(sequence)

    def show_info(self):
        print(f"cur_seq: {self.cur_seq}")
        print(
            f"cur_frame: {self.cur_frame}/{self.seqs[self.cur_seq].get_bracket_size()}")
        print(f"expo: {self.expo}")
        print(f"episode_count: {self.episode_count}")
        print(f"time_reverse: {self.b_time_reverse}, flip_horizontal: {self.b_flip_horizontal}, flip_vertical: {self.b_flip_vertical}, time_acc_ratio: {self.time_acc_ratio}")
        print("----------------------------------------")


class ReplayBuffer(object):
    def __init__(self, params):
        self.state_dim = params['state_dim']
        self.action_dim = params['action_dim']
        self.max_size = params['rb_buffer_size']
        self.count = 0
        self.size = 0
        self.img_d, self.img_w, self.img_h = self.state_dim
        self.s = np.zeros((self.max_size, self.img_d, self.img_w, self.img_h))
        self.a = np.zeros((self.max_size, self.action_dim))
        self.r = np.zeros((self.max_size, 1))
        self.s_ = np.zeros((self.max_size, self.img_d, self.img_w, self.img_h))
        self.dw = np.zeros((self.max_size, 1))

    def store(self, s, a, r, s_, dw):
        self.s[self.count] = s
        self.a[self.count] = a
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.dw[self.count] = dw
        self.count = (self.count + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        index = np.random.choice(self.size, size=batch_size)
        batch_s = torch.tensor(self.s[index], dtype=torch.float)
        batch_a = torch.tensor(self.a[index], dtype=torch.float)
        batch_r = torch.tensor(self.r[index], dtype=torch.float)
        batch_s_ = torch.tensor(self.s_[index], dtype=torch.float)
        batch_dw = torch.tensor(self.dw[index], dtype=torch.float)
        return batch_s, batch_a, batch_r, batch_s_, batch_dw

    def all(self):
        batch_s = torch.tensor(self.s[:self.size], dtype=torch.float)
        batch_a = torch.tensor(self.a[:self.size], dtype=torch.float)
        batch_r = torch.tensor(self.r[:self.size], dtype=torch.float)
        batch_s_ = torch.tensor(self.s_[:self.size], dtype=torch.float)
        batch_dw = torch.tensor(self.dw[:self.size], dtype=torch.float)
        return batch_s, batch_a, batch_r, batch_s_, batch_dw

    def len(self):
        return self.size

    def clear(self):
        self.count = 0
        self.size = 0
        self.s = np.zeros((self.max_size, self.img_d, self.img_w, self.img_h))
        self.a = np.zeros((self.max_size, self.action_dim))
        self.r = np.zeros((self.max_size, 1))
        self.s_ = np.zeros((self.max_size, self.img_d, self.img_w, self.img_h))
        self.dw = np.zeros((self.max_size, 1))
