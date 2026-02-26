import torch
from torch.utils import data
import numpy as np
import os
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
# import spacy

from torch.utils.data._utils.collate import default_collate
from data_loaders.humanml.utils.word_vectorizer import WordVectorizer
from data_loaders.humanml.utils.get_opt import get_opt
from ..scripts.motion_process import recover_root_rot_pos, recover_from_ric
from data_loaders.humanml.utils.metrics import cross_combination_joints
# import spacy

def collate_fn(batch):
    if batch[0][-1] is None:
        batch = [b[:-1] for b in batch]
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)



class TextOnlyDataset(data.Dataset):
    def __init__(self, opt, mean, std, split_file):
        self.mean = mean
        self.std = std
        self.opt = opt
        self.data_dict = []
        self.max_length = 20
        self.pointer = 0
        self.fixed_length = 120


        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        # id_list = id_list[:200]

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'text':[text_dict]}
                                new_name_list.append(new_name)
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'text': text_data}
                    new_name_list.append(name)
            except:
                pass

        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = new_name_list

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        text_list = data['text']

        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']
        return None, None, caption, None, np.array([0]), self.fixed_length, None, None


# A wrapper class for t2m original dataset for MDM purposes
class HumanML3D(data.Dataset):
    def __init__(self, mode, datapath='./dataset/mpl_opt.txt', split="train", control_joint=0, density=100, **kwargs):
        self.mode = mode
        
        self.dataset_name = 'mpl'
        self.dataname = 'mpl'

        # Configurations of T2M dataset and KIT dataset is almost the same
        abs_base_path = f'.'
        dataset_opt_path = pjoin(abs_base_path, datapath)
        device = None  # torch.device('cuda:4') # This param is not in use in this context
        opt = get_opt(dataset_opt_path, device)
        opt.meta_dir = pjoin(abs_base_path, opt.meta_dir)
        opt.motion_dir = pjoin(abs_base_path, opt.motion_dir)
        opt.text_dir = pjoin(abs_base_path, opt.text_dir)
        opt.model_dir = pjoin(abs_base_path, opt.model_dir)
        opt.checkpoints_dir = pjoin(abs_base_path, opt.checkpoints_dir)
        opt.data_root = pjoin(abs_base_path, opt.data_root)
        opt.save_root = pjoin(abs_base_path, opt.save_root)
        opt.meta_dir = './dataset'
        self.opt = opt
        print('Loading dataset %s ...' % opt.dataset_name)

        if mode == 'gt':
            self.mean = np.load(pjoin(opt.data_root, 'Mean_global.npy'))
            self.std = np.load(pjoin(opt.data_root, 'Std_global.npy'))
        elif mode in ['train', 'eval', 'text_only']:
            self.mean = np.load(pjoin(opt.data_root, 'Mean_global.npy'))
            self.std = np.load(pjoin(opt.data_root, 'Std_global.npy'))

        if mode == 'eval':
            self.mean_for_eval = np.load(pjoin(opt.meta_dir, 'mean_global.npy'))
            self.std_for_eval = np.load(pjoin(opt.meta_dir, 'std_global.npy'))

        self.split_file = pjoin(opt.data_root, f'{split}.txt')
        if mode == 'text_only':
            self.t2m_dataset = TextOnlyDataset(self.opt, self.mean, self.std, self.split_file)
        else:
            self.w_vectorizer = WordVectorizer(pjoin(abs_base_path, 'glove'), 'our_vab')
            self.t2m_dataset = MPLDataset(opt, self.mean, self.std, self.split_file, opt.dataset_name,
                                          opt.motion_dir, opt.text_dir, opt.pressure_dir, opt.joints_dir,
                                          opt.unit_length, opt.max_motion_length, opt.max_text_len, self.w_vectorizer)
            self.num_actions = 1 # dummy placeholder

        assert len(self.t2m_dataset) > 1, 'You loaded an empty dataset, ' \
                                          'it is probably because your data dir has only texts and no motions.\n' \
                                          'To train and evaluate MDM you should get the FULL data as described ' \
                                          'in the README file.'

    def __getitem__(self, item):
        return self.t2m_dataset.__getitem__(item)

    def __len__(self):
        return self.t2m_dataset.__len__()



class MPLDataset(data.Dataset):
    def __init__(self, opt, mean, std, split_file, dataset_name, motion_dir, text_dir, pressure_dir, joints_dir,  unit_length, max_motion_length,
                 max_text_length, w_vectorizer, training_stage='stage2', augmentation_scale=5, evaluation=False):
        # self.times = times
        self.training_stage = training_stage
        self.augmentation_scale = augmentation_scale
        self.evaluation = evaluation
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = max_motion_length
        self.max_text_len = max_text_length
        self.unit_length = unit_length

        self.w_vectorizer = w_vectorizer
        min_motion_len = 40
        joints_num = 22
        new_name_choices = 'ABCDEFGHIJKLMNOPQRSTUVW'
        data_dict = {}

        spatial_norm_path = './dataset/mpl_spatial_norm'

        self.raw_mean = np.load(pjoin(spatial_norm_path, 'Mean_raw.npy'))
        self.raw_std = np.load(pjoin(spatial_norm_path, 'Std_raw.npy'))

        id_list = []
        # self.raw_mean = mean
        # self.raw_std = std

        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []

        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(motion_dir, name + '.npy'), mmap_mode='r')

                if (len(motion)) < min_motion_len or (len(motion) > 160):
                    continue
                text_data = []
                with cs.open(pjoin(text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens

                        text_data.append(text_dict)

                pressure_data = np.load(pjoin(pressure_dir, name + '.npz'), mmap_mode='r')['pressure']
                
                data_dict[name] = {'motion': motion,
                                    'length': len(motion),
                                    "text": text_data,
                                    'pressure': pressure_data
                                    }
                new_name_list.append(name)
                length_list.append(len(motion))
            except Exception as e:
                print(f"Error processing {name}: {e}")
                continue

        print("Length of dataset:", len(data_dict))


        # root_rot_velocity (B, seq_len, 1)
        std[0:1] = std[0:1] / opt.feat_bias
        # root_linear_velocity (B, seq_len, 2)
        std[1:3] = std[1:3] / opt.feat_bias
        # root_y (B, seq_len, 1)
        std[3:4] = std[3:4] / opt.feat_bias
        # ric_data (B, seq_len, (joint_num - 1)*3)
        std[4: 4 + (joints_num - 1) * 3] = std[4: 4 + (joints_num - 1) * 3] / 1.0
        # rot_data (B, seq_len, (joint_num - 1)*6)
        std[4 + (joints_num - 1) * 3: 4 + (joints_num - 1) * 9] = std[4 + (joints_num - 1) * 3: 4 + (
                    joints_num - 1) * 9] / 1.0
        # local_velocity (B, seq_len, joint_num*3)
        std[4 + (joints_num - 1) * 9: 4 + (joints_num - 1) * 9 + joints_num * 3] = std[
                                                                                    4 + (joints_num - 1) * 9: 4 + (
                                                                                                joints_num - 1) * 9 + joints_num * 3] / 1.0
        # foot contact (B, seq_len, 4)
        std[4 + (joints_num - 1) * 9 + joints_num * 3:] = std[
                                                            4 + (joints_num - 1) * 9 + joints_num * 3:] / opt.feat_bias

        assert 4 + (joints_num - 1) * 9 + joints_num * 3 + 4 == mean.shape[-1]
        np.save(pjoin(opt.meta_dir, 'mean_global.npy'), mean)
        np.save(pjoin(opt.meta_dir, 'std_global.npy'), std)

        if self.evaluation:
            # self.w_vectorizer = GloVe('./glove', 'our_vab')
            name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
        else:
            name_list, length_list = new_name_list, length_list
        
        self.mean = mean
        self.std = std

        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        if self.evaluation:
            self.reset_max_len(self.max_length)
        
    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def transform(self, data, mean=None, std=None):
        if mean is None and std is None:
            return (data - self.mean) / self.std
        else:
            return (data - mean) / std

    def inv_transform(self, data, mean=None, std=None):
        if mean is None and std is None:
            return data * self.std + self.mean
        else:
            return data * std + mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = (self.pointer + item) % len(self.data_dict)
        name = self.name_list[idx]
        data = self.data_dict[name]

        motion, m_length, text_list, pressure = data['motion'], data['length'], data['text'], data['pressure']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']


        if len(tokens) < self.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        if self.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.unit_length) * self.unit_length

        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length] # mlength,263
        pressure = pressure[idx:idx+m_length+1] # mlength+1,160,120

        joints = recover_from_ric(torch.from_numpy(motion.copy()).float(), 22)
        joints = joints.numpy()
        mask_seq = np.zeros((joints.shape[0], 22, 3)).astype(np.bool)
        choose_joint = [7, 8, 10, 11]
        for cj in choose_joint:
            mask_seq[:, cj] = True
        joints = (joints - self.raw_mean.reshape(22, 3)) / self.raw_std.reshape(22, 3)
        feet_hint = joints * mask_seq
        feet_hint = feet_hint.reshape(feet_hint.shape[0], -1)
        "Z Normalization"

        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                    np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                    ], axis=0) # 160, 263
            
            pressure = np.concatenate([pressure,
                                    np.zeros((self.max_motion_length - m_length, pressure.shape[1], pressure.shape[2]))
                                    ], axis=0) # 161, 160, 120
            feet_hint = np.concatenate([feet_hint,
                                    np.zeros((self.max_motion_length - m_length, feet_hint.shape[1]))
                                    ], axis=0)
            joints = np.concatenate([joints,
                                    np.zeros((self.max_motion_length - m_length, joints.shape[1], joints.shape[2]))
                                    ], axis=0)
        elif m_length > self.max_motion_length:
            if not self.evaluation:
                idx = random.randint(0, self.max_motion_length - m_length)
                motion = motion[idx:idx + self.max_motion_length]
                pressure = pressure[idx:idx + self.max_motion_length + 1]

        return word_embeddings, pos_one_hots, caption, sent_len, pressure, motion, m_length, '_'.join(tokens), feet_hint, joints