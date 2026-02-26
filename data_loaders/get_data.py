# This code is based on https://github.com/GuyTevet/motion-diffusion-model
from torch.utils.data import DataLoader
from data_loaders.tensors import collate as all_collate
from data_loaders.tensors import t2m_collate, mpl_collate

import torch
from torch.utils.data import DistributedSampler as _DistributedSampler
from torch.utils.data import DataLoader

def get_dataset_class(name):
    if name == "humanml" or name == "mpl":
        from data_loaders.humanml.data.dataset import HumanML3D
        return HumanML3D
    elif name == "kit":
        from data_loaders.humanml.data.dataset import KIT
        return KIT
    else:
        raise ValueError(f'Unsupported dataset name [{name}]')

def get_collate_fn(name, hml_mode='train'):
    if hml_mode == 'gt':
        from data_loaders.humanml.data.dataset import collate_fn as t2m_eval_collate
        return t2m_eval_collate
    if name in ["humanml", "kit"]:
        return t2m_collate
    elif name == 'mpl':
        return mpl_collate
    else:
        return all_collate


def get_dataset(name, num_frames, split='train', hml_mode='train', control_joint=0, density=100):
    DATA = get_dataset_class(name)
    dataset = DATA(split=split, num_frames=num_frames, mode=hml_mode, control_joint=control_joint, density=density)
    return dataset


def get_dataset_loader(opt, name, batch_size, num_frames, split='train', hml_mode='train', control_joint=0, density=100):
    dataset = get_dataset(name, num_frames, split, hml_mode, control_joint, density)
    collate = get_collate_fn(name, hml_mode)
    shuffle = False
    sampler = DistributedSampler(
            dataset, opt.world_size, opt.rank, shuffle=shuffle)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, pin_memory=False,
        num_workers=8, drop_last=True, collate_fn=collate,
    )

    return loader

class DistributedSampler(_DistributedSampler):

    def __init__(self,
                 dataset,
                 num_replicas=None,
                 rank=None,
                 shuffle=True,
                 round_up=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle
        self.round_up = round_up
        if self.round_up:
            self.total_size = self.num_samples * self.num_replicas
        else:
            self.total_size = len(self.dataset)

    def __iter__(self):
        # deterministically shuffle based on epoch
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        if self.round_up:
            indices = (
                indices *
                int(self.total_size / len(indices) + 1))[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        if self.round_up:
            assert len(indices) == self.num_samples

        return iter(indices)