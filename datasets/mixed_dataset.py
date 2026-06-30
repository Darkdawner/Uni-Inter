import torch
from torch.utils.data import Dataset
import os
import json
from .behave import BehaveData
from .ntu import NTUData
from .chi3d import Chi3dData
from .trumans import TrumansData
from .manip import ManipData


SPLIT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)


class MixedDataset(Dataset):
    def __init__(self, cfg, mode='train'):
        suffix = 'train' if mode == 'train' else 'test'

        self.dbs_behave = json.load(open(os.path.join(SPLIT_DIR, f'data_splits/behave_{suffix}.json')))
        self.dbs_ntu = json.load(open(os.path.join(SPLIT_DIR, f'data_splits/ntu_{suffix}.json')))
        self.dbs_chi3d = json.load(open(os.path.join(SPLIT_DIR, f'data_splits/chi3d_{suffix}.json')))
        self.dbs_trumans = json.load(open(os.path.join(SPLIT_DIR, f'data_splits/trumans_{suffix}.json')))
        self.dbs_manip = json.load(open(os.path.join(SPLIT_DIR, f'data_splits/manip_{suffix}.json')))

        self.max_length = cfg.max_frames
        self.mode = mode

        self.BehaveData = BehaveData(cfg)
        self.NTUData = NTUData(cfg, mode=mode)
        self.Chi3dData = Chi3dData(cfg, mode=mode)
        self.TrumansData = TrumansData(cfg)
        self.ManipData = ManipData(cfg, mode=mode)

        # Weighted sampling: oversample small datasets for balance
        self.dbs = (
            self.dbs_manip
            + self.dbs_behave
            + self.dbs_ntu * 3
            + self.dbs_chi3d * 3
            + self.dbs_trumans * 30
        )
    
    def __len__(self):
        return len(self.dbs)
    
    def __getitem__(self, idx):
        db = self.dbs[idx]
        name = db['name']
        dataset_class = db['dataset_class']
        sequence_length = db['sequence_length']

        if dataset_class in ('behave', 'trumans', 'manip'):
            start_frame = 0
        else:
            start_frame = sequence_length // 2 - self.max_length // 2

        augment = (self.mode == 'train')

        if dataset_class == 'behave':
            human_voxel_index, cond_voxel_grid, object_motion, valid_mask, text = self.BehaveData.get_data_by_name(name, start_frame, augment=augment)
        elif dataset_class == 'ntu':
            human_voxel_index, cond_voxel_grid, object_motion, valid_mask, text = self.NTUData.get_data_by_name(name, start_frame, augment=augment)
        elif dataset_class == 'chi3d':
            human_voxel_index, cond_voxel_grid, object_motion, valid_mask, text = self.Chi3dData.get_data_by_name(name, start_frame, augment=augment)
        elif dataset_class == 'trumans':
            human_voxel_index, cond_voxel_grid, object_motion, valid_mask, text = self.TrumansData.get_data_by_name(name, start_frame, augment=augment)
        elif dataset_class == 'manip':
            human_voxel_index, cond_voxel_grid, object_motion, valid_mask, text = self.ManipData.get_data_by_name(name, start_frame, augment=augment)

        human_voxel_index = torch.from_numpy(human_voxel_index).float()
        cond_voxel_grid = torch.from_numpy(cond_voxel_grid).float()

        meta_data = {
            'human_voxel_index': human_voxel_index,
            'cond_voxel_grid': cond_voxel_grid,
            'object_motion': object_motion,
            'valid_mask': valid_mask,
            'text': text,
            'name': str(name),
        }

        return meta_data
