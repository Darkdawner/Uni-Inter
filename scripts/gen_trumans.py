"""Generate predictions for TRUMANS evaluation."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from configs import cfg
from models import Diffusion, UNet4D
from datasets import TrumansData
from utils import load_state
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import json
import clip
import numpy as np


class TrumansTestDataset(Dataset):
    def __init__(self, cfg):
        self.dbs = json.load(open(os.path.join(
            os.path.dirname(__file__), os.pardir, 'data_splits/trumans_test.json'
        )))
        self.max_length = cfg.max_frames
        self.TrumansData = TrumansData(cfg)

    def __len__(self):
        return len(self.dbs)

    def __getitem__(self, idx):
        db = self.dbs[idx]
        name = db['name']
        start_frame = 0

        human_voxel_index, cond_voxel_grid, object_motion, valid_mask, text = \
            self.TrumansData.get_data_by_name(name, start_frame, augment=False)

        return {
            'human_voxel_index': torch.from_numpy(human_voxel_index).float(),
            'cond_voxel_grid': torch.from_numpy(cond_voxel_grid).float(),
            'object_motion': object_motion,
            'valid_mask': valid_mask,
            'text': text,
            'name': str(name),
        }


def main():
    test_dataset = TrumansTestDataset(cfg)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=int(cfg.num_threads), drop_last=True
    )

    diffusion = Diffusion(
        noise_steps=cfg.noise_steps,
        motion_size=(40, 22, 48, 48, 48),
        device=cfg.device,
        ddim_timesteps=cfg.ddim_timesteps,
        scheduler=cfg.scheduler
    )

    model = UNet4D(in_channels=40, t_dim=512).to(cfg.device)
    clip_model, _ = clip.load(cfg.clip_model_path, device=cfg.device)

    checkpoint = torch.load(cfg.checkpoint_path, map_location=cfg.device)
    load_state(model, checkpoint)

    output_dir = './eval_trumans/pred/'
    os.makedirs(output_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            human_voxel_index = batch['human_voxel_index'].to(cfg.device)
            cond_voxel_grid = batch['cond_voxel_grid'].to(cfg.device)
            text_data_input = batch['text']
            names = batch['name']

            text = clip.tokenize(text_data_input, truncate=True).to(cfg.device)
            text_features = clip_model.encode_text(text).float()

            predicted_x0 = diffusion.sample_ddim_pred_x0_sum(
                model, cond_voxel_grid, text_features,
                sample_num=cond_voxel_grid.shape[0], cond_out=False
            )

            predicted_x0 = predicted_x0 / predicted_x0.sum(dim=(3, 4, 5), keepdim=True)
            predicted_x0 = predicted_x0.cpu().numpy()

            x_p = np.sum(predicted_x0, axis=(4, 5))
            y_p = np.sum(predicted_x0, axis=(3, 5))
            z_p = np.sum(predicted_x0, axis=(3, 4))

            grid = np.arange(48).reshape(1, 1, 1, -1)
            x = np.sum(x_p * grid, axis=3, keepdims=True)
            y = np.sum(y_p * grid, axis=3, keepdims=True)
            z = np.sum(z_p * grid, axis=3, keepdims=True)

            motion_pred = np.concatenate((x, y, z), axis=3)[0]

            voxel_size = cfg.voxel_size
            min_bound = -1 * (cfg.grid_size / 2) * voxel_size
            motion_pred = (motion_pred.reshape(-1, 3) * voxel_size + min_bound).reshape(cfg.max_frames, -1, 3)

            np.save(os.path.join(output_dir, f'{names[0]}.npy'), motion_pred)


if __name__ == '__main__':
    main()
