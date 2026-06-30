from configs import cfg
from models import Diffusion, UNet4D
from utils import load_state
from datasets import MixedDataset
from tqdm import tqdm
import torch
import os
import clip
import numpy as np
import matplotlib
matplotlib.use('Agg')

from utils.visualization import plot_3d_motion

kinematic = [[0, 1, 4, 7, 10], [0, 2, 5, 8, 11], [0, 3, 6, 9, 12, 15], [9, 13, 16, 18, 20], [9, 14, 17, 19, 21]]


def main():
    test_dataset = MixedDataset(cfg, mode='test')
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=True,
        num_workers=0, drop_last=True
    )

    diffusion = Diffusion(
        noise_steps=cfg.noise_steps,
        motion_size=(40, 22, 48, 48, 48),
        device=cfg.device,
        ddim_timesteps=cfg.ddim_timesteps,
        scheduler=cfg.scheduler
    )

    model = UNet4D(in_channels=40, t_dim=512).to(cfg.device)
    clip_model, preprocess = clip.load(cfg.clip_model_path, device=cfg.device)

    checkpoint = torch.load(cfg.checkpoint_path, map_location=cfg.device)
    load_state(model, checkpoint)

    model.eval()
    with torch.no_grad():
        bar_test = tqdm(enumerate(test_loader), total=len(test_loader))
        for i, batch in bar_test:
            human_voxel_index = batch['human_voxel_index'].to(cfg.device)
            cond_voxel_grid = batch['cond_voxel_grid'].to(cfg.device)
            object_motion = batch['object_motion'].numpy()
            name = batch['name'][0]
            text_data_input = batch['text']

            text = clip.tokenize(text_data_input, truncate=True).to(cfg.device)
            with torch.no_grad():
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

            grid_x = np.arange(48).reshape(1, 1, 1, -1)
            grid_y = np.arange(48).reshape(1, 1, 1, -1)
            grid_z = np.arange(48).reshape(1, 1, 1, -1)

            x = np.sum(x_p * grid_x, axis=3, keepdims=True)
            y = np.sum(y_p * grid_y, axis=3, keepdims=True)
            z = np.sum(z_p * grid_z, axis=3, keepdims=True)

            motion_gt = human_voxel_index.cpu().numpy()
            motion_pred = np.concatenate((x, y, z), axis=3)

            voxel_size = cfg.voxel_size
            grid_size = cfg.grid_size
            half_grid_size = grid_size / 2
            min_bound = -1 * half_grid_size * voxel_size

            motion_pred = motion_pred[0]
            motion_gt = motion_gt[0]

            motion_pred = (motion_pred.reshape(-1, 3) * voxel_size + min_bound).reshape(cfg.max_frames, -1, 3)
            motion_gt = (motion_gt.reshape(-1, 3) * voxel_size + min_bound).reshape(cfg.max_frames, -1, 3)

            motion_pred = np.concatenate((motion_pred, object_motion[0]), axis=1)
            motion_gt = np.concatenate((motion_gt, object_motion[0]), axis=1)

            # Save visualization
            output_dir = './tmp_results/'
            os.makedirs(output_dir, exist_ok=True)

            video_path = os.path.join(output_dir, f'{name}_pred_top.gif')
            plot_3d_motion(video_path, kinematic, motion_pred, title=text_data_input[0], fps=10, radius=2, view=180)

            video_path = os.path.join(output_dir, f'{name}_pred_front.gif')
            plot_3d_motion(video_path, kinematic, motion_pred, title=text_data_input[0], fps=10, radius=2, view=90)

            video_path = os.path.join(output_dir, f'{name}_gt_top.gif')
            plot_3d_motion(video_path, kinematic, motion_gt, title=text_data_input[0], fps=10, radius=2, view=180)

            video_path = os.path.join(output_dir, f'{name}_gt_front.gif')
            plot_3d_motion(video_path, kinematic, motion_gt, title=text_data_input[0], fps=10, radius=2, view=90)

            print(f'Saved visualization for: {name}')


if __name__ == '__main__':
    main()
