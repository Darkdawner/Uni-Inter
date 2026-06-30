from configs import cfg
from models import Diffusion, UNet4D
from utils import make_optimizer, make_lr_scheduler, load_state
from datasets import MixedDataset
from tqdm import tqdm
import clip
import torch
import os
import torch.nn.functional as F
from tensorboardX import SummaryWriter


class GaussianVoxel(torch.nn.Module):
    def __init__(self, sigma=5.0):
        super(GaussianVoxel, self).__init__()
        self.sigma = sigma

    def forward(self, voxel_indices):
        B, T, N, _ = voxel_indices.shape

        x = torch.arange(48, device=voxel_indices.device)
        y = torch.arange(48, device=voxel_indices.device)
        z = torch.arange(48, device=voxel_indices.device)
        grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')

        grid_x = grid_x[None, None, None, ...].repeat(B, T, N, 1, 1, 1)
        grid_y = grid_y[None, None, None, ...].repeat(B, T, N, 1, 1, 1)
        grid_z = grid_z[None, None, None, ...].repeat(B, T, N, 1, 1, 1)

        x_coords = voxel_indices[:, :, :, 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        y_coords = voxel_indices[:, :, :, 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        z_coords = voxel_indices[:, :, :, 2].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        d = (grid_x - x_coords)**2 + (grid_y - y_coords)**2 + (grid_z - z_coords)**2

        voxel_grid = torch.exp(-d / (2 * self.sigma**2))

        return voxel_grid, grid_x, grid_y, grid_z


def motion_to_vec(motion):
    B, T, J, _ = motion.shape
    child_joints = [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
    father_joints = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    motion_vector = motion[:, :, father_joints, :] - motion[:, :, child_joints, :]

    motion_skel_length = torch.norm(motion_vector, dim=-1, keepdim=True)
    normed_motion_vector = motion_vector / (motion_skel_length + 1e-8)

    return normed_motion_vector, motion_skel_length


def main(args):
    args.local_rank = int(os.environ['LOCAL_RANK'])

    torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(args.local_rank)
    cfg.device = torch.device("cuda", args.local_rank)

    train_dataset = MixedDataset(cfg, mode='train')
    dataset_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.batch_size,
        shuffle=(dataset_sampler is None),
        num_workers=int(cfg.num_threads),
        pin_memory=True, sampler=dataset_sampler, drop_last=True
    )

    diffusion = Diffusion(
        noise_steps=cfg.noise_steps,
        motion_size=(40, 22, 48, 48, 48),
        device=cfg.device,
        ddim_timesteps=cfg.ddim_timesteps,
        scheduler=cfg.scheduler
    )

    Gaussian = GaussianVoxel(sigma=3)

    model = UNet4D(in_channels=40, t_dim=512).to(cfg.device)
    optimizer = make_optimizer(cfg, model.parameters())
    scheduler = make_lr_scheduler(cfg, optimizer)

    clip_model, preprocess = clip.load(cfg.clip_model_path, device=cfg.device)

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if torch.cuda.device_count() > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=False
        )

    iteration = 0
    epochId = 0
    writer = SummaryWriter(log_dir='runs')

    model.train()
    if args.local_rank == 0:
        print('---------------------Start Training-----------------------')
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    while True:
        bar_train = tqdm(enumerate(train_loader), total=len(train_loader))
        train_loader.sampler.set_epoch(epochId)

        for i, batch in bar_train:
            human_voxel_index = batch['human_voxel_index'].cuda()
            valid_mask = batch['valid_mask'].cuda()
            human_voxel_grid, grid_x, grid_y, grid_z = Gaussian(human_voxel_index)
            human_voxel_grid = human_voxel_grid / (human_voxel_grid.sum(dim=(3, 4, 5), keepdim=True) + 1e-8)
            cond_voxel_grid = batch['cond_voxel_grid'].cuda()
            text_data_input = batch['text']

            text = clip.tokenize(text_data_input, truncate=True).cuda()
            with torch.no_grad():
                text_features = clip_model.encode_text(text).float()

            iteration += 1
            optimizer.zero_grad()

            t = diffusion.sample_timesteps(human_voxel_grid.shape[0]).cuda()
            x_t, noise = diffusion.noise_motion(human_voxel_grid, t)

            predicted_x0 = model(x_t, cond_voxel_grid, text_features, t)

            # Recover predicted motion from voxel distribution
            x_p = torch.sum(predicted_x0, dim=(4, 5))
            y_p = torch.sum(predicted_x0, dim=(3, 5))
            z_p = torch.sum(predicted_x0, dim=(3, 4))

            x = (x_p * grid_x[:, :, :, :, 0, 0]).sum(dim=3, keepdim=True)
            y = (y_p * grid_y[:, :, :, 0, :, 0]).sum(dim=3, keepdim=True)
            z = (z_p * grid_z[:, :, :, 0, 0, :]).sum(dim=3, keepdim=True)
            predicted_motion = torch.cat([x, y, z], dim=-1)

            valid_mask = valid_mask.unsqueeze(-1)

            # Position loss + velocity loss
            loss_recover = (
                (predicted_motion * valid_mask - human_voxel_index * valid_mask).square().mean()
                + (predicted_motion * valid_mask - human_voxel_index * valid_mask).abs().mean()
                + (((predicted_motion * valid_mask)[:, 1:, :, :] - (predicted_motion * valid_mask)[:, :-1, :, :])
                   - ((human_voxel_index * valid_mask)[:, 1:, :, :] - (human_voxel_index * valid_mask)[:, :-1, :, :])).square().mean()
                + (((predicted_motion * valid_mask)[:, 1:, :, :] - (predicted_motion * valid_mask)[:, :-1, :, :])
                   - ((human_voxel_index * valid_mask)[:, 1:, :, :] - (human_voxel_index * valid_mask)[:, :-1, :, :])).abs().mean()
            )

            # Skeleton vector loss
            pred_normed_motion_vector, pred_motion_skel_length = motion_to_vec(predicted_motion * valid_mask)
            gt_normed_motion_vector, gt_motion_skel_length = motion_to_vec(human_voxel_index * valid_mask)

            loss_recover += (
                (pred_normed_motion_vector - gt_normed_motion_vector).square().mean()
                + (pred_normed_motion_vector - gt_normed_motion_vector).abs().mean()
                + ((pred_normed_motion_vector[:, 1:, :, :] - pred_normed_motion_vector[:, :-1, :, :])
                   - (gt_normed_motion_vector[:, 1:, :, :] - gt_normed_motion_vector[:, :-1, :, :])).square().mean()
                + ((pred_normed_motion_vector[:, 1:, :, :] - pred_normed_motion_vector[:, :-1, :, :])
                   - (gt_normed_motion_vector[:, 1:, :, :] - gt_normed_motion_vector[:, :-1, :, :])).abs().mean()
            )

            loss_recover += (
                (pred_motion_skel_length - gt_motion_skel_length).square().mean()
                + (pred_motion_skel_length - gt_motion_skel_length).abs().mean()
                + ((pred_motion_skel_length[:, 1:, :, :] - pred_motion_skel_length[:, :-1, :, :])
                   - (gt_motion_skel_length[:, 1:, :, :] - gt_motion_skel_length[:, :-1, :, :])).square().mean()
                + ((pred_motion_skel_length[:, 1:, :, :] - pred_motion_skel_length[:, :-1, :, :])
                   - (gt_motion_skel_length[:, 1:, :, :] - gt_motion_skel_length[:, :-1, :, :])).abs().mean()
            )

            # Initial orientation loss
            pelvis_pred = predicted_motion[:, 0, 0, :]
            pelvis_gt = human_voxel_index[:, 0, 0, :]
            left_hip_pred = predicted_motion[:, 0, 1, :]
            right_hip_pred = predicted_motion[:, 0, 2, :]
            left_hip_gt = human_voxel_index[:, 0, 1, :]
            right_hip_gt = human_voxel_index[:, 0, 2, :]

            v1_pred = left_hip_pred - pelvis_pred
            v2_pred = right_hip_pred - pelvis_pred
            v1_gt = left_hip_gt - pelvis_gt
            v2_gt = right_hip_gt - pelvis_gt

            normal_pred = torch.cross(v1_pred, v2_pred, dim=-1)
            normal_gt = torch.cross(v1_gt, v2_gt, dim=-1)

            normal_pred = normal_pred / (torch.norm(normal_pred, dim=-1, keepdim=True) + 1e-8)
            normal_gt = normal_gt / (torch.norm(normal_gt, dim=-1, keepdim=True) + 1e-8)

            loss_normal = (1 - torch.nn.functional.cosine_similarity(normal_pred, normal_gt, dim=-1)).mean()

            # Voxel grid loss
            valid_mask = valid_mask.unsqueeze(-1).unsqueeze(-1)
            loss_voxel_grid = 1000000 * (
                (predicted_x0 * valid_mask - human_voxel_grid * valid_mask).square().mean()
                + (predicted_x0 * valid_mask - human_voxel_grid * valid_mask).abs().mean()
            )

            losses = loss_recover + 10 * loss_normal + loss_voxel_grid

            losses.backward()
            optimizer.step()
            scheduler.step()

            loss_dict = {
                'loss_recover': loss_recover,
                'loss_normal': loss_normal,
                'loss_voxel_grid': loss_voxel_grid,
                'loss_total': losses
            }

            for key in loss_dict:
                writer.add_scalar(key, loss_dict[key].mean(), global_step=iteration)

            lr = optimizer.param_groups[0]["lr"]
            bar_train.set_description(
                f"iter: {iteration} voxel: {loss_voxel_grid.item():.4f} "
                f"normal: {loss_normal.item():.4f} recover: {loss_recover.item():.4f} lr: {lr:.6f}"
            )
            bar_train.refresh()

            if args.local_rank == 0:
                if iteration % cfg.save_freq == 0:
                    snapshot_name = os.path.join(cfg.checkpoint_dir, f'checkpoint_iter_{iteration}.pth')
                    torch.save({'state_dict': model.state_dict()}, snapshot_name)
                    print(f'Saved checkpoint: {snapshot_name}')
                if iteration >= cfg.max_iter:
                    print("\nTraining finished!")
                    return 0
        epochId += 1


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Uni-Inter Training")
    args = parser.parse_args()
    main(args)
