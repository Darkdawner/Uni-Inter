import numpy as np
import os
import torch

class cfg:
    # Voxel representation
    voxel_size = np.array([0.1, 0.05, 0.1])
    grid_size = np.array([48, 48, 48])
    max_object_points = 1000

    # Motion
    max_frames = 40

    # Training
    batch_size = 4
    num_threads = 16
    base_lr = 3e-5
    weight_decay = 0
    warm_up_iters = 0
    warm_up_factor = 0.001
    max_iter = 500000
    save_freq = 3000

    # Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Diffusion
    noise_steps = 1000
    ddim_timesteps = 100
    scheduler = 'Cosine'  # 'Cosine' | 'Linear' | 'Sqrt' | 'Sigmoid'

    # Paths (modify these according to your data location)
    data_root = os.environ.get('UNIINTER_DATA_ROOT', '../../datasets')
    clip_model_path = './ViT-B-32.pt'
    checkpoint_dir = './saved_checkpoints/'
    checkpoint_path = './behave_ntu_chi3d_trumans_manip.pth'
