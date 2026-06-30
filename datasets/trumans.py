import numpy as np
import os
import random

class TrumansData():
    def __init__(self, cfg):
        data_root = cfg.data_root
        self.human_motion_data_path = os.path.join(data_root, 'trumans/subject/')
        self.object_motion_data_path = os.path.join(data_root, 'trumans/object/object/')
        self.scene_path = os.path.join(data_root, 'trumans/object/scene/')
        self.voxel_size = cfg.voxel_size
        self.grid_size = cfg.grid_size
        self.max_frames = cfg.max_frames
        self.max_object_points = cfg.max_object_points
    

    def get_data_by_name(self, name, start_frame, augment=False):

        text = name.split('_')[1]

        human_motion = np.load(os.path.join(self.human_motion_data_path, name + '.npy'))
        frames_motion = human_motion.shape[0]

        assert frames_motion >= self.max_frames, 'The frames of motion are not enough!'
        assert start_frame <= frames_motion-self.max_frames, 'The start frame is too large!'

        

        object_motion = np.load(os.path.join(self.object_motion_data_path, name + '.npy'))
        scene_motion = np.load(os.path.join(self.scene_path, name + '.npy'))
        if scene_motion[:,:,1].min() == 0:
            human_motion[:,:,1] += 1

        if augment:
            theta = np.random.uniform(0, 2 * np.pi)
            human_motion = self.random_rotate_around_y(human_motion, theta)
            object_motion = self.random_rotate_around_y(object_motion, theta)
            scene_motion = self.random_rotate_around_y(scene_motion, theta)
        

        human_motion = human_motion[start_frame:start_frame+self.max_frames]
        object_motion = object_motion[start_frame:start_frame+self.max_frames]

        origin = human_motion[0:1,0:1]
        human_motion = human_motion - origin
        object_motion = object_motion - origin
        scene_motion = scene_motion - origin


        object_voxel_grid = self.get_object_voxel_grid(object_motion)
        object_voxel_grid = self.get_scene_voxel_grid(scene_motion, object_voxel_grid)
        human_voxel_index, valid_mask = self.get_human_voxel_grid(human_motion)

        scene_motion = scene_motion.repeat(self.max_frames, axis=0)
        object_motion = np.concatenate([object_motion, scene_motion], axis=1)

        if object_motion.shape[1] > self.max_object_points:
            idx = np.random.choice(object_motion.shape[1], self.max_object_points, replace=False)
            object_motion = object_motion[:,idx]
        else:
            object_motion = np.concatenate([object_motion,
                                     np.zeros((object_motion.shape[0], self.max_object_points-object_motion.shape[1], object_motion.shape[2]))
                                     ], axis=1)

        

        return human_voxel_index, object_voxel_grid, object_motion, valid_mask, text
    

    def get_object_voxel_grid(self, object_motion):

        half_grid_size = self.grid_size / 2

        voxel_grid = np.zeros((self.max_frames, *self.grid_size, 3))

        min_bound = -1 * half_grid_size * self.voxel_size

        voxel_indices = ((object_motion.reshape(-1,3) - min_bound) / self.voxel_size).round().astype(int).reshape(self.max_frames,-1,3)

        for t in range(self.max_frames):
            voxel_indices_tmp = voxel_indices[t]
            valid_mask = np.all((voxel_indices_tmp >= 0) & (voxel_indices_tmp < self.grid_size), axis=1)
            valid_voxels = voxel_indices_tmp[valid_mask]
            voxel_grid[t, valid_voxels[:,0], valid_voxels[:,1], valid_voxels[:,2]] = np.array([0,1,0])
        
        return voxel_grid

    
    def get_scene_voxel_grid(self, scene_motion, voxel_grid=None):

        half_grid_size = self.grid_size / 2

        if voxel_grid is None:
            voxel_grid = np.zeros((self.max_frames, *self.grid_size, 3))

        min_bound = -1 * half_grid_size * self.voxel_size

        voxel_indices = ((scene_motion.reshape(-1,3) - min_bound) / self.voxel_size).round().astype(int).reshape(-1,3)

        voxel_indices_tmp = voxel_indices
        valid_mask = np.all((voxel_indices_tmp >= 0) & (voxel_indices_tmp < self.grid_size), axis=1)
        valid_voxels = voxel_indices_tmp[valid_mask]
        voxel_grid[:, valid_voxels[:,0], valid_voxels[:,1], valid_voxels[:,2]] = np.array([0,0,1])
        
        return voxel_grid


    def get_human_voxel_grid(self, human_motion):

        T, J, D = human_motion.shape

        half_grid_size = self.grid_size / 2

        min_bound = -1 * half_grid_size * self.voxel_size

        voxel_indices = ((human_motion.reshape(-1,3) - min_bound) / self.voxel_size).reshape(self.max_frames,-1,3)
        valid_mask = np.all((voxel_indices >= 0) & (voxel_indices <= self.grid_size-1), axis=2)
      
        return voxel_indices, valid_mask
    

    def random_rotate_around_y(self, motion, theta):
        T, N, _ = motion.shape
        
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        rotation_matrix = np.array([
            [cos_theta, 0, sin_theta],
            [0, 1, 0],
            [-sin_theta, 0, cos_theta]
        ])
        
        motion_reshaped = motion.reshape(-1, 3)
        rotated_motion = np.dot(motion_reshaped, rotation_matrix.T)
        rotated_motion = rotated_motion.reshape(T, N, 3)
        
        return rotated_motion
