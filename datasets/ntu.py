import numpy as np
import os
import random

class NTUData():
    def __init__(self, cfg, mode):
        data_root = cfg.data_root
        split = 'train' if mode == 'train' else 'test'
        self.human1_motion_data_path = os.path.join(data_root, f'ntu/{split}/person1/')
        self.human2_motion_data_path = os.path.join(data_root, f'ntu/{split}/person2/')
        self.human1_motion_smpl_data_path = os.path.join(data_root, f'ntu/{split}/person1_smplx/')
        self.human2_motion_smpl_data_path = os.path.join(data_root, f'ntu/{split}/person2_smplx/')

        self.text_dict = {
            0: "punching or slapping other person",
            1: "kicking other person",
            2: "pushing other person",
            3: "pat on back of other person",
            4: "point finger at the other person",
            5: "hugging other person",
            6: "giving something to other person",
            7: "touch other person's pocket",
            8: "handshaking",
            9: "walking towards each other",
            10: "walking apart from each other",
            11: "hit other person with something",
            12: "wield knife towards other person",
            13: "knock over other person (hit with body)",
            14: "grab other person's stuff",
            15: "shoot at other person with a gun",
            16: "step on foot",
            17: "high-five",
            18: "cheers and drink",
            19: "carry something with other person",
            20: "take a photo of other person",
            21: "follow other person",
            22: "whisper in other person's ear",
            23: "exchange things with other person",
            24: "support somebody with hand",
            25: "finger-guessing game (playing rock-paper-scissors)",
        }

        self.voxel_size = cfg.voxel_size
        self.grid_size = cfg.grid_size
        self.max_frames = cfg.max_frames
        self.max_object_points = cfg.max_object_points
        self.mirror_flip = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20]
    
    def get_data_by_name(self, name, start_frame, augment=False):

        text_id = int(name[-3:])-1
        text = self.text_dict[text_id]
        

        human1_motion = np.load(os.path.join(self.human1_motion_data_path, name + '.npy'))
        human2_motion = np.load(os.path.join(self.human2_motion_data_path, name + '.npy'))

        human1_motion = human1_motion[:,self.mirror_flip,:]
        human2_motion = human2_motion[:,self.mirror_flip,:]


        frames_motion = human1_motion.shape[0]
        assert frames_motion >= self.max_frames, 'The frames of motion are not enough!'
        assert start_frame <= frames_motion-self.max_frames, 'The start frame is too large!'

        human1_smpl_motion = np.load(os.path.join(self.human1_motion_smpl_data_path, name + '.npy'))
        human2_smpl_motion = np.load(os.path.join(self.human2_motion_smpl_data_path, name + '.npy'))


        if augment:
            theta = np.random.uniform(0, 2 * np.pi)
            human1_motion = self.random_rotate_around_y(human1_motion, theta)
            human2_motion = self.random_rotate_around_y(human2_motion, theta)
            human1_smpl_motion = self.random_rotate_around_y(human1_smpl_motion, theta)
            human2_smpl_motion = self.random_rotate_around_y(human2_smpl_motion, theta)
        
        dice = random.random()
        if not augment:
            dice = 1 # for test
        if dice < 0.5:
            subject_motion = human1_motion
            object_motion = human2_motion
            object_smpl_motion = human2_smpl_motion
        else:
            subject_motion = human2_motion
            object_motion = human1_motion
            object_smpl_motion = human1_smpl_motion
        
        subject_motion = subject_motion[start_frame:start_frame+self.max_frames]
        object_motion = object_motion[start_frame:start_frame+self.max_frames]
        object_smpl_motion = object_smpl_motion[start_frame:start_frame+self.max_frames]


        origin = subject_motion[0:1,0:1]
        subject_motion = subject_motion - origin
        object_motion = object_motion - origin
        object_smpl_motion = object_smpl_motion - origin

        object_voxel_grid = self.get_object_smpl_voxel_grid(object_smpl_motion)
        subject_voxel_index, valid_mask = self.get_subject_voxel_grid(subject_motion)

        if object_motion.shape[1] > self.max_object_points:
            idx = np.random.choice(object_motion.shape[1], self.max_object_points, replace=False)
            object_motion = object_motion[:,idx]
        else:
            object_motion = np.concatenate([object_motion,
                                     np.zeros((object_motion.shape[0], self.max_object_points-object_motion.shape[1], object_motion.shape[2]))
                                     ], axis=1)

        if object_smpl_motion.shape[1] > self.max_object_points:
            idx = np.random.choice(object_smpl_motion.shape[1], self.max_object_points, replace=False)
            object_smpl_motion = object_smpl_motion[:,idx]
        else:
            object_smpl_motion = np.concatenate([object_smpl_motion,
                                     np.zeros((object_smpl_motion.shape[0], self.max_object_points-object_smpl_motion.shape[1], object_smpl_motion.shape[2]))
                                     ], axis=1)


        return subject_voxel_index, object_voxel_grid, object_motion, valid_mask, text

    def get_object_smpl_voxel_grid(self, object_smpl_motion):

        half_grid_size = self.grid_size / 2

        voxel_grid = np.zeros((self.max_frames, *self.grid_size, 3))

        min_bound = -1 * half_grid_size * self.voxel_size

        voxel_indices = ((object_smpl_motion.reshape(-1,3) - min_bound) / self.voxel_size).round().astype(int).reshape(self.max_frames,-1,3)

        for t in range(self.max_frames):
            voxel_indices_tmp = voxel_indices[t]
            valid_mask = np.all((voxel_indices_tmp >= 0) & (voxel_indices_tmp < self.grid_size), axis=1)
            valid_voxels = voxel_indices_tmp[valid_mask]
            voxel_grid[t, valid_voxels[:,0], valid_voxels[:,1], valid_voxels[:,2]] = np.array([1,0,0])
        
        return voxel_grid

    
    def get_subject_voxel_grid(self, human_motion):

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
