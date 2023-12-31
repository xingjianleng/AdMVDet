import os
import numpy as np
import cv2
import re
from torchvision.datasets import VisionDataset

# from src.environment.carla_gym import CarlaMultiCameraEnv
from src.environment.carla_gym_seq import CarlaCameraSeqEnv


class CarlaX(VisionDataset):
    def __init__(self, opts, host, port, tm_port, spawn_strategy, seed=None, root=os.path.expanduser("~/Data/CarlaX")):
        os.makedirs(root, exist_ok=True)
        super().__init__(root)
        # image of shape C,H,W (C,N_row,N_col); xy indexging; x,y (w,h) (n_col,n_row)
        # CarlaX has xy-indexing
        # CarlaX has consistent unit: meter (m) for calibration & pos annotation
        self.__name__ = 'CarlaX'
        self.env = CarlaCameraSeqEnv(opts, spawn_strategy, seed, host, port, tm_port)
        self.img_shape = [opts["cam_y"], opts["cam_x"]]  # H,W 
        x_min, x_max, y_min, y_max, _, _ = opts["spawn_area"]
        # annotation accuracy of 2.5 cm, opts["map_expand"] = 40
        self.worldgrid_shape = [int((y_max - y_min) * opts["map_expand"]),
                                int((x_max - x_min) * opts["map_expand"])]  # N_row,N_col
        self.num_cam, self.num_frame = opts["num_cam"], opts["num_frame"]
        # world x,y correspond to w,h
        self.indexing = 'xy'
        self.world_indexing_from_xy_mat = np.eye(3)
        self.world_indexing_from_ij_mat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        # image is in xy indexing by default
        self.img_xy_from_xy_mat = np.eye(3)
        self.img_xy_from_ij_mat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        # unit in meters
        self.worldcoord_unit = 1
        self.worldcoord_from_worldgrid_mat = np.array([[1 / self.env.opts["map_expand"], 0, x_min],
                                                       [0, 1 / self.env.opts["map_expand"], y_min],
                                                       [0, 0, 1]])
        self.intrinsic_matrices, self.extrinsic_matrices = self.env.camera_intrinsics, self.env.camera_extrinsics

    def get_worldgrid_from_worldcoord(self, world_coord):
        coord_x, coord_y = world_coord[0, :], world_coord[1, :]
        x_min, x_max, y_min, y_max, _, _ = self.env.opts["spawn_area"]
        grid_x = (coord_x - x_min) * self.env.opts["map_expand"]
        grid_y = (coord_y - y_min) * self.env.opts["map_expand"]
        return np.array([grid_x, grid_y], dtype=int)

    def get_worldcoord_from_worldgrid(self, worldgrid):
        grid_x, grid_y = worldgrid[0, :], worldgrid[1, :]
        x_min, x_max, y_min, y_max, _, _ = self.env.opts["spawn_area"]
        coord_x = x_min + grid_x / self.env.opts["map_expand"]
        coord_y = y_min + grid_y / self.env.opts["map_expand"]
        return np.array([coord_x, coord_y])
