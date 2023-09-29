import os
import json

from src.datasets.frameDataset import frameDataset
from src.datasets.multiviewx import MultiviewX
from src.datasets.wildtrack import Wildtrack
from src.datasets.carlax import CarlaX


def dataset_stat(dataset):
    num_cam = dataset.base.num_cam
    img_res = (dataset.base.img_shape[1], dataset.base.img_shape[0])
    num_frame = dataset.base.num_frame
    world_region = (dataset.base.worldgrid_shape[0] * 2.5 / 100,
                    dataset.base.worldgrid_shape[1] * 2.5 / 100)
    
    if dataset.base.__name__ == 'CarlaX':
        crowd = 0

        frame_range = dataset.frames[-1] - dataset.frames[0] + 1
        for i in range(frame_range):
            curr_gt = dataset[i]

        crowd = 0
        for gt in dataset.world_gt.values():
            crowd += len(gt[0]) / len(dataset.world_gt)
        avg_coverage = (num_cam * dataset.Rworld_coverage.squeeze().mean(0)).mean()
    else:
        crowd = 0
        for gt in dataset.world_gt.values():
            crowd += len(gt[0]) / len(dataset.world_gt)

        avg_coverage = (num_cam * dataset.Rworld_coverage.squeeze().mean(0)).mean()

    output = f'{dataset.base.__name__}, num_cam: {num_cam}, img_res: {img_res}, num_frame: {num_frame}, world_region: {world_region}, crowd: {crowd}, avg_coverage: {avg_coverage}'
    print(output)


if __name__ == '__main__':
    dataset_stat(frameDataset(Wildtrack(os.path.expanduser('~/Data/Wildtrack')), 'test'))
    dataset_stat(frameDataset(MultiviewX(os.path.expanduser('~/Data/MultiviewX')), 'test'))
    with open('./cfg/RL/1.cfg', 'r') as fp:
        opts = json.load(fp)
    dataset_stat(frameDataset(CarlaX(opts, '127.0.0.1', 3000, 5000, "uniform", 2023), 'test'))
