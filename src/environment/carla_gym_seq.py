import copy
import os
from queue import Queue
import random
import docker
import time

import carla
import cv2
import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt

from src.utils.stats import sample_from_gmm_2d
from .cameras import build_cam
from .utils import loc_dist, pflat

# Hack-import objects for spawning
SpawnActor = carla.command.SpawnActor


def run_carla(off_screen=True, quality="Epic", gpu=["0"]):
    client = docker.from_env()

    env = {}  # Environment variables to set in the container
    volumes = {}

    if off_screen:
        command = "/bin/bash ./CarlaUE4.sh -RenderOffScreen -quality-level={}".format(quality)
        volumes = {"/tmp/.X11-unix": {"bind": "/tmp/.X11-unix", "mode": "rw"}}
    else:
        env["DISPLAY"] = os.environ['DISPLAY']
        command = "/bin/bash ./CarlaUE4.sh -quality-level={}".format(quality)

    container = None
    try:
        container = client.containers.run("carlasim/carla:0.9.14", 
                                          command=command, 
                                          detach=True, 
                                          privileged=True, 
                                          network_mode="host", 
                                          environment=env, 
                                          volumes=volumes, 
                                          device_requests=[docker.types.DeviceRequest(device_ids=gpu, capabilities=[['gpu']])])
         # wait for carla to start
        while container.status == "created":
            container.reload()
            time.sleep(2.0)
        print(container.logs().decode('utf-8'))
    except Exception as e:
        print(f"Error executing command: {e}")

    return container


def draw_bbox(obs, info):
    imgs = obs["images"]
    gts = info["pedestrian_gts"]
    fig, axs = plt.subplots(2, 2, figsize=(39, 22))

    for cam, img in imgs.items():
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        for anno in gts:
            anno = anno["views"][cam]
            bbox = tuple(
                [
                    int(anno["xmin"]),
                    int(anno["ymin"]),
                    int(anno["xmax"]),
                    int(anno["ymax"]),
                ]
            )
            if bbox[0] == -1 and bbox[1] == -1 and bbox[2] == -1 and bbox[3] == -1:
                continue
            img = cv2.rectangle(img, bbox[:2], bbox[2:], (0, 255, 0), 2)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axs[cam // 2, cam % 2].imshow(img)
        axs[cam // 2, cam % 2].set_title(f'Camera {cam + 1}')

    return fig, axs


class CarlaCameraSeqEnv(gym.Env):
    """
    The CARLA environment for single-camera multi-frame pedestrian detection
    """

    def __init__(self, opts, spawn_strategy, seed=None, host="127.0.0.1", port=2000, tm_port=8000):
        self.opts = opts
        self.spawn_strategy = spawn_strategy

        # if seed is provided, set seed to generators
        # otherwise randomly initialise generators
        if seed is not None:
            self.random_generator = random.Random(seed)
            self.np_random_generator = np.random.default_rng(seed)
        else:
            self.random_generator = random.Random()
            self.np_random_generator = np.random.default_rng()

        # Connect to the CARLA simulator
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world(self.opts["map"])

        # CarlaX is xy indexing; x,y (w,h) (n_col,n_row)
        # x_min, x_max, _, _, _, _ = opts["spawn_area"]
        # self.map_width = x_max - x_min
        # print(self.map_width)

        self.num_cam = self.opts["num_cam"]
        # step counter for deciding whether episode ends
        self.step_counter = 0

        # Define your environment's observation space
        self.observation_space = spaces.Dict(
            {
                "images": spaces.Box(
                    low=0,
                    high=255,
                    shape=(int(self.opts['cam_y']), int(self.opts['cam_x']), 3),
                    dtype=np.uint8,
                ),
                "camera_configs": spaces.Box(np.zeros([7]), np.ones([7]), dtype=np.float64)
            }
        )

        # Define your environment's action space
        self.action_space = spaces.Box(np.zeros([7]), np.ones([7]), dtype=np.float64)

        # Define any other attributes or variables needed for your environment
        # turn on sync mode
        traffic_manager = self.client.get_trafficmanager(tm_port)
        settings = self.world.get_settings()
        traffic_manager.set_synchronous_mode(True)
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)

        # world actors
        self.camera_configs = {}
        self.camera_intrinsics = {}
        self.camera_extrinsics = {}
        self.cameras = {}
        self.img_cam_buffer = {}
        self.pedestrians = []
        self.pedestrian_gts = []

    def format_action(self, action):
        # Encode an action with effective action space to a fixed actions space as in gym env
        rtn_action = np.zeros(self.action_space.shape)
        for action_name, value in zip(self.opts["env_action_space"].split("-"), action):
            if 'x' == action_name: rtn_action[0] = value
            if 'y' == action_name: rtn_action[1] = value
            if 'z' == action_name: rtn_action[2] = value
            if 'pitch' == action_name: rtn_action[3] = value
            if 'yaw' == action_name: rtn_action[4] = value
            if 'roll' == action_name: rtn_action[5] = value
            if 'fov' == action_name: rtn_action[6] = value
        # clip action to corresponding range
        rtn_action = np.clip(rtn_action, self.action_space.low, self.action_space.high)
        return rtn_action
    
    def encode_camera_cfg(self, cfg):
        # convert location/rotation/fov (different ranges) to [0, 1]
        # location
        x_min, x_max, y_min, y_max, z_min, z_max = self.opts["spawn_area"]
        # x, y, z = location[:, 0], location[:, 1], location[:, 2]
        x = (cfg["x"] - x_min) / (x_max - x_min)
        y = (cfg["y"] - y_min) / (y_max - y_min)
        z = (cfg["z"] - z_min) / (z_max - z_min)
        location = np.array([x, y, z])
        # rotation
        # pitch, yaw, roll = rotation[:, 0], rotation[:, 1], rotation[:, 2]
        pitch = (cfg["pitch"] + 90) / 180
        yaw = (cfg["yaw"] + 180) / 360
        roll = (cfg["roll"] + 90) / 180
        rotation = np.array([pitch, yaw, roll])

        # fov
        fov = np.array(cfg["fov"] / 180).reshape(-1)

        # put location, rotation, fov together
        act = np.concatenate([location, rotation, fov], axis=0)
        return act
    
    def decode_camera_cfg(self, action):
        # action is for a SINGLE camera
        location, rotation, fov = action[:3], action[3:6], action[6]
        x_min, x_max, y_min, y_max, z_min, z_max = self.opts["spawn_area"]
        # location based on spawn area (x_min,x_max,y_min,y_max)
        x, y, z = location[0], location[1], location[2]
        x = x * (x_max - x_min) + x_min
        y = y * (y_max - y_min) + y_min
        z = z * (z_max - z_min) + z_min
        location = np.array([x, y, z])
        # rotation
        pitch, yaw, roll = rotation[0], rotation[1], rotation[2]
        # pitch Y-axis rotation angle: -90~90
        pitch = pitch * 180 - 90
        # yaw Z-axis rotation angle: -180~180
        yaw = yaw * 360 - 180
        # roll X-axis rotation angle: -90~90
        roll = roll * 180 - 90
        rotation = np.array([pitch, yaw, roll])
        # fov: [0, 180)
        fov = 180 * fov
        return location, rotation, fov

    def encode_current_cam_action(self):
        action = np.zeros((self.opts["num_cam"], 7))
        # record the unnormalized actions
        for i, camera in self.cameras.items():
            locations = camera.get_transform().location
            rotations = camera.get_transform().rotation
            action[i, 0:3] = np.array([locations.x, locations.y, locations.z])
            action[i, 3:6] = np.array([rotations.pitch, rotations.yaw, rotations.roll])
            action[i, 6] = float(camera.attributes["fov"])

        # normalization (x-y-z)
        x_min, x_max, y_min, y_max, z_min, z_max = self.opts["spawn_area"]
        action[:, 0] = (action[:, 0] - x_min) / (x_max - x_min)
        action[:, 1] = (action[:, 1] - y_min) / (y_max - y_min)
        action[:, 2] = (action[:, 2] - z_min) / (z_max - z_min)

        # normalization (pitch-yaw-roll)
        action[:, 3] = (action[:, 3] + 90) / 180
        action[:, 4] = (action[:, 4] + 180) / 360
        action[:, 5] = (action[:, 5] + 90) / 180

        # normalization (fov)
        action[:, 6] = action[:, 6] / 180
        return action

    def action(self, act):
        # camera config for the next camera
        # allow more flexible choice of action space (x-y-z-pitch-yaw-roll-fov)
        # convert normalised action space to unnormalised ones
        _location, _rotation, _fov = self.decode_camera_cfg(act)
        action_space = self.opts["env_action_space"].split("-")
        # default settings for limited action space
        location, rotation, fov = np.array(self.opts["cam_pos_lst"])[self.step_counter], \
            np.array(self.opts["cam_dir_lst"])[self.step_counter], \
            np.array(self.opts["cam_fov"]).reshape([1])
        if 'x' in action_space: location[0] = _location[0]
        if 'y' in action_space: location[1] = _location[1]
        if 'z' in action_space: location[2] = _location[2]
        if 'pitch' in action_space: rotation[0] = _rotation[0]
        if 'yaw' in action_space: rotation[1] = _rotation[1]
        if 'roll' in action_space: rotation[2] = _rotation[2]
        if 'fov' in action_space: fov = _fov

        act = np.concatenate([location, rotation, fov], axis=0)
        return act

    def reset(self, seed=None, render_all_cams=True):
        # if a new seed is provided, set generator to used new seed
        # otherwise use old seed
        if seed is not None:
            self.random_generator = random.Random(seed)
            self.np_random_generator = np.random.default_rng(seed)

        # Reset the environment to its initial state and return the initial observation
        self.reset_cameras()
        self.respawn_pedestrians()
        self.step_counter = 0

        # NOTE: render all cameras by default
        cam_range = list(range(self.num_cam)) if render_all_cams else [0, ]
        images = self.render()
        observation = {
            "images": {cam: images[cam] for cam in cam_range},
            "camera_configs": {cam: self.encode_camera_cfg(self.camera_configs[cam]) for cam in cam_range}
        }
        info = {"pedestrian_gts": self.pedestrian_gts,
                "camera_intrinsics": self.camera_intrinsics,
                "camera_extrinsics": self.camera_extrinsics}  # Set any additional information

        # NOTE: Remember that Python only returns a reference to these objects
        # you may need to use copy.deepcopy() to avoid effects from further steps
        return observation, info

    def step(self, action):
        self.step_counter += 1
        # the input action would be an array of 7 numbers, as defined in action space
        # values are in the range of 0-1
        
        # get camera config encoding from the current camera positions
        # set the current action to the set of actions
        actions = self.encode_current_cam_action()
        actions[self.step_counter] = self.format_action(action)

        # directly use reset_cam to set camera positions
        self.reset_cameras(actions)

        # wait for two ticks to update the camera actors
        for i in range(2):
            self.world.tick()

        # update pedestrian bbox from each camera view
        for i, pedestrian in enumerate(self.pedestrians):
            actor = self.world.get_actor(pedestrian)
            self.pedestrian_gts[i]["views"][self.step_counter] = self.get_pedestrian_view(actor, self.step_counter)

        # Update the state, calculate the reward, and check for termination
        # Set the current observation
        observation = {
            "images": {self.step_counter: self.render()[self.step_counter]},
            "camera_configs": {self.step_counter: self.encode_camera_cfg(self.camera_configs[self.step_counter])},
        }
        # Set the reward for the current step
        reward = 0
        # Set condition for the end of episode: after a fixed number of step() call
        done = self.step_counter + 1 >= self.num_cam  # Set whether the episode has terminated or not
        # Set any additional information, the info can be used to calculate reward outside gym env
        info = {"pedestrian_gts": copy.deepcopy(self.pedestrian_gts),
                "camera_intrinsics": self.camera_intrinsics,
                "camera_extrinsics": self.camera_extrinsics, }

        # NOTE: Remember that Python only returns a reference to these objects
        # you may need to use copy.deepcopy() to avoid effects from further steps
        return observation, reward, done, info

    def render(self):
        # Render the environment
        images = {}
        # start listening the camera images
        for cam, camera in self.cameras.items():
            camera.listen(self.img_cam_buffer[cam].put)
        self.world.tick()
        # wait for sync until get all images
        for cam, queue_buffer in self.img_cam_buffer.items():
            image = queue_buffer.get()
            images[cam] = process_img(image)
        # end listening
        for camera in self.cameras.values():
            camera.stop()
        return images

    def close(self):
        # Clean up any resources or connections
        # after capturing all frames, destroy all actors
        self.client.apply_batch(
            [carla.command.DestroyActor(id) for id in self.pedestrians]
        )
        for camera in self.cameras.values():
            camera.destroy()

    def respawn_pedestrians(self):
        # Destroy existing actors, create new ones randomly
        self.client.apply_batch(
            [carla.command.DestroyActor(id) for id in self.pedestrians]
        )
        # spawn parameter, make the spawn area 0.5m smaller
        min_x, max_x = self.opts["spawn_area"][0:2]
        min_y, max_y = self.opts["spawn_area"][2:4]
        min_x, min_y = min_x + 0.5, min_y + 0.5
        max_x, max_y = max_x - 0.5, max_y - 0.5

        if self.spawn_strategy == 'gmm':
            # randomize the gmm parameters for the current scene
            num_peaks = self.random_generator.choice(list(range(1, 4)))
            weights = self.np_random_generator.random(num_peaks)
            weights /= weights.sum()
            means_x = self.np_random_generator.uniform(min_x, max_x, size=(num_peaks, 1))
            means_y = self.np_random_generator.uniform(min_y, max_y, size=(num_peaks, 1))
            means = np.hstack((means_x, means_y))
            variances = self.np_random_generator.uniform(4, 9, size=(num_peaks, 2))
            cov = []
            for variance in variances:
                cov.append(np.diag(variance))
            cov = np.array(cov)
        # Spawn pedestrians
        spawn_points = []
        for _ in range(self.opts["spawn_count"]):
            spawn_point = carla.Transform()
            loc = None
            while loc is None:
                # random initialise x, y, z
                loc = carla.Location()
                if self.spawn_strategy == 'uniform':
                    loc.x = self.random_generator.uniform(min_x, max_x)
                    loc.y = self.random_generator.uniform(min_y, max_y)
                elif self.spawn_strategy == 'gmm':
                    sample = sample_from_gmm_2d(self.np_random_generator, weights, means, cov, 1)[0]
                    loc.x = sample[0]
                    loc.y = sample[1]
                else:
                    raise NotImplementedError(f'{self.spawn_strategy} not implemented')
                # avoid collision with ground
                loc.z = 1.0
                if len(spawn_points):
                    distances = [
                        loc_dist(previous.location, loc) for previous in spawn_points
                    ]
                    if min(distances) < 1.0:
                        # To precent collisions with other road users
                        loc = None
            spawn_point.location = loc
            # randomise yaw -> [0, 359] -> [0, 360)
            spawn_point.rotation = carla.Rotation(0, self.random_generator.random() * 360, 0)
            spawn_points.append(spawn_point)
        batch = []
        bps_pedestrians = self.world.get_blueprint_library().filter(
            "walker.pedestrian.*"
        )
        for spawn_point in spawn_points:
            walker_bp = self.random_generator.choice(bps_pedestrians)
            # make sure all pedestrians are vincible
            if walker_bp.has_attribute("is_invincible"):
                walker_bp.set_attribute("is_invincible", "false")
            batch.append(SpawnActor(walker_bp, spawn_point))
        # apply spawn pedestrian
        results = self.client.apply_batch_sync(batch, True)
        self.pedestrians = []
        for res in results:
            if not res.error:
                # if error happens, very likely to be spawning failure caused by collision
                self.pedestrians.append(res.actor_id)
        # print(f"{len(self.pedestrians)} pedestrians spawned")

        # print("Stabilizing world...")
        # for _ in range(50):
        #     self.world.tick()
        #     time.sleep(0.01)
        self.world.tick()
        # print("World stabilized")

        self.update_pedestrian_gts()

    def update_pedestrian_gts(self):
        self.pedestrian_gts = []
        for pedestrian in self.pedestrians:
            # 1. new ground truth format
            actor = self.world.get_actor(pedestrian)
            loc = actor.get_location()

            bbox = actor.bounding_box
            l = bbox.extent.x * 2
            w = bbox.extent.y * 2
            h = bbox.extent.z * 2
            # Pedestrians' z value is at their middle height for some reason
            ped_z = loc.z - h / 2.0
            v = actor.get_velocity()
            rot = actor.get_transform().rotation
            forward = rot.get_forward_vector()

            # ped_worldcoord = [loc.x, loc.y]
            # ped_pos = int(get_pos_from_worldcoord(ped_worldcoord, *self.origin, self.map_width, self.opts["map_expand"]))

            self.pedestrian_gts.append(
                {
                    "id": pedestrian,
                    "x": loc.x,
                    "y": loc.y,
                    "z": ped_z,
                    "l": l,
                    "w": w,
                    "h": h,
                    "v_x": v.x,
                    "v_y": v.y,
                    "v_z": v.z,
                    "pitch": rot.pitch,
                    "roll": rot.roll,
                    "yaw": rot.yaw,
                    "forward_x": forward.x,
                    "forward_y": forward.y,
                    "forward_z": forward.z,
                    # "positionID": ped_pos,
                    "views": {cam: self.get_pedestrian_view(actor, cam) for cam in range(self.num_cam)},
                }
            )
        # print(f"Collected {len(self.pedestrian_gts)} pedestrian information")

    def get_pedestrian_view(self, actor, cam):
        bbox = actor.bounding_box
        verts = bbox.get_world_vertices(actor.get_transform())

        # prepare 2D bbox
        x_max = 0
        x_min = float(self.opts["cam_x"])
        y_max = 0
        y_min = float(self.opts["cam_y"])
        for vert in verts:
            # convert vert to homogeneous coordinate, vert is a carla.Location
            vert_homo = np.array([vert.x, vert.y, vert.z, 1])
            p_homo = self.camera_intrinsics[cam] @ self.camera_extrinsics[cam] @ vert_homo
            p = pflat(p_homo)
            if p[0] > x_max:
                x_max = p[0]
            if p[0] < x_min:
                x_min = p[0]
            if p[1] > y_max:
                y_max = p[1]
            if p[1] < y_min:
                y_min = p[1]

        # x_min, y_min, x_min, x_max = float(x_min), float(y_min), float(x_max), float(y_max)
        # Add the object to the frame (ensure it is inside the image)
        # if (
        #         x_min > 1
        #         and x_max < float(self.opts["cam_x"]) - 2
        #         and y_min > 1
        #         and y_max < float(self.opts["cam_y"]) - 2
        # ):
        if (
                x_max > 100
                and x_min < float(self.opts["cam_x"]) - 100
                and y_max > 100
                and y_min < float(self.opts["cam_y"]) - 100
        ):
            pedestrian_view = {"viewNum": cam, "xmin": x_min, "ymin": y_min, "xmax": x_max, "ymax": y_max, }
        else:
            # same as the convention (MultiviewX), feed invalid value (-1) if bbox outside picture
            pedestrian_view = {"viewNum": cam, "xmin": -1, "ymin": -1, "xmax": -1, "ymax": -1, }

        return pedestrian_view

    def reset_cameras(self, action=None):
        # destroy existing cameras
        for camera in self.cameras.values():
            camera.destroy()
        self.camera_configs = {}
        self.camera_intrinsics = {}
        self.camera_extrinsics = {}
        self.cameras = {}
        self.img_cam_buffer = {}

        camera_bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(self.opts["cam_x"]))
        camera_bp.set_attribute("image_size_y", str(self.opts["cam_y"]))

        if action is None:
            locations = np.array(self.opts["cam_pos_lst"])
            rotations = np.array(self.opts["cam_dir_lst"])
            fovs = np.array(self.opts["cam_fov"]).reshape(-1, 1).repeat(self.opts["num_cam"], 0)
        else:
            new_action = []
            for i, act in enumerate(action):
                if i == self.step_counter:
                    new_action.append(self.action(act))
                else:
                    location, rotation, fov = self.decode_camera_cfg(act)
                    new_action.append(np.concatenate([location, rotation, fov.reshape(1)], axis=0))
            action = np.array(new_action)
            locations, rotations, fovs = action[:, :3], action[:, 3:6], action[:, 6].reshape(-1, 1)

        for cam, (cam_pos, cam_dir, fov) in enumerate(zip(locations, rotations, fovs)):
            loc = carla.Location(*cam_pos)
            rot = carla.Rotation(*cam_dir)
            camera_bp.set_attribute("fov", str(fov.item()))
            camera_init_trans = carla.Transform(loc, rot)
            # spawn the camera
            camera = self.world.spawn_actor(camera_bp, camera_init_trans)
            # record camera related information
            self.cameras[cam] = camera

        # wait for one tick to update the camera actors
        self.world.tick()

        for cam, camera in self.cameras.items():
            # save camera configs, rather than projection matrices
            # projection/intrinsic/extrinsic matrices can be built from configs
            cam_config, intrinsic, extrinsic = get_camera_config(camera)
            self.camera_configs[cam] = cam_config
            self.camera_intrinsics[cam] = intrinsic
            self.camera_extrinsics[cam] = extrinsic
            self.img_cam_buffer[cam] = Queue(maxsize=0)


def process_img(img):
    # img is carla.libcarla.Image object;
    # returns a ndarray
    img_bgra = np.reshape(np.copy(img.raw_data), (img.height, img.width, 4))
    img_rgb = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2RGB)
    return img_rgb


def get_camera_config(camera):
    image_w = int(camera.attributes["image_size_x"])
    image_h = int(camera.attributes["image_size_y"])
    fov = float(camera.attributes["fov"])

    transform = camera.get_transform()
    loc = transform.location
    rot = transform.rotation

    f = image_w / (2.0 * np.tan(fov * np.pi / 360))
    Cx = image_w / 2.0
    Cy = image_h / 2.0

    # camera value is consist of 9 elements
    cam_value = {
        "x": loc.x,
        "y": loc.y,
        "z": loc.z,
        "pitch": rot.pitch,
        "roll": rot.roll,
        "yaw": rot.yaw,
        "f": f,
        "Cx": Cx,
        "Cy": Cy,
    }

    # config is consist of 7 elements
    cam_config = {
        "x": loc.x,
        "y": loc.y,
        "z": loc.z,
        "pitch": rot.pitch,
        "roll": rot.roll,
        "yaw": rot.yaw,
        "fov": fov,
    }

    _, intrinsic, extrinsic = build_cam(**cam_value)
    return cam_config, intrinsic, extrinsic


if __name__ == '__main__':
    import json

    with open('cfg/RL/2.cfg', "r") as fp:
        dataset_config = json.load(fp)

    env = CarlaCameraSeqEnv(dataset_config)
    observation, info = env.reset()
    done = False
    while not done:
        observation, reward, done, info = env.step(np.random.rand(dataset_config["num_cam"], 7))
