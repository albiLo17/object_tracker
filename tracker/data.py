
"""Open the fold recording and dump shapes + a sample frame from one camera.

Usage:
    python read_example.py [path/to/file.h5]
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from dataclasses import dataclass
import pickle

import cv2
import h5py
import imageio
import numpy as np
import torchvision
from PIL import Image
from torchvision.transforms import functional as F
import torch




@dataclass
class CameraIntrinsics:
    width: int
    height: int
    fov_x: float
    fov_y: float
    fx: float
    fy: float
    cx: float
    cy: float


def camera_intrinsics(rendering_attrs: dict) -> CameraIntrinsics:
    width = int(rendering_attrs["width"])
    height = int(rendering_attrs["height"])
    fov_x = fov_y = float(rendering_attrs["camera_angle_x"])
    fx = fy = (width / 2) / math.tan(fov_x / 2)
    return CameraIntrinsics(
        width=width,
        height=height,
        fov_x=fov_x,
        fov_y=fov_y,
        fx=fx,
        fy=fy,
        cx=width / 2,
        cy=height / 2
    )


def load_frames_from_directory(directory):
    # 1. Gather and sort filenames
    # The 4-digit padding ensures '0010.png' comes after '0009.png'
    files = sorted([f for f in os.listdir(directory) if f.endswith('.png')])

    image_list = []

    print(f"Loading {len(files)} frames from {directory}...")

    for filename in files:
        img_path = os.path.join(directory, filename)
        img_bgr =  cv2.imread(img_path)            # 3. Convert to Torch Tensor
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # This automatically scales pixel values to [0.0, 1.0]
        # and reshapes to (Channels, Height, Width)
        image_list.append(img_rgb)

    return image_list

def sim_data_loader(h5_path: Path, cam_id=0) -> (np.ndarray, np.ndarray, np.ndarray, CameraIntrinsics):
    with h5py.File(h5_path, "r") as f:
        sim = dict(f["metadata/simulation_parameters"].attrs)
        rendering = dict(f["metadata/rendering_parameters"].attrs)
        print(f"task={sim['task']}, dt={sim['dt']}, seed={sim['random_seed']}")
        print(f"image_save_freq={sim['image_save_freq']}, save_freq={sim['save_freq']}")

        intrinsics = camera_intrinsics(rendering)
        extrinsics = f[f"metadata/rendering_parameters/frames/frame_{cam_id}/transform_matrix"][:]

        cam_frames = sorted(
            f["metadata/rendering_parameters/frames"].keys(),
            key=lambda k: int(k.split("_")[1]),
        )

        print(f"available camera frames: {len(cam_frames)} (frame_0 ... frame_39)")

        meshes = sorted(f["training"].keys())
        print(f"\nmeshes ({len(meshes)}): {meshes}")

        mesh_name = meshes[0]
        mesh = f["training"][mesh_name]
        print(f"\n--- {mesh_name} ---")
        print(f"  rest_positions: {mesh['rest_positions'].shape}")
        print(f"  edges: {mesh['edges'].shape}, faces: {mesh['faces'].shape}")

        traj_name = sorted(k for k in mesh if k.startswith("trajectory_"))[0]
        traj = mesh[traj_name]
        steps = sorted(k for k in traj if k.startswith("step_"))
        rendered = [s for s in steps if "images" in traj[s]]

        images = []
        depth = []
        pcd = []
        for rendered_step_key in rendered:

            # print(f"  {traj_name}: {len(steps)} steps, {len(rendered)} rendered")
            #
            # step = traj[rendered[0]]
            # print(f"  step={rendered[0]} cameras={cams}")

            step = traj[rendered_step_key]
            cam_keys = sorted(step['images'].keys())
            cam_key = f'cam_{cam_id}'
            assert cam_key in cam_keys
            #
            # positions = step["positions"][:]
            # gripper = step["gripper_pos"][:]
            # # print(f"    rgb {rgb.shape} {rgb.dtype}, "
            # #       f"depth {depth.shape} {depth.dtype} "
            # #       f"[{depth.min():.3f}, {depth.max():.3f}] m")
            # # print(f"    pointcloud {pcd.shape} {pcd.dtype}")
            # # print(f"    positions {positions.shape}, gripper {gripper.shape}")
            images.append(step[f"images/{cam_key}"][:])
            depth.append(step[f"depth/{cam_key}"][:])
            pcd.append(step[f"pointclouds/{cam_key}"][:])

            # frame_idx = cam.split("_")[1]

        return images, depth, pcd, intrinsics, extrinsics

def robot_data_loader(pkl_path, cam='left'):

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    if cam == 'left':
        intrinsics_rgb = CameraIntrinsics(
            width=640,
            height=480,
            fov_x=2 * math.atan(640 / (2 * 614.91)) * (180 / math.pi),
            fov_y=2 * math.atan(480 / (2 * 615.047)) * (180 / math.pi),
            fx=614.91,
            fy=615.047,
            cx=314.035,
            cy=239.564
        )

        intrinsics_depth = CameraIntrinsics(
            width=640,
            height=480,
            fov_x=2 * math.atan(640 / (2 * 384.943)) * (180 / math.pi),
            fov_y=2 * math.atan(480 / (2 * 384.943)) * (180 / math.pi),
            fx=384.943,
            fy=384.943,
            cx=319.227,
            cy=238.486
        )
    elif cam == 'right':
        intrinsics_rgb = CameraIntrinsics(
            width=640,
            height=480,
            fov_x=2 * math.atan(640 / (2 * 605.699)) * (180 / math.pi),
            fov_y=2 * math.atan(480 / (2 * 605.53)) * (180 / math.pi),
            fx=605.699,
            fy=605.53,
            cx=323.649,
            cy=246.309
        )

        intrinsics_depth = CameraIntrinsics(
            width=640,
            height=480,
            fov_x=2 * math.atan(640 / (2 * 385.7)) * (180 / math.pi),
            fov_y=2 * math.atan(480 / (2 * 385.7)) * (180 / math.pi),
            fx=385.7,
            fy=385.7,
            cx=325.66,
            cy=236.769
        )
    else:
        raise Exception(f"Unknown camera {cam}")

    images = []
    depths = []
    for step in data['steps']:
        observation_data = step['observation']
        if cam == 'left':
            image = observation_data['image/external_left']
            depth = observation_data['depth/external_left']
        elif cam == 'right':
            image = observation_data['image/external_right']
            depth = observation_data['depth/external_right']
        else:
            raise Exception(f"Unknown camera {cam}")

        images.append(image)
        depths.append(depth)


    # TODO Crop image to be more centered and have smaller resolution

    return images, depths, intrinsics_rgb, intrinsics_depth

def zed_camera_dataloader(pkl_path):

    intrinsics_rgb = CameraIntrinsics(
        width=640,
        height=480,
        fov_x=2 * math.atan(640 / (2 * 605.699)) * (180 / math.pi),
        fov_y=2 * math.atan(480 / (2 * 605.53)) * (180 / math.pi),
        fx=605.699,
        fy=605.53,
        cx=323.649,
        cy=246.309
    )

    intrinsics_depth = CameraIntrinsics(
        width=640,
        height=480,
        fov_x=2 * math.atan(640 / (2 * 385.7)) * (180 / math.pi),
        fov_y=2 * math.atan(480 / (2 * 385.7)) * (180 / math.pi),
        fx=385.7,
        fy=385.7,
        cx=325.66,
        cy=236.769
    )

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    images = []
    depths = []
    for step in data:
        images.append(step[f'external_back'].squeeze())
        depths.append(step[f'external_back_depth'].squeeze())

    return images, depths, intrinsics_rgb, intrinsics_depth


if __name__ == "__main__":
    # sim_data_path = Path(
    #     "../data/fold/fold_meshes_with_hole_3meshes_3cams_seed_2026.h5"
    # )


    robot_data_path = Path(
        "../data/robot/oculus_teleop.pkl"
    )

    rgbs, depths, intrinsics_rgb, extrinsics_rgb = zed_camera_dataloader(robot_data_path)

    out_dir = robot_data_path.parent / "sample_frame"
    out_dir.mkdir(exist_ok=True)


    rgb = rgbs[0]
    depth = depths[0]
    imageio.imwrite(out_dir / f"_right.png", rgb)
    depth_vis = np.zeros_like(depth, dtype=np.uint8)
    valid = depth > 0
    if valid.any():
        d = depth[valid]
        depth_vis[valid] = (255 * (1 - (d - d.min()) / (d.max() - d.min() + 1e-8))).astype(np.uint8)
    imageio.imwrite(out_dir / f"depth_right.png", depth_vis)
    # np.save(out_dir / f"pcd.npy", pcd)
    print(f"\nwrote sample to {out_dir}")
