import numpy as np
import matplotlib.pyplot as plt
import cv2
from tracker.object_tracker import TrackerMultiView, MaskSelectionInterface
import os
from utils import load_demo_dataset_piper, process_piper_observation
import argparse
import imageio
from pathlib import Path
import tqdm

def overlay_mask_on_rgb(rgb, mask, alpha=0.4, color=(255, 0, 0)):
    """
    Overlay a mask onto an RGB image.
    :param rgb: (H, W, 3) RGB image (uint8)
    :param mask: (H, W) boolean or uint8 mask
    :param alpha: transparency of overlay
    :param color: BGR color for overlay
    :return: (H, W, 3) RGB image
    """
    overlay = rgb.copy()
    overlay[mask > 0] = (
        (1 - alpha) * overlay[mask > 0] + alpha * np.array(color[::-1])
    )
    return overlay.astype(np.uint8)

def load_rgbs(data_path, num_traj,control_mode,camera_names=["base_camera", "wrist_camera"]):
    # load piper dataset
    trajectories = load_demo_dataset_piper(data_path, num_traj=num_traj, control_mode=control_mode)
    # Pre-process the observations
    rgbs_traj_dict_list = []
    for obs_traj_dict in trajectories["observations"]:
        _obs_traj_dict = process_piper_observation(obs_traj_dict)
        rgbs_traj_dict_list.append(_obs_traj_dict["rgb"])

    # start processing only the first trajectory
    traj_id = 0
    # get rgb files
    rgbs = {camera_names[0]: [], camera_names[1]: [], }
    for rgb_images in rgbs_traj_dict_list[traj_id]:
        rgb_images_base = rgb_images[3:]
        rgb_images_wrist = rgb_images[:3]
        # Convert from (C, H, W) to (H, W, C)
        img_base = cv2.cvtColor(np.transpose(rgb_images_base, (1, 2, 0)), cv2.COLOR_BGR2RGB)
        img_wrist = cv2.cvtColor(np.transpose(rgb_images_wrist, (1, 2, 0)), cv2.COLOR_BGR2RGB)
        rgbs[camera_names[0]].append(img_base)
        rgbs[camera_names[1]].append(img_wrist)
    print("loaded")

    # # Debug: save images to disk
    # cv2.imwrite(f'debug_base.png', cv2.cvtColor(img_base, cv2.COLOR_RGB2BGR))
    # cv2.imwrite(f'debug_wrist.png', cv2.cvtColor(img_wrist, cv2.COLOR_RGB2BGR))

    return rgbs

def get_masks(rgbs, camera_names, tracker):
    all_masks = []
    # create tqdm bar
    pbar = tqdm.tqdm(total=len(rgbs[camera_names[0]]), desc="Processing frames")
    for t in range(pbar.total):
    # for t in range(len(rgbs[camera_names[0]])):
        rgbs_t = {camera: rgbs[camera][t] for camera in camera_names}
        
        masks_cameras = tracker.get_masks(rgbs_t)
        # update pbar description with the ratio of frames
        pbar.set_description(f"Processing frames {t+1}/{pbar.total}")
        pbar.update(1)
        # print(f'Generated masks for time {t}')
        all_masks.append(masks_cameras)

    return all_masks

def make_mask_gif(rgbs, masks, output_dir, gif_name="mask_debug"):
    for camera in rgbs:
        frames = []
        rgb_sequence = rgbs[camera]
        mask_sequence = masks[camera]

        for idx, (rgb, mask) in enumerate(zip(rgb_sequence, mask_sequence)):
            # Ensure mask is boolean
            if mask.dtype != bool:
                mask = mask > 0

            # Overlay mask as blue
            overlayed = overlay_mask_on_rgb(rgb, mask, alpha=0.4, color=(0, 0, 255))
            
            # Convert RGB to BGR for OpenCV text drawing
            overlayed_bgr = cv2.cvtColor(overlayed, cv2.COLOR_RGB2BGR)
            cv2.putText(
                overlayed_bgr,
                f"Frame {idx}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            # Convert back to RGB for GIF
            overlayed_rgb = cv2.cvtColor(overlayed_bgr, cv2.COLOR_BGR2RGB)
            frames.append(overlayed_rgb)

        # Save GIF
        gif_path = output_dir / f"{camera}_{gif_name}.gif"
        imageio.mimsave(gif_path, frames, duration=0.2)
        print(f"Saved GIF for {camera}: {gif_path}")

def compute_iou(mask1, mask2):
    """
    Compute IoU between two binary masks.
    """
    mask1 = mask1 > 0
    mask2 = mask2 > 0
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Track cube script")
    parser.add_argument('--data_path', type=str, default="/beegfs/scratch/user/alonghin/object_tracker/piper_datasets/piper_albi/debug_sync", help='Path to the dataset')
    parser.add_argument('--data_replay_path', type=str, default="/beegfs/scratch/user/alonghin/object_tracker/piper_datasets/piper_albi/debug_sync", help='Path to the dataset')
    parser.add_argument('--num_traj', type=int, default=1, help='Number of trajectories to load')
    parser.add_argument('--control_mode', type=str, default='pd_joint_pos', help='Control mode to use')
    args = parser.parse_args()


    camera_names = ["base_camera", "wrist_camera"]
    rgbs = load_rgbs(data_path=args.data_path, 
                     num_traj=args.num_traj, 
                     control_mode=args.control_mode, 
                     camera_names=camera_names)
    
    rgbs_replay = load_rgbs(data_path=args.data_replay_path,
                            num_traj=args.num_traj, 
                            control_mode=args.control_mode, 
                            camera_names=camera_names)
        
    
    # set trakcer and mask selection interface
    labels = ["cube"]
    tracker = TrackerMultiView(camera_names=camera_names, labels=labels) 
    
    ################## Track with XMem ##################

    all_masks = get_masks(rgbs, camera_names, tracker)
    # reset the mask or feed all of them together at the same time, but it increases memory
    tracker.reset()  # Reset tracker to initlaize the mask of the first image sequence
    all_masks_replay = get_masks(rgbs_replay, camera_names, tracker)
        
    # get rope masks for camera 0

    masks = {camera_names[0]: [], camera_names[1]: [], }
    for cam in camera_names:
        masks[cam] = [all_masks[t][cam]['cube'] for t in range(len(all_masks))]

    masks_replay = {camera_names[0]: [], camera_names[1]: [], }
    for cam in camera_names:
        masks_replay[cam] = [all_masks_replay[t][cam]['cube'] for t in range(len(all_masks_replay))]

    # Output folder
    output_dir = Path("debug_mask_gifs")
    output_dir.mkdir(exist_ok=True)

    make_mask_gif(rgbs, masks, output_dir, gif_name="mask_debug")
    make_mask_gif(rgbs_replay, masks_replay, output_dir, gif_name="mask_replay_debug")

    # Compute IoU between masks of the two sequences
    camera = camera_names[1]  # Choose one camera to compute IoU


    print(f"\nProcessing camera: {camera}")

    skip_initial = 10

    num_timesteps = min(len(masks[camera][skip_initial:]), len(masks_replay[camera][skip_initial:]))
    ious = []

    # Compute IoU per timestep
    for t in range(num_timesteps):
        mask1 = masks[camera][t]
        mask2 = masks_replay[camera][t]
        iou = compute_iou(mask1, mask2)
        ious.append(iou)


    ious = np.array(ious)
    print("IoUs per timestep:", ious)

    # Find timestep with lowest IoU
    min_idx = np.argmin(ious)
    min_iou = ious[min_idx]
    print(f"Lowest IoU: {min_iou:.3f} at timestep {min_idx}")


    # Retrieve RGBs and masks
    rgb1 = rgbs[camera][min_idx]
    rgb2 = rgbs_replay[camera][min_idx]
    mask1 = masks[camera][min_idx]
    mask2 = masks_replay[camera][min_idx]

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # RGB1 with overlay
    overlay1 = overlay_mask_on_rgb(rgb1, mask2)
    axes[0,0].imshow(overlay1)
    axes[0,0].set_title(f"RGB 1 + Mask 2 (Frame {min_idx})")
    axes[0,0].axis('off')

    # Mask 1 only
    axes[0,1].imshow(mask1, cmap='gray')
    axes[0,1].set_title("Mask 1")
    axes[0,1].axis('off')

    # RGB2 with overlay
    overlay2 = overlay_mask_on_rgb(rgb2, mask1)
    axes[1,0].imshow(overlay2)
    axes[1,0].set_title(f"RGB 2 + Mask 1 (Frame {min_idx})")
    axes[1,0].axis('off')

    # Mask 2 only
    axes[1,1].imshow(mask2, cmap='gray')
    axes[1,1].set_title("Mask 2")
    axes[1,1].axis('off')

    fig.suptitle(f"Camera: {camera} - Lowest IoU: {min_iou:.3f}", fontsize=16)

    # Save figure
    save_path = output_dir / f"{camera}_lowest_iou_pair.png"
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

    print(f"Saved debug image to: {save_path}")



    