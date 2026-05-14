import time
from pathlib import Path
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor, AutoImageProcessor, AutoModelForDepthEstimation
from transformers import Sam2VideoModel, Sam2VideoProcessor
from accelerate import Accelerator

from data import sim_data_loader, robot_data_loader, load_frames_from_directory, zed_camera_dataloader

import open3d as o3d



# from fusion_class import Fusion


def save_pc(points, filename):
    # Convert torch to numpy
    # Create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Save (Open3D infers format from extension like .ply or .pcd)
    o3d.io.write_point_cloud(filename, pcd)


def save_pc_list(pc_list, folder_name="point_clouds"):
    os.makedirs(folder_name, exist_ok=True)

    print(f"Saving {len(pc_list)} point clouds to {folder_name}...")

    for i, pc in enumerate(pc_list):
        # Using the custom PLY function from the previous step
        filename = os.path.join(folder_name, f"pc_{i:04d}.ply")
        save_pc(pc, filename)

    print("Done.")


def align_depth_torch(noisy_metric, mono_depth):
    """
    Fits mono_depth to noisy_metric using Least Squares:
    noisy_metric ≈ s * mono_depth + t
    """
    # 1. Create mask for valid pixels (ignore sensor holes/zeros)
    mask = (noisy_metric > 0.1) & (mono_depth > 0)

    # Flatten to 1D for the linear system
    target = noisy_metric[mask].view(-1, 1)  # Metric (Y)
    source = mono_depth[mask].view(-1, 1)  # Mono (X)

    # 2. Build design matrix A = [source, 1]
    ones = torch.ones_like(source)
    A = torch.cat([source, ones], dim=1)

    # 3. Solve Ax = B (Returns [scale, shift])
    # rcond=None handles potential rank deficiencies
    solution = torch.linalg.lstsq(A, target).solution
    s, t = solution[0][0], solution[1][0]

    # 4. Transform the full mono depth map
    aligned_depth = s * mono_depth + t
    return torch.clamp(aligned_depth, min=0.0)


def project_depth(depth, mask, intrinsics, device="cuda"):
    """
    Projects a single depth and mask pair into a 3D point cloud.
    OpenCV Convention: x-right, y-down, z-forward.
    """
    # 1. Prepare tensors (ensure 2D and correct device)
    d = depth.detach().to(device).squeeze()
    m = mask.detach().to(device).squeeze().bool()

    h, w = d.shape

    # 2. Generate coordinate grids for this specific frame size
    v, u = torch.arange(h, device=device), torch.arange(w, device=device)
    uu, vv = torch.meshgrid(u, v, indexing='xy')

    # 3. Sparse Extraction
    # We only pull values where the mask is True
    z_sparse = d[m]
    u_sparse = uu[m]
    v_sparse = vv[m]

    # 4. 3D Back-projection Math
    # $x = (u - c_x) * z / f_x$
    # $y = (v - c_y) * z / f_y$
    x_sparse = (u_sparse - intrinsics.cx) * z_sparse / intrinsics.fx
    y_sparse = (v_sparse - intrinsics.cy) * z_sparse / intrinsics.fy

    # 5. Stack into [N, 3]
    return torch.stack([x_sparse, y_sparse, z_sparse], dim=-1)


def save_depth_as_heatmaps(depth_list, mask_list, image_list, output_dir, cmap_name='magma', save_original=True):
    """
    Converts a list of depth tensors to colorized heatmaps using global normalization
    calculated from the entire depth map (ignoring the mask for bounds).
    """
    os.makedirs(output_dir, exist_ok=True)
    colormap = plt.get_cmap(cmap_name)

    depth_dir = os.path.join(output_dir, 'depth')
    os.makedirs(depth_dir, exist_ok=True)

    if save_original:
        rgb_dir = os.path.join(output_dir, 'rgb')
        os.makedirs(rgb_dir, exist_ok=True)

    # --- STEP 1: CALCULATE GLOBAL BOUNDS (ENTIRE SCENE) ---
    print("Calculating global depth range from full frames...")

    global_min = float('inf')
    global_max = float('-inf')

    for depth_tensor in depth_list:
        d_val = depth_tensor.detach().cpu().numpy() if torch.is_tensor(depth_tensor) else np.array(depth_tensor)

        # We take the min and max of the entire frame
        global_min = min(global_min, d_val.min())
        global_max = max(global_max, d_val.max())

    print(f"Global Scene Range: Min={global_min:.4f}, Max={global_max:.4f}")

    # --- STEP 2: PROCESS FRAMES ---
    for i, (depth_tensor, mask, image) in enumerate(zip(depth_list, mask_list, image_list)):
        # Convert depth to numpy (H, W)
        depth_np = depth_tensor.detach().cpu().numpy().squeeze() if torch.is_tensor(
            depth_tensor) else depth_tensor.squeeze()
        if depth_np.ndim > 2:
            depth_np = depth_np[..., 0]

        # Normalize using the GLOBAL scene values
        if global_max > global_min:
            depth_norm = np.clip((depth_np - global_min) / (global_max - global_min), 0, 1)
        else:
            depth_norm = np.zeros_like(depth_np)

        # Apply Colormap
        heatmap_rgba = colormap(depth_norm)
        heatmap_rgb = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)

        # Ensure mask and image are ready for compositing
        m_np = mask.detach().cpu().numpy().squeeze() if torch.is_tensor(mask) else np.array(mask).squeeze()
        img_np = image.detach().cpu().numpy() if torch.is_tensor(image) else np.array(image)

        # Scale image if it's float [0, 1]
        if img_np.dtype != np.uint8:
            img_np = (img_np * 255).astype(np.uint8) if img_np.max() <= 1.0 else img_np.astype(np.uint8)

        # Composite: Use the heatmap only where the mask is True, else the original image
        final_image = np.where(m_np[..., None].astype(bool), heatmap_rgb, img_np)

        # Save

        Image.fromarray(final_image).save(os.path.join(depth_dir, f"depth_{i:04d}.png"))
        if save_original:
            Image.fromarray(img_np).save(os.path.join(rgb_dir, f"rgb_{i:04d}.png"))

    print(f"Success! {len(depth_list)} frames saved to {output_dir}")


if __name__ == "__main__":

    # Model preparation
    device = Accelerator().device
    # sam_model = Sam3TrackerVideoModel.from_pretrained("facebook/sam3").to(device, dtype=torch.bfloat16)
    # sam_processor = Sam3TrackerVideoProcessor.from_pretrained("facebook/sam3")
    sam_model = Sam2VideoModel.from_pretrained("facebook/sam2.1-hiera-tiny").to(device, dtype=torch.bfloat16)
    sam_processor = Sam2VideoProcessor.from_pretrained("facebook/sam2.1-hiera-tiny")
    # fusion_model = Fusion(1)

    depth_model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf", device_map="auto")
    depth_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")

    # Data loading
    # h5_path = Path("../data/fold/fold_meshes_with_hole_3meshes_3cams_seed_2026.h5")
    # images, depth, pcd, intrinsics, extrinsics = data_loader(h5_path)
    # images = images * 25
    # input_points=[[[[220, 220], [260, 260], [10, 10]]]]
    # input_labels=[[[1, 1, 0]]]
    # images = load_frames_from_directory('../data/sweater/frames_300')

    robot_data_path = Path("../data/robot/oculus_teleop.pkl")
    # images, metric_depths, intrinsics_rgb, intrinsics_depth = robot_data_loader(robot_data_path, 'right') FOR rollout_000 and rollout_001
    images, metric_depths, intrinsics_rgb, intrinsics_depth = zed_camera_dataloader(robot_data_path)
    # input_points = [[[[450, 210]]]]   # Rollout_000
    # input_points = [[[[400, 190]]]]     # Rollout_001]
    # input_labels=[[[1]]]
    input_points = [[[[300, 220], [575, 250], [800, 180], [800, 460], [770, 360]]]]     # ocolus_teleop
    input_labels=[[[1, 1, 1, 1, 0]]]                         # ocolus_teleop
    # Initialize session for streaming
    inference_session = sam_processor.init_video_session(
        inference_device=device,
        dtype=torch.bfloat16,
    )

    # Add point input on first frame
    sam_inputs = sam_processor(images=images[0], device=device, return_tensors="pt")
    sam_processor.add_inputs_to_inference_session(
        inference_session=inference_session,
        frame_idx=0,
        obj_ids=1,
        input_points=input_points,
        input_labels=input_labels,
        original_size=sam_inputs.original_sizes[0],  # need to be provided when using streaming video inference
    )

    depths = []
    masks = []
    point_clouds = []
    original_point_clouds = []

    # Process frames one by one
    t1 = time.time()
    for frame_idx, frame in enumerate(images):
        # if frame_idx == 0:
        sam_inputs = sam_processor(images=frame, device=device, return_tensors="pt")
        # Process current frame
        sam_tracker_video_output = sam_model(inference_session=inference_session, frame=sam_inputs.pixel_values[0])
        video_res_masks = sam_processor.post_process_masks(
            [sam_tracker_video_output.pred_masks], original_sizes=sam_inputs.original_sizes, binarize=False
        )[0]

        mask = video_res_masks[0][0] > 0
            # fusion_model.xmem_process([frame], mask[None, :, :])

        depth_inputs = depth_processor(images=frame, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = depth_model(**depth_inputs)
        post_processed_output = depth_processor.post_process_depth_estimation(
            outputs,
            target_sizes=[(frame.shape[0], frame.shape[1])],
        )
        depth_prediction = post_processed_output[0]['predicted_depth']

        # mask = fusion_model.xmem_process([frame], None)[0, :, :, 1]

        metric_depth = torch.tensor(metric_depths[frame_idx], dtype=torch.float32, device=device)
        aligned_depth = align_depth_torch(metric_depth, depth_prediction)
        point_cloud = project_depth(aligned_depth, mask, intrinsics_depth, device=device)
        original_point_cloud = project_depth(metric_depth, mask, intrinsics_depth, device=device)

        masks.append(mask.detach().cpu().numpy())
        depths.append(aligned_depth.detach().cpu().numpy())
        point_clouds.append(point_cloud.detach().cpu().numpy())
        original_point_clouds.append(original_point_cloud.detach().cpu().numpy())

        #
        # mask_np = video_res_masks[0][0].to(torch.float32).cpu().numpy() > 0
        # frame_np = np.array(frame)
        #
        #
        # # 3. Create a solid grey background of the same shape
        # background = np.full(frame_np.shape, 128, dtype=np.uint8)
        # mask_3d = np.expand_dims(mask_np, axis=-1)
        # mask_3d = mask_3d * 0.9 + 0.1
        # processed_frame = mask_3d * frame_np
        # # processed_frame = np.where(mask_3d, frame_np, background)
        #
        # result_img = Image.fromarray(processed_frame.astype(np.uint8))
        # result_img.save(os.path.join('../output/', f"frame_{frame_idx:04d}.png"))
        #
        # print(f"Frame {frame_idx}: mask shape {video_res_masks.shape}")
    delta_time = time.time() - t1
    print("N_images: ", len(images))
    print("Total: {:.2f} seconds".format(delta_time))
    print("Frequency: {:.2f} Hz".format(len(images) / delta_time))

    output_dir = Path("../output/robot_oculus_teleop_sam")
    if not output_dir.exists():
        output_dir.mkdir()
    save_pc_list(point_clouds, output_dir / 'point_clouds')
    save_pc_list(original_point_clouds, output_dir / 'original_point_clouds')
    save_depth_as_heatmaps(depths, masks, images, output_dir=output_dir, save_original=False)

