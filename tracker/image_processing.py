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

# from fusion_class import Fusion


def align_depth(noisy_metric, mono_depth):
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


class DepthEstimator:

    def __init__(self, depth_model='depth-anything', device="cuda"):

        if depth_model == 'depth-anything':
            self.depth_model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf", device_map="auto")
            self.depth_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
        else:
            raise Exception(f"Unknown depth model: {depth_model}")

        self.device = device

    @torch.no_grad()
    def estimate_depth(self, image):

        depth_inputs = self.depth_processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.depth_model(**depth_inputs)
        post_processed_output = self.depth_processor.post_process_depth_estimation(
            outputs,
            target_sizes=[(image.shape[0], image.shape[1])],
        )
        depth_prediction = post_processed_output[0]['predicted_depth']

        return depth_prediction


class Segmenter:

    def __init__(self, model='sam2-tiny', device="cuda"):

        if model == 'sam2-tiny':
            self.sam_model = Sam2VideoModel.from_pretrained("facebook/sam2.1-hiera-tiny").to(device, dtype=torch.bfloat16)
            self.sam_processor = Sam2VideoProcessor.from_pretrained("facebook/sam2.1-hiera-tiny")
        elif model == 'sam3':
            self.sam_model = Sam3TrackerVideoModel.from_pretrained("facebook/sam3").to(device, dtype=torch.bfloat16)
            self.sam_processor = Sam3TrackerVideoProcessor.from_pretrained("facebook/sam3")
        else:
            raise Exception(f"Unknown segmentation model: {model}")

        self.sam_inference_session = None
        self.device = device

    @torch.no_grad()
    def initialize(self, image_shape, input_points, input_labels):
        self.sam_inference_session = self.sam_processor.init_video_session(
            inference_device=self.device,
            dtype=torch.bfloat16,
        )

        # Add point input on first frame
        self.sam_processor.add_inputs_to_inference_session(
            inference_session=self.sam_inference_session,
            frame_idx=0,
            obj_ids=1,
            input_points=input_points,
            input_labels=input_labels,
            original_size=image_shape,  # need to be provided when using streaming video inference
        )

    @torch.no_grad()
    def estimate_mask(self, image):
        if self.sam_inference_session is None:
            raise Exception("Segmentation model not initialized.")

        sam_inputs = self.sam_processor(images=image, device=self.device, return_tensors="pt")
        # Process current frame
        sam_tracker_video_output = self.sam_model(inference_session=self.sam_inference_session, frame=sam_inputs.pixel_values[0])
        video_res_masks = self.sam_processor.post_process_masks(
            [sam_tracker_video_output.pred_masks], original_sizes=sam_inputs.original_sizes, binarize=False
        )[0]
        mask = video_res_masks[0][0] > 0
        return mask
