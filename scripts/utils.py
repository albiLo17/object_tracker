
from h5py import Dataset, File, Group
import numpy as np
import cv2

PIPER_JOINTS_UPPER_LIMITS = np.deg2rad(np.array([154.0, 195.0, 0.0, 106.0, 75.0, 100.0, 70.0]))  # NOTE: gripper values gotten from a mix of piper manual and observation min max
PIPER_JOINTS_LOWER_LIMITS = np.deg2rad(np.array([-154.0, 0.0, -175.0, -106.0, -75.0, -100.0, -3.0]))
PIPER_VEL_LIMITS = np.deg2rad(np.array([180, 195, 180, 225, 225, 225, 700]))  # NOTE: gripper limit gotten from 70 / 0.1
PIPER_EXCL_OBS_KEYS = ["action", "timestamp", "ee_pose", "joint_vel", "language_instruction"]


def load_content_from_h5_file(file):
    if isinstance(file, (File, Group)):
        return {key: load_content_from_h5_file(file[key]) for key in list(file.keys())}
    elif isinstance(file, Dataset):
        return file[()]
    else:
        raise NotImplementedError(f"Unspported h5 file type: {type(file)}")


def load_hdf5(
    path,
):
    print("Loading HDF5 file", path)
    file = File(path, "r")
    ret = load_content_from_h5_file(file)
    file.close()
    print("Loaded")
    return ret

def load_traj_hdf5(path, num_traj=None):
    print("Loading HDF5 file", path)
    file = File(path, "r")
    keys = list(file.keys())
    if num_traj is not None:
        assert num_traj <= len(keys), f"num_traj: {num_traj} > len(keys): {len(keys)}"
        keys = sorted(keys, key=lambda x: int(x.split("_")[-1]))
        keys = keys[:num_traj]
    ret = {key: load_content_from_h5_file(file[key]) for key in keys}
    file.close()
    print("Loaded")
    return ret

def load_demo_dataset_piper(path, num_traj, control_mode="pd_joint_pos"): 
    raw_data = load_traj_hdf5(path, num_traj)

    if control_mode == "pd_joint_pos":
        dataset = {
            "observations": [{k: v for k, v in raw_data[traj].items() if k not in PIPER_EXCL_OBS_KEYS} for traj in raw_data],
            "actions": [raw_data[traj]["action"][:-1] for traj in raw_data] # NOTE: remove last action since it has no next state
        }
    else:
        raise NotImplementedError(f"{control_mode} control mode not supported yet")

    return dataset




def process_piper_observation(
    obs,
    image_keys=('wrist_rgb_compressed', 'base_rgb_compressed'),
    joints_low_limits=PIPER_JOINTS_LOWER_LIMITS,
    joints_high_limits=PIPER_JOINTS_UPPER_LIMITS,
    decode_images=True,
    image_res_scale=1.0,
):
    """
    Processes a trajectory-formatted observation dictionary with:
    - compressed image arrays of shape (T,)
    - state arrays of shape (T, D)
    Returns:
        dict:
            'rgb': np.ndarray, shape (T, 6, H, W)
            'state': np.ndarray, shape (T, D)
    """
    T = len(obs[image_keys[0]])  # Trajectory length

    rgb_frames = []
    for t in range(T):
        images = []
        for key in image_keys:
            if decode_images:
                compressed = obs[key][t]
                img_bgr = cv2.imdecode(np.frombuffer(compressed, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img_bgr is None:
                    raise ValueError(f"Failed to decode image for key '{key}' at timestep {t}.")
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = obs[key][t]
            
            if image_res_scale != 1.0:
                img_rgb = cv2.resize(
                    img_rgb,
                    None,
                    fx=image_res_scale,
                    fy=image_res_scale,
                    interpolation=cv2.INTER_AREA
                )

            images.append(img_rgb)

        if len(image_keys) == 2 and  images[0].shape[:2] != images[1].shape[:2]:
            raise ValueError("Mismatched image dimensions between image keys.")

        rgb_concat = np.concatenate(images, axis=2)  # (H, W, 6)
        rgb_chw = np.transpose(rgb_concat, (2, 0, 1))  # (6, H, W)
        rgb_frames.append(rgb_chw)

    rgb_tensor = np.stack(rgb_frames, axis=0)  # (T, 6, H, W)

    # Concatenate and normalize joint positions
    joints_pos = np.concatenate([obs[k] for k in ['joint_pos', 'gripper_joint']], axis=-1)
    joints_pos_norm = (joints_pos - joints_low_limits) / (joints_high_limits - joints_low_limits + 1e-8)


    return {
        'rgb': rgb_tensor,
        'state': joints_pos_norm,
    }