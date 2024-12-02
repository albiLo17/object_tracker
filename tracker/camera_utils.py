import os
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import DBSCAN
from scipy.spatial.transform import Rotation
import pdb
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
import copy
from scipy.spatial import cKDTree, Delaunay, KDTree
import torch
# def plot_pcd(points, colors=None):
#     """Function that plots a 3D pointcloud with colors if colors is not None"""
#     fig = plt.figure(figsize=(10, 10))
#     ax = fig.add_subplot(111, projection='3d')
#
#     # Re-scale colors to [0, 1]
#     if colors is not None:
#         colors = colors / 255.0
#
#     # Plot using a scatter plot and color by the RGB colors
#     ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c=colors, marker='.')
#
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#
#     plt.show()



##################################################



def extend_mask(mask, depth, roi):
    mask_full = np.zeros_like(depth)
    if not np.any(mask == None):
        mask_full[roi[0]:roi[1], roi[2]:roi[3]] = mask
    return mask_full

def get_masked_pointcloud(rgb, depth, K, distortion, mask):
    points, colors, valid_depth = rgb_to_pcd2(rgb, depth, K, distortion, mask=None, plot=False)
    # pdb.set_trace()
    cloth_mask = mask.flatten()
    cloth_mask = cloth_mask[valid_depth]
    points_masked = points[cloth_mask == 1]
    colors_masked = colors[cloth_mask == 1]

    return points_masked, colors_masked

def pointcloud_projection_with_mask_index(rgb, depth, k, distortion, mask,
                          rotation=np.eye(3), translation=np.zeros(3),):
    points, colors, pixel_indices, valid_indices = rgb_to_pcd_with_index(rgb, depth, k, distortion, mask=mask, plot=False)
    points_transformed = transform_points(copy.deepcopy(points), rotation, translation)

    return points_transformed, colors, pixel_indices, valid_indices


def pointcloud_processing(rgb, depth, k, distortion, mask,
                          rotation=np.eye(3), translation=np.zeros(3),
                          remove_outliers=False, voxelize=False, voxel_size=0.008):

    points, colors = get_masked_pointcloud(rgb, depth, K=k, distortion=distortion, mask=mask)
    points_transformed = transform_points(copy.deepcopy(points), rotation, translation)



    if len(points_transformed) > 0 and remove_outliers:
        points_transformed, colors = remove_outliers_dbscan(
            points_transformed, colors, eps=0.01, min_samples=50)

    if len(points_transformed) > 0 and voxelize:
        points_transformed, colors = voxel_downsample(
            points_transformed, colors, voxel_size=voxel_size)

    return points_transformed, colors


def compute_iou(set1, set2, grid_size=0.01):
    try:
        # Project 3D points onto 2D by ignoring the Z dimension
        set1_2d = set1[:, :2]
        set2_2d = set2[:, :2]

        # Compute axis-aligned bounding boxes (AABB)
        min_bound = np.minimum(np.min(set1_2d, axis=0), np.min(set2_2d, axis=0))
        max_bound = np.maximum(np.max(set1_2d, axis=0), np.max(set2_2d, axis=0))

        # Create occupancy grids
        x_grid = np.arange(min_bound[0], max_bound[0], grid_size)
        y_grid = np.arange(min_bound[1], max_bound[1], grid_size)
        grid1 = np.zeros((len(x_grid), len(y_grid)), dtype=bool)
        grid2 = np.zeros((len(x_grid), len(y_grid)), dtype=bool)

        # Mark occupied cells in the grids
        for x in set1_2d:
            i, j = int((x[0] - min_bound[0]) / grid_size), int((x[1] - min_bound[1]) / grid_size)
            grid1[i, j] = True
        for x in set2_2d:
            i, j = int((x[0] - min_bound[0]) / grid_size), int((x[1] - min_bound[1]) / grid_size)
            grid2[i, j] = True

        # Compute intersection and union
        intersection = np.logical_and(grid1, grid2)
        union = np.logical_or(grid1, grid2)
        inter_area = np.sum(intersection)
        union_area = np.sum(union)

        # Compute the IoU
        iou = inter_area / union_area if union_area != 0 else 0
    except:
        iou = 0

    return iou


def voxel_downsample(points, colors=None, voxel_size=0.005):
    """
    Perform voxel downsampling on a point cloud and also include the colors.

    :param points: NumPy array of points (Nx3).
    :param colors: NumPy array of colors corresponding to the points (Nx3).
    :param voxel_size: Size of the voxel in which to downsample.
    :return: Downsampled point cloud points and colors as NumPy arrays.
    """


    # Convert numpy points and colors to Open3D point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        # Check if the number of points and colors match
        if points.shape[0] != colors.shape[0]:
            raise ValueError("The number of points and colors must be the same.")
        point_cloud.colors = o3d.utility.Vector3dVector(colors)

    # Perform voxel downsampling
    downsampled_point_cloud = point_cloud.voxel_down_sample(voxel_size)

    # Convert back to numpy arrays
    downsampled_points = np.asarray(downsampled_point_cloud.points)
    if colors is not None:
        downsampled_colors = np.asarray(downsampled_point_cloud.colors)
        return downsampled_points, downsampled_colors

    return downsampled_points

def load_camera_config(camera_name, dir_path=None, file_path=None, optitrack=False, link_frame=False):

    if dir_path is None:
        dir_path = os.path.join(os.path.dirname(os.getcwd()), "config")
    else:
        if file_path is None:
            if not optitrack:
                file = os.path.join(dir_path, '{}.json'.format(camera_name))
            else:
                if not link_frame:
                    file = os.path.join(dir_path, '{}_optitrack.json'.format(camera_name))
                else:
                    file = os.path.join(dir_path, '{}_link_optitrack.json'.format(camera_name))
            with open(file, 'r') as jsonfile:
                config = json.load(jsonfile)

            print(f'Loaded camera config from {file}')
        else:
            with open(file_path, 'r') as jsonfile:
                config = json.load(jsonfile)
            print(f'Loaded camera config from {file_path}')

    rotation = np.array(config['Rotation'])
    quaternion = np.array(config['Quaternion'])
    translation = np.array(config['Translation'])

    return rotation, quaternion, translation

def load_camera_intrinsics(camera_intrinsics_path=None, file_path=None):
    with open(camera_intrinsics_path, 'r') as jsonfile:
        config = json.load(jsonfile)
    K = np.asarray(config['K'])
    distortion = np.asarray(config['distortion'])


    return K, distortion

def load_camera_extrinsic(optitrack=True, config_path = '../config'):
    # Camera configs:
    optitrack = optitrack
    k = {}
    distortions = {}
    B_X_C = {}
    for camera_name in ['front_camera', 'back_camera']:
        K, distortion = load_camera_intrinsics(
            camera_intrinsics_path=os.path.join(config_path, f'{camera_name}_intrinsic.json'))
        k[camera_name] = K
        distortions[camera_name] = distortion
        # extrinsic
        rotation, quaternion, translation = load_camera_config(camera_name, dir_path=config_path, optitrack=optitrack)
        rotation_inv, t_inv, quaternion_inv = get_inverse_transform(rotation=rotation, translation=translation)
        B_X_C[camera_name] = {'rot': rotation_inv,
                 'quat': quaternion_inv,
                 't': t_inv}

    return k, distortions, B_X_C

def remove_outliers_dbscan(point_cloud, colors, eps=0.5, min_samples=10):
    """
    Remove outliers from a point cloud using DBSCAN clustering.

    :param point_cloud: NumPy array of shape (N, 3) representing the point cloud.
    :param eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    :param min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
    :return: Filtered point cloud as a NumPy array.
    """
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(point_cloud)
    labels = clustering.labels_

    # Points with label -1 are outliers
    filtered_indices = labels != -1
    return point_cloud[filtered_indices], colors[filtered_indices]

def get_inverse_transform(rotation=None, translation=None, quaternion=None):
    assert (rotation is not None) or (quaternion is not None)

    if rotation is None:
        # Convert quaternion to rotation matrix
        rotation = Rotation.from_quat(quaternion).as_matrix()

    # Create 4x4 transformation matrix
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = translation

    # Compute inverse transformation matrix
    T_inv = np.linalg.inv(T)

    # Extract inverse rotation and translation
    rotation_inv = T_inv[:3, :3]
    translation_inv = T_inv[:3, 3]

    # Convert inverse rotation matrix to quaternion
    quaternion_inv = Rotation.from_matrix(rotation_inv).as_quat()

    return rotation_inv, translation_inv, quaternion_inv



def rgbd_to_pcd(rgb_image, depth_image, K, distortion):
    # Undistort images
    h, w = rgb_image.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K, distortion, (w, h), 1, (w, h))

    # Undistort
    mapx, mapy = cv2.initUndistortRectifyMap(K, distortion, None, new_camera_matrix, (w, h), 5)
    undistorted_rgb_image = cv2.remap(rgb_image, mapx, mapy, cv2.INTER_LINEAR)
    undistorted_depth_image = cv2.remap(depth_image, mapx, mapy, cv2.INTER_LINEAR)

    # Compute point cloud
    points = []
    colors = []

    for v in range(h):
        for u in range(w):
            color = undistorted_rgb_image[v, u]
            Z = undistorted_depth_image[v, u]
            if Z == 0:  # Skip no depth information
                continue
            X = (u - K[0, 2]) * Z / K[0, 0]
            Y = (v - K[1, 2]) * Z / K[1, 1]
            points.append((X, Y, Z))
            colors.append(color)

    # Convert to numpy arrays
    points = np.array(points)
    colors = np.array(colors)

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Downsample points if there are too many to plot efficiently
    # TODO: eventually reintegrate this
    # if len(points) > 100000:
    #     indices = np.random.choice(len(points), 100000, replace=False)
    #     points = points[indices]
    #     colors = colors[indices]

    # Re-scale colors to [0, 1]
    colors = colors / 255.0

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c=colors, marker='.')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_z_label('Z')
    plt.show()

def transform_points(points, rotation, translation):
    """
    Apply a transformation to a set of points given rotation and translation in an efficient way
    :param points: set of points to be transformed
    :param rotation: rotation matrix
    :param translation: translation vector
    :return: transformed points
    """
    # pdb.set_trace()
    transformation = np.eye(4)
    transformation[:3, :3] = rotation
    transformation[:3, 3] = translation

    # Convert to homogeneous coordinates
    num_points = points.shape[0]
    points_homogeneous = np.ones((num_points, 4))
    points_homogeneous[:, :3] = points

    # Apply transformation
    transformed_points_homogeneous = np.dot(transformation, points_homogeneous.T).T

    # Convert back to 3D
    transformed_points = transformed_points_homogeneous[:, :3]

    return transformed_points

def rgb_to_pcd_with_index(rgb_image, depth_image, K, distortion, mask=None, plot=False):
    h, w = rgb_image.shape[:2]

    # Generate grid of coordinates
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    # Flatten the grid to single column vectors
    u = u.flatten()
    v = v.flatten()

    # Flatten the depth image to single column vector
    depth = depth_image.flatten()

    # Apply the mask if provided
    if mask is not None:
        mask_flat = mask.flatten()
        mask_valid = mask_flat > 0
    else:
        mask_valid = np.ones(depth.shape, dtype=bool)

    # Mask out depth values of zero
    valid_depth = (depth > 0) & mask_valid

    # Store the indices of valid depth points
    valid_indices = np.where(valid_depth)[0]

    # Pick only the valid points and colors
    u_valid = u[valid_depth]
    v_valid = v[valid_depth]
    depth_valid = depth[valid_depth]
    colors_valid = rgb_image[v_valid, u_valid]  # Use advanced numpy indexing to get colors

    # Convert depth from uint16 to float and scale it if necessary (e.g., millimeters to meters)
    depth_valid = depth_valid.astype(np.float32) / 1000.0

    # Reproject to 3D space by applying the inverse of the intrinsic matrix
    x = (u_valid - K[0, 2]) * depth_valid / K[0, 0]
    y = (v_valid - K[1, 2]) * depth_valid / K[1, 1]
    z = depth_valid

    # Stack to homogeneous coordinates
    points_3D = np.vstack((x, y, z)).transpose()

    # Store the valid pixel indices for mapping
    pixel_indices = np.vstack((u_valid, v_valid)).transpose()

    if plot:
        colors_valid = colors_valid[points_3D[:, 2] < 0.8]
        points_3D = points_3D[points_3D[:, 2] < 0.8]

        # Plotting
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Re-scale colors to [0, 1]
        colors_valid = colors_valid / 255.0

        # Plot using a scatter plot and color by the RGB colors
        ax.scatter(points_3D[:, 0], points_3D[:, 1], points_3D[:, 2], s=1, c=colors_valid, marker='.')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()

    return points_3D, colors_valid, pixel_indices, valid_indices

def rgb_to_pcd2(rgb_image, depth_image, K, distortion, mask=None, plot=False):
    # Undistort the RGB image
    h, w = rgb_image.shape[:2]
    # new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K, distortion, (w, h), 1, (w, h))
    # undistorted_rgb_image = cv2.undistort(rgb_image, K, distortion, None, new_camera_matrix)

    # Generate grid of coordinates
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    # Flatten the grid to single column vectors
    u = u.flatten()
    v = v.flatten()

    # Flatten the depth image to single column vector
    depth = depth_image.flatten()

    # Apply the mask if provided
    if mask is not None:
        mask_flat = mask.flatten()
        mask_valid = mask_flat > 0
    else:
        mask_valid = np.ones(depth.shape, dtype=bool)

    # Mask out depth values of zero
    valid_depth = (depth > 0) & mask_valid

    # Pick only the valid points and colors
    # pdb.set_trace()
    u = u[valid_depth]
    v = v[valid_depth]
    depth = depth[valid_depth]
    colors = rgb_image[v, u]  # Use advanced numpy indexing to get colors
    # colors = undistorted_rgb_image[v, u]  # Use advanced numpy indexing to get colors

    # Convert depth from uint16 to float and scale it if necessary (e.g., millimeters to meters)
    depth = depth.astype(np.float32) / 1000.0

    # Reproject to 3D space by applying the inverse of the intrinsic matrix
    # To speed up, precompute the inversion of the intrinsic matrix
    # K_inv = np.linalg.inv(K)
    x = (u - K[0, 2]) * depth / K[0, 0]
    y = (v - K[1, 2]) * depth / K[1, 1]
    z = depth

    # Stack to homogeneous coordinates
    points_3D = np.vstack((x, y, z)).transpose()

    # TODO: eventually reintegrate this
    # if len(points_3D) > 100000:
    #     indices = np.random.choice(len(points_3D), 100000, replace=False)
    #     points = points_3D[indices]
    #     colors = colors[indices]
    # else:
    #     points = points_3D
    points = points_3D

    if plot:
        colors = colors[points[:, 2] < 0.8]
        points = points[points[:, 2] < 0.8]

        # Plotting
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Re-scale colors to [0, 1]
        colors = colors / 255.0

        # Plot using a scatter plot and color by the RGB colors
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c=colors, marker='.')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()

    return points, colors, valid_depth

def plot_pcd(pcd_list, elev=0, azim=0, limits=True, return_fig=False, centered=False):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(pcd_list)):
        points, colors = pcd_list[i]
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c=colors / 255, marker='.')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # add axes limits
    if limits:
        if centered:
            ax.set_xlim3d(-0.4, 0.4)
            ax.set_ylim3d(-0.4, 0.4)
            ax.set_zlim3d(-0.4, 0.4)
        else:
            ax.set_xlim3d(0, 1.0)
            ax.set_ylim3d(-0.5, 0.5)
            ax.set_zlim3d(-0.5, 0.5)

    # set elev and azim to rotate the plot
    ax.view_init(elev=elev, azim=azim)
    if return_fig:
        return fig
    plt.show()

def masked_depth_to_3d_points(depth_image, color_image, mask, K):
    """
    Convert depth image and color image to 3D points using a mask.

    :param depth_image: Depth image (2D numpy array).
    :param color_image: Color image (3D numpy array, HxWx3).
    :param mask: Binary mask (2D numpy array, same size as depth_image).
    :param K: Intrinsic camera matrix (3x3 numpy array).
    :return: 3D points (Nx3 numpy array), Colors (Nx3 numpy array).
    """

    # Ensure mask is boolean
    # mask = mask > 0

    # Get pixel coordinates
    height, width = depth_image.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))

    # Flatten the arrays
    u_flattened = u.flatten()
    v_flattened = v.flatten()
    depth_flattened = depth_image.flatten()

    # Convert depth from uint16 to float and scale it if necessary
    depth_flattened = depth_flattened.astype(np.float32) / 1000.0  # Assuming depth in millimeters

    # Reproject to 3D space
    x = (u_flattened - K[0, 2]) * depth_flattened / K[0, 0]
    y = (v_flattened - K[1, 2]) * depth_flattened / K[1, 1]
    z = depth_flattened

    # Stack to homogeneous coordinates
    points_3D = np.vstack((x, y, z)).transpose()

    # Flatten and mask the color image
    colors = color_image.reshape(-1, 3)
    mask_flattened = mask.flatten()

    # Apply mask
    points_3D_masked = points_3D[mask_flattened]
    colors_masked = colors[mask_flattened]

    return points_3D_masked, colors_masked


def farthest_point_sampling(points, num_samples):
    """
    Selects a subset of points using the Farthest Point Sampling (FPS) algorithm.

    Parameters:
    - points: A NumPy array of shape (N, D) where N is the number of points and D is the dimensionality.
    - num_samples: The number of points to select.

    Returns:
    - A NumPy array of the selected points.
    """
    # Initialize an array to hold indices of the selected points
    selected_indices = np.zeros(num_samples, dtype=int)
    # The first point is selected randomly
    selected_indices[0] = np.random.randint(len(points))
    # Initialize a distance array to track the shortest distance of each point to the selected set
    distances = np.full(len(points), np.inf)

    # Loop to select points
    for i in range(1, num_samples):
        # Update the distances based on the newly added point
        dist_to_new_point = np.linalg.norm(points - points[selected_indices[i - 1]], axis=1)
        distances = np.minimum(distances, dist_to_new_point)
        # Select the point with the maximum distance to the set of selected points
        selected_indices[i] = np.argmax(distances)

    # Return the selected points
    return selected_indices


def compute_edges_index(points, k=3, delaunay=False, sim_data=False, norm_threshold=0.01):
    if delaunay:
        if sim_data:
            points2d = points[:, [0,2]]
        else:
            points2d = points[:, :2]
        tri = Delaunay(points2d)
        edges = set()
        faces = []
        for simplex in tri.simplices:
            valid_face = True
            current_edges = []

            for i in range(3):
                p1, p2 = simplex[i], simplex[(i + 1) % 3]
                edge = (min(p1, p2), max(p1, p2))
                current_edges.append(edge)
                # Calculate the norm (distance) between the points
                norm = np.linalg.norm(points2d[p1] - points2d[p2])

                # Check if the edge meets the threshold condition
                if norm_threshold is not None and norm > norm_threshold:
                    valid_face = False
                else:
                    edges.add(edge)

            # Add the face if all edges are valid
            if valid_face:
                faces.append(simplex)

        edge_index = np.asarray(list(edges))
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        # Convert faces list to a tensor
        faces = torch.tensor(np.asarray(faces), dtype=torch.long).t().contiguous()
        return edge_index, faces
    else:
        # Use a k-D tree for efficient nearest neighbors computation
        tree = cKDTree(points)
        # For simplicity, we find the 3 nearest neighbors; you can adjust this number
        _, indices = tree.query(points, k=k+1)

        # Skip the first column because it's the point itself
        edge_index = np.vstack({tuple(sorted([i, j])) for i, row in enumerate(indices) for j in row[1:]})
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return edge_index




if __name__=='__main__':
    camera_intrinsics_path = '/home/marco/franka_ws/src/vision/config/front_camera_intrinsic.json'
    with open(camera_intrinsics_path, 'r') as jsonfile:
        config = json.load(jsonfile)
    K = np.asarray(config['K'])
    distortion = np.asarray(config['distortion'])

    rgb_image_path = '/home/marco/franka_ws/src/vision/src/data/rgb.png'
    depth_image_path = '/home/marco/franka_ws/src/vision/src/data/depth.png'

    # Load images
    rgb_image = cv2.imread(rgb_image_path)
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)  # Assuming 16-bit depth image

    rgb_to_pcd2(rgb_image, depth_image, K, distortion)

    xyz_pcd = np.load('/home/marco/franka_ws/src/vision/src/data/xyz_pcd.npy')
    rgb_pcd = np.load('/home/marco/franka_ws/src/vision/src/data/rgb_pcd.npy')

    if len(xyz_pcd) > 100000:
        indices = np.random.choice(len(xyz_pcd), 100000, replace=False)
        points = xyz_pcd[indices]
        colors = rgb_pcd[indices]

    colors = colors[points[:, 2] < 0.8]
    points = points[points[:, 2] < 0.8]

    # Plotting
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Re-scale colors to [0, 1]
    colors = colors / 255.0

    # Plot using a scatter plot and color by the RGB colors
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c=colors, marker='.')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()
    print()