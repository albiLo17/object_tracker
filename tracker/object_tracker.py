import os
import sys

# append to the system path the path to the vision folder
folder_path = os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), '..'), '..'))
sys.path.append(folder_path)
from tracker.camera_utils import (
    load_camera_extrinsic,
    get_masked_pointcloud,
    transform_points,
    farthest_point_sampling,
    compute_edges_index,
)
from tracker.file_manager import Trajectory
from tracker.fusion_class import Fusion
from tracker.viz import plot_pcd_list, plot_mesh
import copy
import json
import numpy as np
import cv2
import h5py
import copy
from scipy.spatial.transform import Rotation as R
import torch
import matplotlib.pyplot as plt

class Extrinsics:
    def __init__(self):
        self.rot = None
        self.quat = None
        self.t = None

class Intrinsics:
    def __init__(self):
        self.K = None
        self.distortion = None
        self.h = None
        self.w = None

class Camera:

    def __init__(self, camera_name, camera_id, config_path, world_to_sim=True):
        self.camera_name = camera_name
        self.camera_id = camera_id

        self.B_X_C = Extrinsics()
        self.intrinsics = Intrinsics()
        self.load_config(config_path)

        self.world_to_sim = world_to_sim

    def load_config(self, config_path):

        self.intrinsics.K, self.intrinsics.distortion, self.intrinsics.w, self.intrinsics.h = self.load_intrinsics(
            camera_intrinsics_path=os.path.join(config_path, f'{self.camera_name}_intrinsic.json'))

        rotation, quaternion, translation = self.load_extrinsics(camera_extrinsics_path=config_path)
        rotation_inv, t_inv, quaternion_inv = self.get_inverse_transform(
            rotation=rotation, translation=translation
        )
        self.B_X_C.rot = rotation_inv
        self.B_X_C.quat = quaternion_inv
        self.B_X_C.t = t_inv

    def load_intrinsics(self, camera_intrinsics_path):
        with open(camera_intrinsics_path, 'r') as jsonfile:
            config = json.load(jsonfile)
        K = np.asarray(config['K'])
        distortion = np.asarray(config['distortion'])
        w = np.asarray(config['size'][1])
        h = np.asarray(config['size'][0])

        return K, distortion, w, h

    def load_extrinsics(self, camera_extrinsics_path=None):
        file = os.path.join(camera_extrinsics_path, '{}.json'.format(self.camera_name))
        if not os.path.exists(file):
            print(f'Camera extrinsics file not found at {file}')

        with open(file, 'r') as jsonfile:
            config = json.load(jsonfile)
        print(f'Loaded camera config from {file}')

        rotation = np.array(config['Rotation'])
        quaternion = np.array(config['Quaternion'])
        translation = np.array(config['Translation'])

        return rotation, quaternion, translation

    def get_inverse_transform(self, rotation=None, translation=None, quaternion=None):
        assert (rotation is not None) or (quaternion is not None)

        if rotation is None:
            # Convert quaternion to rotation matrix
            rotation = R.from_quat(quaternion).as_matrix()

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
        quaternion_inv = R.from_matrix(rotation_inv).as_quat()

        return rotation_inv, translation_inv, quaternion_inv

    def depth_to_pcd(self, rgb, depth, mask=None):
        if self.world_to_sim:
            transform = self.B_X_C
        else:
            id_trans = np.eye(4)
            transform = Extrinsics()
            transform.rot =  id_trans[:3, :3]
            # transfomr rotation matrix to quaternion with scipy
            r = R.from_matrix(id_trans[:3, :3])
            transform.quat = r.as_quat()
            transform.t = id_trans[:3, 3]

        points_masked, colors_masked = get_masked_pointcloud(rgb, depth, self.intrinsics.K, self.intrinsics.distortion, mask)
        points_transformed = transform_points(copy.deepcopy(points_masked), transform.rot, transform.t)

        return points_transformed, colors_masked

    # TODO: should this be coming from the dataloader??
    def pcd_to_graph(self, pcd, num_samples, norm_threshold):
        sampled_points_indeces = farthest_point_sampling(pcd, num_samples)
        edge_index, faces = compute_edges_index(
            pcd[sampled_points_indeces],
            k=3,
            delaunay=True,
            sim_data=False,
            norm_threshold=norm_threshold,
        )

        return pcd[sampled_points_indeces], edge_index, faces
        
    def get_mesh(self, rgb, depth, mask, num_samples, norm_threshold=0.01):
        points, colors = self.depth_to_pcd(rgb, depth, mask)
        pos, edge_index, faces = self.pcd_to_graph(points, num_samples, norm_threshold)

        return pos, colors, edge_index, faces

# Function to capture user clicks and generate masks
def click_event(event, x, y, flags, params):
    input_point, input_label, label, predictor, rgb_image = params['input_point'], params['input_label'], params['label'], params['predictor'], params['rgb_image']

    if event == cv2.EVENT_LBUTTONDOWN:
        # Append the point to the respective label's points list
        input_point[label].append([x, y])
        input_label[label] = np.array([1] * len(input_point[label]))

        # Predict the mask
        masks, scores, logits = predictor.predict(
            point_coords=np.array(input_point[label]),
            point_labels=input_label[label],
            multimask_output=False
        )

        # Update the final mask to be the last predicted mask
        params['final_mask'][label] = masks[0]

        # Display the mask and the points
        plt.figure(figsize=(10, 10))
        plt.imshow(rgb_image)
        plt.imshow(params['final_mask'][label], cmap='jet', alpha=0.5)

        # Plot the clicked points
        for point in input_point[label]:
            plt.plot(point[0], point[1], 'ro')  # 'ro' makes red points

        plt.title(f"Mask and Points for {label}")
        plt.axis('off')
        plt.show()


class MaskSelectionInterface:
    
    def __init__(self, sam_predictor):
        self.sam_predictor = sam_predictor
        
            
    def generate_mask(self, rgb_image, labels=["cloth", "gripper"]):
    
        rgb_image = copy.deepcopy(rgb_image).astype(np.uint8)
        image = cv2.cvtColor(rgb_image.astype(np.uint8), cv2.COLOR_RGB2BGR)
        # Process the image with the model to create embeddings
        self.sam_predictor.set_image(image)

        # Initialize empty points and labels
        input_point = {label: [] for label in labels}
        input_label = {label: np.array([]) for label in labels}
        final_mask = {label: None for label in labels}

        # Iterate over each label
        for label in labels:
            print(f"Select points for the {label} in the image.")
            while True:
                # Display the image
                cv2.imshow(f"Select points for {label}", image)

                # Set up parameters for the click event
                params = {
                    'input_point': input_point,
                    'input_label': input_label,
                    'label': label,
                    'predictor': self.sam_predictor,
                    'rgb_image': rgb_image,
                    'final_mask': final_mask
                }
                # Set mouse callback function to capture user clicks
                cv2.setMouseCallback(f"Select points for {label}", click_event, param=params)

                # Wait for user input
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                # Ask if the user is satisfied
                satisfied = input(f"Are you satisfied with the mask for {label}? (yes/no): ")
                if satisfied.lower() == 'yes':
                    break

        print("Mask generation complete.")
        return final_mask


    def show_mask(self, mask, ax, random_color=False, borders = True):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask = mask.astype(np.uint8)
        mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        if borders:
            import cv2
            contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
        ax.imshow(mask_image)
    
    def show_points(self, coords, labels, ax, marker_size=375):
        pos_points = coords[labels==1]
        neg_points = coords[labels==0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

    def show_box(self, box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

    def show_masks(self, image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            self.show_mask(mask, plt.gca(), borders=borders)
            if point_coords is not None:
                assert input_labels is not None
                self.show_points(point_coords, input_labels, plt.gca())
            if box_coords is not None:
                # boxes
                self.show_box(box_coords, plt.gca())
            if len(scores) > 1:
                plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
            plt.axis('off')
            plt.show()


class TrackerMultiView:
    def __init__(self, camera_names, camera_config_path, labels=["cloth", "gripper"]):
        self.camera_names = camera_names
        self.cameras = [Camera(camera_name=name, camera_id=i, config_path=camera_config_path) for i, name in enumerate(camera_names)]
        
        self.labels = labels
        
        self.fusion = Fusion(num_cam=len(camera_names)*len(labels), feat_backbone='dinov2', device='cuda:0', dtype=torch.float32)
        
        self.mask_selection_interface = MaskSelectionInterface(sam_predictor=self.fusion.sam_model)
        
        # keep track of the current time
        self.t = 0
        
    def get_init_masks(self, rgb_images, labels):
        masks_cameras = {camera: {} for camera in self.camera_names}
        for camera, rgb_image in zip(self.camera_names, rgb_images):
            final_masks = self.mask_selection_interface.generate_mask(rgb_image, labels=labels)
            masks_cameras[camera] = final_masks
        return masks_cameras
    
    def prepare_xmem_input(self, rgb_images, masks_cameras=None):
        all_rgbs = []
        all_init_masks = []
        
        for camera in self.camera_names:  
            rgbs =[]
            masks = []          
            for label in self.labels:
                rgbs.append(rgb_images[camera])
                
                if masks_cameras is not None:   
                    init_masks = masks_cameras[camera][label] 
                    masks.append(init_masks)                    
            
            all_rgbs += rgbs
            all_init_masks += masks
            
        return all_rgbs, all_init_masks
    
    def init_tracker(self, rgb_images):
        # rgbs passed as a dictionary with camera names as keys
        masks_cameras = self.get_init_masks([rgb_images[camera] for camera in self.camera_names], self.labels)
        
        self.fusion.H, self.fusion.W, _ = rgb_images[self.camera_names[0]].shape
        
        all_rgbs, all_init_masks = self.prepare_xmem_input(rgb_images, masks_cameras)            
        xmem_masks = self.fusion.xmem_process(np.asarray(all_rgbs).astype(np.uint8), torch.from_numpy(np.asarray(all_init_masks).astype(np.uint8)))
        
        return masks_cameras
    
    def get_masks(self, rgb_images):
        if self.t == 0:
            masks_cameras = self.init_tracker(rgb_images)            
        else:
            all_rgbs, _ = self.prepare_xmem_input(rgb_images)
            xmem_cameras = self.fusion.xmem_process(np.asarray(all_rgbs).astype(np.uint8), None)
            
            masks_cameras = {camera: {} for camera in self.camera_names}
            for c, camera in enumerate(self.camera_names): 
                for l, label in enumerate(self.labels):
                    masks_cameras[camera].update({label: xmem_cameras[c*len(self.labels) + l, :, :, 1].cpu().numpy()})            
                        
        self.t += 1
        
        return masks_cameras


if __name__ == '__main__':
    camera_names = ["front_camera", "back_camera"]
    camera_names = ["camera"]
    config_path = '/home/omniverse/workspace/object_tracker/data/marco/config'
    camera = Camera(camera_name=camera_names[0], camera_id=0, config_path=config_path)

    # # Load the trajectory data
    # traj = Trajectory(sample_name='TOWEL', trajectory_id=0, dataset_name='.', camera_names=camera_names)    
    # traj.load_data_from_folder()
    
    # load rgb images from foler
    rgb_folder = '/home/omniverse/workspace/object_tracker/data/marco/bag1/rgb'
    rgb_files = sorted([os.path.join(rgb_folder, f) for f in os.listdir(rgb_folder) if f.endswith('.png')])
    rgbs = {camera_names[0]: []}
    for f in rgb_files:
        rgbs[camera_names[0]].append(cv2.imread(f)[:,:,::-1])
    print("loaded")
        
    
    # set trakcer and mask selection interface
    labels = ["rope"]
    tracker = TrackerMultiView(camera_names=camera_names, camera_config_path=config_path, labels=labels) 
    mask_selection_interface = MaskSelectionInterface(sam_predictor=tracker.fusion.sam_model)
    

    rgb_image = rgbs[camera_names[0]][0]  
    
    # show image 
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb_image)
    plt.show()
    
    
    # get masks at time t=0
    masks_cameras = {camera: {} for camera in camera_names}
    for camera in camera_names:
        rgb_image = rgbs[camera][0]
        final_masks = mask_selection_interface.generate_mask(rgb_image)
        masks_cameras[camera] = final_masks

        
    ################## Testing xmem outside the class ##################
    # intit xmem:
    # rgbs_0 = {camera: traj.rgbs[camera][0] for camera in camera_names}
    # all_rgbs, all_init_masks = tracker.prepare_xmem_input(rgbs_0, masks_cameras)
    # for all rgbs and masks, plot a double sided figure with on the left the rgb and on the right the mask
    # for i in range(len(all_rgbs)):
    #     fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    #     ax[0].imshow(all_rgbs[i])
    #     ax[1].imshow(all_init_masks[i])
    #     plt.show()
    # tracker.fusion.H, tracker.fusion.W, _ = rgbs_0[tracker.camera_names[0]].shape
    
    # xmem_masks = tracker.fusion.xmem_process(np.asarray(all_rgbs).astype(np.uint8), torch.from_numpy(np.asarray(all_init_masks).astype(np.uint8)))
    # tracker.t += 1
    
    # # get masks at time t=1    
    # rgbs_1 = {camera: traj.rgbs[camera][1] for camera in camera_names}
    # all_rgbs_1, _ = tracker.prepare_xmem_input(rgbs_1)
    # xmem_masks = tracker.fusion.xmem_process(np.asarray(all_rgbs_1).astype(np.uint8), None)
    
    
    ################## Testing xmem whitin the class ##################

    all_masks = []
    for t in range(len(rgbs[camera_names[0]])):
        rgbs_t = {camera: rgbs[camera][t] for camera in camera_names}
        
        masks_cameras = tracker.get_masks(rgbs_t)
        print(f'Generated masks for time {t}')
        all_masks.append(masks_cameras)
        
    # get rope masks for camera 0
    rope_masks = [all_masks[t][camera_names[0]]['rope'] for t in range(len(all_masks))]
    
    # apply the mask to the rgb with white background and save in a folder rgb_masked
    rgb_folder_masked = '/home/omniverse/workspace/object_tracker/data/marco/bag1/rgb_masked'
    if not os.path.exists(rgb_folder_masked):
        os.makedirs(rgb_folder_masked)
        
    for t in range(len(rgbs[camera_names[0]])):
        rgb = rgbs[camera_names[0]][t]
        mask = rope_masks[t]
        mask = mask > 0.5
        rgb_masked = np.where(mask[..., None], rgb, 255)
        # convert rgb to bgr
        bgr_masked = cv2.cvtColor(rgb_masked, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(rgb_folder_masked, f'rgb_masked_{t}.png'), bgr_masked)
        
    
        
    # front_camera_id = 0
    # front_cloth_id = len(camera_names)*front_camera_id + 0
    # front_gripper_id = len(camera_names)*front_camera_id + 1
    
    # back_camera_id = 1
    # back_cloth_id = len(camera_names)*back_camera_id + 0
    # back_gripper_id = len(camera_names)*back_camera_id + 1
    
    # t = 0
    # for t in range(len(traj.rgbs[camera_names[0] ] ) ):
    #     fig, ax = plt.subplots(2, 3, figsize=(10, 5))
    #     ax[0][0].imshow(traj.rgbs[camera_names[0]][t])
    #     ax[0][1].imshow(all_masks[t]['front_camera']['cloth'])
    #     ax[0][2].imshow(all_masks[t]['front_camera']['gripper'])
    #     ax[1][0].imshow(traj.rgbs[camera_names[1]][t])
    #     ax[1][1].imshow(all_masks[t][camera_names[1]]['cloth'])
    #     ax[1][2].imshow(all_masks[t][camera_names[1]]['gripper'])
    #     plt.show()
    # print('Done')


    ################ TEST POINTCLOUD ################
    # rgb = np.load(os.path.join(".", 'rgb.npy'))
    # depth = np.load(os.path.join(".", 'depth.npy'))
    # mask = np.load(os.path.join(".", 'mask.npy'))
    
    # pos, colors, edge_index, faces = front_camera.get_mesh(rgb, depth, mask, num_samples=200, norm_threshold=0.01)