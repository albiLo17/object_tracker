import numpy as np
import math
import cv2
import os
import json
import glob
import sys
import h5py
# append to the system path the path to the vision folder
folder_path = os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), '..'), '..'))
sys.path.append(folder_path)
import matplotlib.pyplot as plt
import torch
import torch_geometric

# append to the system path the path to the vision folder
folder_path = os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), '..'), '..'))
sys.path.append(folder_path)

def get_mesh_data(points, faces):
    mesh = torch_geometric.data.Data(pos=points, face=faces)
    mesh = torch_geometric.transforms.FaceToEdge(remove_faces=False)(mesh)
    mesh = torch_geometric.transforms.GenerateMeshNormals()(mesh)

    return mesh


class Trajectory:
    
    def __init__(self, sample_name, trajectory_id, dataset_name, camera_names):
        self.sample_name = sample_name
        self.trajectory_id = trajectory_id
        self.camera_names = camera_names
        
        # TODO: this is in case we need to load from folder
        self.dataset_path =  f'{dataset_name}/data/{sample_name}/{trajectory_id:05}'
        self.data_traj_path = os.path.join(self.dataset_path, 'trajectory')
        
        for path in [self.dataset_path, self.data_traj_path]:
            if not os.path.exists(path):
                os.makedirs(path)
        
        self.rgbs = {}
        self.depths = {}
        self.masks = {}
        self.gripper_masks = {}
        self.init_trajectory()
        
        self.mesh_predictions = []
        self.mesh_refinements = []
        self.init_mesh = None
        self.goal_mesh = None
        self.edge_faces = None
        self.edge_index = None
        
        # # other relevant features we want in trajectory
        # self.pos = []
        # self.gripper_pos = []
        # self.actions = []
        # self.done = []
        # self.grasped_partilce = None
        # self.pick = None
        # self.place = None
            
    def init_trajectory(self,):
        self.rgbs = {camera: [] for camera in self.camera_names}
        self.depths = {camera: [] for camera in self.camera_names}
        self.masks = {camera: [] for camera in self.camera_names}
        self.gripper_masks =  {camera: [] for camera in self.camera_names}
        
    def add_data(self, camera_name, rgb, depth, mask=None, gripper_mask=None):
        self.rgbs[camera_name].append(rgb)
        self.depths[camera_name].append(depth)
        if mask is not None:
            self.masks[camera_name].append(mask)
        if gripper_mask is not None:
            self.gripper_masks[camera_name].append(gripper_mask)
            
    def add_mesh(self, mesh, type='init'):
        if type == 'init':
            self.init_mesh = mesh
        elif type == 'goal':
            self.goal_mesh = mesh
        elif type == 'prediction':
            self.mesh_predictions.append(mesh)
        elif type == 'refinement':
            self.mesh_refinements.append(mesh)
        else:
            print('Type not recognized')
            
    def load_data_from_folder(self, dataset_path=None, time_id=-1):
        # this should point to the folder where the data is stored
        if dataset_path:
            datapoint_paths = glob.glob(f'{dataset_path}/data*')
        else:
            datapoint_paths = glob.glob(f'{self.data_traj_path}/data*')
        datapoint_paths.sort()
        
        for camera_name in self.camera_names:
            self.load_camera_data(datapoint_paths, camera_name, time_id=time_id)

            
    def load_camera_data(self, datapoint_paths, camera_name, time_id=-1):
            if time_id == -1:
                # load all 
                for t in range(len(datapoint_paths)):
                    rgb, depth = self.load_images(datapoint_paths[t], camera_name=camera_name)
                    self.rgbs[camera_name].append(rgb)
                    self.depths[camera_name].append(depth)
                    print("Loading masks from folder not implemented yet")
                    # self.masks.append(masks)
            else:
                rgb, depth = self.load_images(datapoint_paths[time_id], camera_name=camera_name)
                self.rgbs[camera_name].append(rgb)
                self.depths[camera_name].append(depth)
                print("Loading masks from folder not implemented yet")
                
    def load_images(self, datapoint_path, camera_name='top_camera'):
        
        f = h5py.File(datapoint_path, 'r')
        if f'{camera_name}' in list(f.keys()):
            rgbd = np.asarray(f[f'{camera_name}'])
            depth = rgbd[:, :, 3]
            rgb = rgbd[:, :, :3]
            return rgb, depth
            
        print(f'{camera_name} not found')
        return None, None

class FileManager:

    def __init__(self, cameras, save_processed_data, num_frames=1, clean_folders=True):
        self.all_transforms = {'train':{}, 'test':{}}
        self.camera_names = [camera.camera_name for camera in cameras]
        self.cameras_ids = {camera.camera_name: camera.camera_id for camera in cameras}

        self.main_folder = save_processed_data
        if self.main_folder[0] != '/':
            self.main_folder = os.getcwd() + self.main_folder[1:]
        self.save_processed_rgb = os.path.join(save_processed_data, 'rgb')                # main folder /rgb
        self.save_processed_masks = os.path.join(save_processed_data, 'masks')             # main folder /masks
        self.save_processed_gripper_masks = os.path.join(save_processed_data, 'gripper_masks')    #   main folder /gripper_masks
        self.save_mesh_predictions = os.path.join(save_processed_data, 'mesh_predictions')    #   main folder /gripper_masks
        self.save_processed_train = os.path.join(save_processed_data, 'train')  #   main folder /gripper_masks
        self.save_processed_test = os.path.join(save_processed_data, 'test')  #   main folder /gripper_masks
        self.save_gaussians = os.path.join(self.main_folder, 'gaussians')  #   main folder /gripper_masks
        # delete the folders if they exist
        if clean_folders:
            for path in [
                self.save_processed_rgb,
                self.save_processed_masks,
                self.save_mesh_predictions,
                self.save_processed_gripper_masks,
                self.save_processed_train,
                self.save_processed_test,
                self.save_gaussians,
            ]:
                if os.path.exists(path):
                    os.system(f'rm -r {path}')
        os.makedirs(self.save_processed_rgb, exist_ok=True)
        os.makedirs(self.save_processed_masks, exist_ok=True)
        os.makedirs(self.save_processed_gripper_masks, exist_ok=True)
        os.makedirs(self.save_processed_train, exist_ok=True)
        os.makedirs(self.save_processed_test, exist_ok=True)
        os.makedirs(self.save_mesh_predictions, exist_ok=True)
        os.makedirs(self.save_gaussians, exist_ok=True)

        self.save_processed_transforms = save_processed_data

        self.all_transforms = { 'train':{}, 'test':{}}
        self.K, self.w, self.h, self.transforms = {}, {}, {}, {}
        for camera in cameras:
            camera_name = camera.camera_name
            self.K[camera_name] = camera.intrinsics.K
            self.w[camera_name]  = camera.intrinsics.w
            self.h[camera_name] = camera.intrinsics.h
            self.transforms[camera_name] = camera.B_X_C

        self.init_transform_file(num_frames)

        # this will be initialized once and stay the same for all the frames
        self.edge_faces = None

    def init_transform_file(self, n_frames):

        for key in self.transforms.keys():       

            self.all_transforms.update({key:{'camera_angle_x': 0, 'camera_angle_y': 0, 'fl_x': 0, 'fl_y': 0, 'w': 0, 'h': 0, 'cx': 0, 'cy': 0, 'n_frames': 0,  'transform_matrix': 0}})
            self.all_transforms[key]['camera_angle_x'] = 2 * math.atan(self.w[key] / (2 * self.K[key][0,0]))
            self.all_transforms[key]['camera_angle_y'] =  2 * math.atan(self.h[key] / (2 * self.K[key][1,1]))
            self.all_transforms[key]['fl_x'] = self.K[key][0,0]
            self.all_transforms[key]['fl_y'] = self.K[key][1,1]
            self.all_transforms[key]['w'] = int(self.w[key])
            self.all_transforms[key]['h'] = int(self.h[key])
            self.all_transforms[key]['cx'] = self.K[key][0,2]
            self.all_transforms[key]['cy'] = self.K[key][1,2]

            self.all_transforms[key]['n_frames'] = n_frames

            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = self.transforms[key].rot@np.asarray([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
            transform_matrix[:3, 3] = self.transforms[key].t

            self.all_transforms[key]['transform_matrix'] = transform_matrix.tolist()

        # average all the camera intrinsics
        for split in ['train', 'test']:
            self.all_transforms[split].update(
            {
            "camera_angle_x": np.asarray([self.all_transforms[key]['camera_angle_x'] for key in self.transforms.keys()]).mean(),
            "camera_angle_y": np.asarray([self.all_transforms[key]['camera_angle_y'] for key in self.transforms.keys()]).mean(),
            "fl_x": np.asarray([self.all_transforms[key]['fl_x'] for key in self.transforms.keys()]).mean(),
            "fl_y": np.asarray([self.all_transforms[key]['fl_y'] for key in self.transforms.keys()]).mean(),
            "w": int(np.asarray([self.all_transforms[key]['w'] for key in self.transforms.keys()]).mean()),
            "h": int(np.asarray([self.all_transforms[key]['h'] for key in self.transforms.keys()]).mean()),
            "cx": np.asarray([self.all_transforms[key]['cx'] for key in self.transforms.keys()]).mean(),
            "cy": np.asarray([self.all_transforms[key]['cy'] for key in self.transforms.keys()]).mean(),
            "n_frames": n_frames,
            "frames": []
            }
            )

    def update_folder(self, rgbs, masks=None, masks_gripper=None,):

        num_frames = len(rgbs[self.camera_names[0]])
        self.init_transform_file(num_frames)

        for camera in self.camera_names:
            time = 0.
            for t in range(num_frames):
                # add config to the all transforms
                for split in ['train', 'test']:
                    # init_dict = copy.deepcopy(all_transforms[camera])
                    init_dict = {}
                    init_dict.update({'file_path': f'./{split}/r_{self.cameras_ids[camera]}_{t}.png'})
                    init_dict.update({'time': min(time, 1.)})
                    init_dict.update({'transform_matrix': self.all_transforms[camera]['transform_matrix']})
                    init_dict.update({'type': 'wrap'})

                    self.all_transforms[split]['frames'].append(init_dict)

                if num_frames > 1:
                    time += 1/(num_frames-1)
                else:
                    time = 0.

                # save rgbs with correct name
                bgr = cv2.cvtColor(rgbs[camera][t], cv2.COLOR_RGB2BGR)

                output_name = f'{self.save_processed_rgb}/r_{self.cameras_ids[camera]}_{t}.png'

                # check first that the image is not there already, if not, process all
                if not os.path.exists(output_name):
                    cv2.imwrite(output_name, bgr.astype(np.uint8))

                    # save masks with correct names
                    # TODO: maske sure that the saving is correct
                    if masks is not None:
                        # convert bool mask to numpy
                        mask = (masks[camera][t]>0).astype(np.uint8)
                        cv2.imwrite(f'{self.save_processed_masks}/r_{self.cameras_ids[camera]}_{t}.png', mask*255)
                        if masks_gripper is not None:
                            g_mask = masks_gripper[camera][t].astype(np.uint8)
                            cv2.imwrite(f'{self.save_processed_gripper_masks}/r_{self.cameras_ids[camera]}_{t}.png', g_mask*255)

                            # remove the gripper from the original mask
                            mask = mask - g_mask
                            mask[mask == 255] = 0

                        # apply the mask
                        # extend the mask to the 3 channels
                        mask = mask[:, :, None]
                        mask = np.repeat(mask, 3, axis=2)
                        bgr = bgr * mask
                        # set to white the masked area
                        bgr[mask == 0] = 255

                    output_name = f'{self.save_processed_train}/r_{self.cameras_ids[camera]}_{t}.png'
                    cv2.imwrite(output_name, bgr.astype(np.uint8))

                    output_name = f'{self.save_processed_test}/r_{self.cameras_ids[camera]}_{t}.png'
                    cv2.imwrite(output_name, bgr.astype(np.uint8))

        # save the transforms as json
        for split in ['train', 'test']:
            with open(f'{self.save_processed_transforms}/transforms_{split}.json', 'w') as jsonfile:
                json.dump(self.all_transforms[split], jsonfile, indent=4)

        print('All data saved in folder {}'.format(self.save_processed_transforms))


    def save_init_mesh(self, pos, edge_faces):
        if not isinstance(pos, torch.Tensor):            
            mesh = get_mesh_data(torch.from_numpy(pos).to(torch.float32), edge_faces)
        else:
            mesh = get_mesh_data(pos.to(torch.float32), edge_faces)
        self.edge_faces = edge_faces
        # meshg name should be formatted as f"mesh_{t:03}.hdf5", where t is the init time
        mesh = mesh.to_dict()
        with h5py.File(os.path.join(self.main_folder, "init_mesh.hdf5"), "w") as f:
            for key, value in mesh.items():
                f.create_dataset(key, data=value.detach().cpu().numpy())
                
    def save_goal_mesh(self, pos):
        if self.edge_faces is None:
            print("You should first initialize the first mesh. Use the method save_init_mesh.")
        # check if pos is a torch already
        if not isinstance(pos, torch.Tensor):
            mesh = get_mesh_data(torch.from_numpy(pos).to(torch.float32), self.edge_faces)
        else:
            mesh = get_mesh_data(pos.to(torch.float32), self.edge_faces)
        # meshg name should be formatted as f"mesh_{t:03}.hdf5", where t is the init time
        mesh = mesh.to_dict()
        with h5py.File(os.path.join(self.main_folder, "goal_mesh.hdf5"), "w") as f:
            for key, value in mesh.items():
                f.create_dataset(key, data=value.detach().cpu().numpy())

    def save_mesh(self, pos, name_mesh):
        # meshg name should be formatted as f"mesh_{t:03}.hdf5", where t is the init time
        if self.edge_faces is None:
            print("You should first initialize the first mesh. Use the method save_init_mesh.")
        if not isinstance(pos, torch.Tensor):
            mesh = get_mesh_data(torch.from_numpy(pos).to(torch.float32), self.edge_faces)
        else:
            mesh = get_mesh_data(pos.to(torch.float32), self.edge_faces)
        mesh = mesh.to_dict()
        with h5py.File(os.path.join(self.save_mesh_predictions, name_mesh), "w") as f:
            for key, value in mesh.items():
                f.create_dataset(key, data=value.detach().cpu().numpy())


if __name__ == '__main__':
    
    from tracker.object_tracker import Camera
    
    camera_names = ["front_camera", "back_camera"]
    config_path = '/media/omniverse/Alberta/cloth_splatting/config'
    front_camera = Camera(camera_name=camera_names[0], camera_id=0, config_path=config_path)
    back_camera = Camera(camera_name=camera_names[1], camera_id=1, config_path=config_path)

    
    existing_traj = Trajectory(sample_name='TOWEL', trajectory_id=0, dataset_name='.', camera_names=camera_names)        
    existing_trajectory_path = existing_traj.dataset_path
    # Test trajectory by loading from folder
    existing_traj.load_data_from_folder(dataset_path=None, time_id=-1)
    
    
    # Then test trajectory by adding data
    duplicate_traj = Trajectory(sample_name='TOWEL', trajectory_id=1, dataset_name='.', camera_names=camera_names)
    # duplicate_traj.add_data(camera_name=camera_names[0], rgb=existing_traj.rgbs[camera_names[0]][0], depth=existing_traj.depths[camera_names[0]][0])
    trajectory_path = existing_traj.dataset_path
    

    # plt.imshow(traj.rgbs[traj.camera_names[0]][0])
    # TODO: the cameras for this file mangaer can be obtained from the tracker
    file_manager = FileManager(cameras=[front_camera, back_camera], save_processed_data=existing_trajectory_path, num_frames=1, clean_folders=True)
    file_manager.update_folder(rgbs=existing_traj.rgbs, masks=None, masks_gripper=None, predictions=None)
    
    duplucate_file_manager = FileManager(cameras=[front_camera, back_camera], save_processed_data=duplicate_traj.dataset_path, num_frames=1, clean_folders=True)
    for t in range(len(existing_traj.rgbs[camera_names[0]])):
        for camera_name in existing_traj.camera_names:
            duplicate_traj.add_data(camera_name=camera_name, rgb=existing_traj.rgbs[camera_name][t], depth=existing_traj.depths[camera_name][t])
        duplucate_file_manager.update_folder(rgbs=duplicate_traj.rgbs, masks=None, masks_gripper=None, predictions=None)
    
    print()
