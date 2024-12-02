
import torch
import os
import cv2
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
from groundingdino.util.inference import Model as GroundingDINOModel
from segment_anything import sam_model_registry, SamPredictor, build_sam
import groundingdino
from groundingdino.util.inference import Model as GroundingDINOModel
from segment_anything import build_sam, SamPredictor
import sys

# append to the system path the path to the installation of XMEM.
# folder_path = "/home/omniverse/workspace/d3fields"
# sys.path.append(folder_path)

from XMem.model.network import XMem
from XMem.inference.data.mask_mapper import MaskMapper
from XMem.inference.inference_core import InferenceCore
from XMem.dataset.range_transform import im_normalization

class Fusion():
    def __init__(self, num_cam, feat_backbone='dinov2', device='cuda:0', dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        
        # hyper-parameters
        self.mu = 0.02
        
        # curr_obs_torch is a dict contains:
        # - 'dino_feats': (K, patch_h, patch_w, feat_dim) torch tensor, dino features
        # - 'depth': (K, H, W) torch tensor, depth images
        # - 'pose': (K, 4, 4) torch tensor, poses of the images
        # - 'K': (K, 3, 3) torch tensor, intrinsics of the images
        self.curr_obs_torch = {}
        self.H = -1
        self.W = -1
        self.num_cam = num_cam
        
        # dino feature extractor
        # self.feat_backbone = feat_backbone
        # if self.feat_backbone == 'dinov2':
        #     self.dinov2_feat_extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').to(self.device)
        # else:
        #     raise NotImplementedError
        # self.dinov2_feat_extractor.eval()
        # self.dinov2_feat_extractor.to(dtype=self.dtype)
        
        # load GroundedSAM model
        curr_path = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(groundingdino.__path__[0], 'config/GroundingDINO_SwinT_OGC.py')
        grounded_checkpoint = os.path.join(curr_path, 'ckpts/groundingdino_swint_ogc.pth')
        # config_file = os.path.join(curr_path, '../gdino_config/GroundingDINO_SwinB.cfg.py')
        # grounded_checkpoint = os.path.join(curr_path, 'ckpts/groundingdino_swinb_cogcoor.pth')
        if not os.path.exists(grounded_checkpoint):
            print('Downloading GroundedSAM model...')
            ckpts_dir = os.path.join(curr_path, 'ckpts')
            os.system(f'mkdir -p {ckpts_dir}')
            # os.system('wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth')
            # os.system(f'mv groundingdino_swinb_cogcoor.pth {ckpts_dir}')
            os.system('wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth')
            os.system(f'mv groundingdino_swint_ogc.pth {ckpts_dir}')
        sam_checkpoint = os.path.join(curr_path, 'ckpts/sam_vit_h_4b8939.pth')
        if not os.path.exists(sam_checkpoint):
            print('Downloading SAM model...')
            ckpts_dir = os.path.join(curr_path, 'ckpts')
            os.system(f'mkdir -p {ckpts_dir}')
            os.system('wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth')
            os.system(f'mv sam_vit_h_4b8939.pth {ckpts_dir}')
        self.ground_dino_model = GroundingDINOModel(config_file, grounded_checkpoint, device=self.device)

        self.sam_model = SamPredictor(build_sam(checkpoint=sam_checkpoint))
        self.sam_model.model = self.sam_model.model.to(self.device)
        
        # load XMem model
        XMem_path = os.path.join(curr_path, 'XMem/saves/XMem.pth')
        if not os.path.exists(XMem_path):
            print('Downloading XMem model...')
            ckpts_dir = os.path.join(curr_path, 'XMem/saves')
            os.system(f'mkdir -p {ckpts_dir}')
            os.system(f'wget https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem.pth')
            os.system(f'mv XMem.pth {ckpts_dir}')
            
        xmem_config = {
            'model': XMem_path,
            'disable_long_term': False,
            'enable_long_term': True,
            'max_mid_term_frames': 10,
            'min_mid_term_frames': 5,
            'max_long_term_elements': 10000,
            'num_prototypes': 128,
            'top_k': 30,
            'mem_every': 5,
            'deep_update_every': -1,
            'save_scores': False,
            'size': 480,
            'key_dim': 64,
            'value_dim': 512,
            'hidden_dim': 64,
            'enable_long_term_count_usage': True,
        }
        
        network = XMem(xmem_config, xmem_config['model']).to(self.device).eval()
        model_weights = torch.load(xmem_config['model'])
        network.load_weights(model_weights, init_as_zero_if_needed=True)
        self.xmem_mapper = MaskMapper()
        self.xmem_processors = [InferenceCore(network, config=xmem_config) for _ in range(self.num_cam)]
        if xmem_config['size'] < 0:
            self.xmem_im_transform = T.Compose([
                T.ToTensor(),
                im_normalization,
            ])
            self.xmem_mask_transform = None
        else:
            self.xmem_im_transform = T.Compose([
                T.ToTensor(),
                im_normalization,
                T.Resize(xmem_config['size'], interpolation=T.InterpolationMode.BILINEAR),
            ])
            self.xmem_mask_transform = T.Compose([
                T.Resize(xmem_config['size'], interpolation=T.InterpolationMode.NEAREST),
            ])
        self.xmem_first_mask_loaded = False
        self.track_ids = [0]
        
    def xmem_process(self, rgb, mask):
        # track the mask using XMem
        # :param: rgb: (K, H, W, 3) np array, color image
        # :param: mask: None or (K, H, W) torch tensor, mask
        # return: out_masks: (K, H, W) torch tensor, mask
        # rgb_tensor = torch.zeros((self.num_cam, 3, self.H, self.W), device=self.device, dtype=torch.float32)
        with torch.no_grad():
            rgb_tensor = []
            for i in range(self.num_cam):
                rgb_tensor.append(self.xmem_im_transform(rgb[i]).to(self.device, dtype=torch.float32))
            rgb_tensor = torch.stack(rgb_tensor, dim=0)
            if self.xmem_mask_transform is not None and mask is not None:
                mask = self.xmem_mask_transform(mask).to(self.device, dtype=torch.float32)
            
            if mask is not None and not self.xmem_first_mask_loaded:
                # converted_masks = []
                for i in range(self.num_cam):
                    _, labels = self.xmem_mapper.convert_mask(mask[i].cpu().numpy(), exhaustive=True)
                    # converted_masks.append(converted_mask)
                converted_masks = [self.xmem_mapper.convert_mask(mask[i].cpu().numpy(), exhaustive=True)[0] for i in range(self.num_cam)]
                # # assume that labels for all views are the same
                # for labels in labels_list:
                #     assert labels == labels_list[0]
                converted_masks = torch.from_numpy(np.stack(converted_masks, axis=0)).to(self.device, dtype=torch.float32)
                for processor in self.xmem_processors:
                    processor.set_all_labels(list(self.xmem_mapper.remappings.values()))
                self.track_ids = [0,] + list(self.xmem_mapper.remappings.values())
            elif mask is not None and self.xmem_first_mask_loaded:
                converted_masks = instance2onehot(mask.to(torch.uint8), len(self.track_ids))
                converted_masks = converted_masks.permute(0, 3, 1, 2).to(self.device, dtype=torch.float32)
                converted_masks = converted_masks[:, 1:] # remove the background
            
            if not self.xmem_first_mask_loaded:
                if mask is not None:
                    self.xmem_first_mask_loaded = True
                else:
                    # no point to do anything without a mask
                    raise ValueError('No mask provided for the first frame')
            
            out_masks = torch.zeros((self.num_cam, self.H, self.W)).to(self.device, dtype=torch.uint8)
            for view_i, processor in enumerate(self.xmem_processors):
                prob = processor.step(rgb_tensor[view_i],
                                      converted_masks[view_i] if mask is not None else None,
                                      self.track_ids[1:] if mask is not None else None,
                                      end=False)
                prob = F.interpolate(prob.unsqueeze(1), (self.H, self.W), mode='bilinear', align_corners=False)[:,0]
                
                out_mask = torch.argmax(prob, dim=0)
                out_mask = (out_mask.detach().cpu().numpy()).astype(np.uint8)
                out_mask = self.xmem_mapper.remap_index_mask(out_mask)
                # out_mask = instance2onehot(out_mask)
                out_masks[view_i] = torch.from_numpy(out_mask).to(self.device, dtype=torch.uint8)
            out_masks = instance2onehot(out_masks, len(self.track_ids))
        return out_masks.to(self.device, dtype=self.dtype)
        


def grounded_instance_sam_new_ver(image,
                                  text_prompts,
                                  dino_model : GroundingDINOModel,
                                  sam_model : SamPredictor,
                                  box_thresholds,
                                  merge_all=False,
                                  device="cuda"):
    # :param image: [H, W, 3] BGR
    assert len(image.shape) == 3
    # cfg
    text_threshold = 0.25
    device = "cuda:0"

    # detect objects
    detections = dino_model.predict_with_classes(
        image=image,
        # classes=enhance_class_name(class_names=text_prompts),
        classes=text_prompts,
        box_threshold=box_thresholds[0],
        text_threshold=text_threshold,
    )
    
    # convert detections to masks
    detections.mask = segment(
        sam_predictor=sam_model,
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        xyxy=detections.xyxy
    )
    labels = ['background']
    for query_i in detections.class_id.tolist():
        labels.append(text_prompts[query_i])
    
    # add detections mask for background
    bg_mask = ~np.bitwise_or.reduce(detections.mask, axis=0)
    bg_conf = 1.0
    detections.mask = np.concatenate([np.expand_dims(bg_mask, axis=0), detections.mask], axis=0)
    detections.confidence = np.concatenate([np.array([bg_conf]), detections.confidence], axis=0)

    return detections.mask, labels, detections.confidence

def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)



def instance2onehot(instance, N = None):
    # :param instance: [**dim] numpy array uint8, val from 0 to N-1
    # :return: [**dim, N] numpy array bool
    if N is None:
        N = instance.max() + 1
    if type(instance) is np.ndarray:
        assert instance.dtype == np.uint8
        out = np.zeros(instance.shape + (N,), dtype=bool)
        for i in range(N):
            out[..., i] = (instance == i)
    elif type(instance) is torch.Tensor:
        assert instance.dtype == torch.uint8
        # assert instance.min() == 0
        out = torch.zeros(instance.shape + (N,), dtype=torch.bool, device=instance.device)
        for i in range(N):
            out[..., i] = (instance == i)
    return out



if __name__=="__main__":
    fusion = Fusion(num_cam=4, feat_backbone='dinov2', device='cuda:0', dtype=torch.float32)