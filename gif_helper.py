import os, sys
import cv2
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from pytorch3d.io import load_obj

from glob import glob
import time
import datetime
import imageio

sys.path.append('./models/')
from FLAME import FLAME, FLAMETex
from renderer import Renderer
import util
torch.backends.cudnn.benchmark = True


class render_utils(object):
    def __init__(self, config, device='cuda'):
        self.batch_size = config.batch_size
        self.image_size = config.image_size
        self.config = config
        self.device = device
        #
        self.flame = FLAME(self.config).to(self.device)
        self.flametex = FLAMETex(self.config).to(self.device)

        self._setup_renderer()
        self.flame_faces = self.render.faces

    def _setup_renderer(self):
        mesh_file = './data/head_template_mesh.obj'
        self.render = Renderer(self.image_size, obj_filename=mesh_file).to(self.device)

    def render_tex_and_normal(self, shapecode, expcode, posecode, texcode, lightcode, cam):
        verts, _, _ = self.flame(shape_params=shapecode, expression_params=expcode, pose_params=posecode)
        trans_verts = util.batch_orth_proj(verts, cam)
        trans_verts[:, :, 1:] = -trans_verts[:, :, 1:]

        albedos = self.flametex(texcode)
        rendering_results = self.render(verts, trans_verts, albedos, lights=lightcode)
        textured_images, normals = rendering_results['images'], rendering_results['normals']
        normal_images = self.render.render_normal(trans_verts, normals)
        return textured_images, normal_images

    def get_flame_faces(self):
        return self.flame_faces


if __name__ == '__main__':
    config = {
        # FLAME
        'flame_model_path': './data/model.pkl',  # acquire it from FLAME project page
        'flame_lmk_embedding_path': './data/landmark_embedding.npy',
        'tex_space_path': './data/FLAME_texture.npz',  # acquire it from FLAME project page
        'camera_params': 3,
        'shape_params': 100,
        'expression_params': 50,
        'pose_params': 6,
        'tex_params': 50,
        'use_face_contour': True,

        'cropped_size': 256,
        'batch_size': 1,
        'image_size': 224,
    }
    config = util.dict2obj(config)
    gif_helper = render_utils(config)
