import os, sys
import cv2
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import datetime

sys.path.append('.')
from models.FLAME import FLAME, FLAMETex
from utils.renderer import Renderer
from utils import util
from utils.config import cfg

torch.backends.cudnn.benchmark = True


class PhotometricFitting(object):
    def __init__(self, device='cuda'):
        self.batch_size = cfg.batch_size
        self.image_size = cfg.image_size
        self.cropped_size = cfg.cropped_size
        self.config = cfg
        self.device = device
        self.flame = FLAME(self.config).to(self.device)
        self.flametex = FLAMETex(self.config).to(self.device)

        self._setup_renderer()

    def _setup_renderer(self):
        self.render = Renderer(self.image_size, obj_filename=cfg.mesh_file).to(self.device)

    def optimize(self, images, landmarks, image_masks, savefolder=None):
        bz = images.shape[0]
        shape = nn.Parameter(torch.zeros(bz, cfg.shape_params).float().to(self.device))
        tex = nn.Parameter(torch.zeros(bz, cfg.tex_params).float().to(self.device))
        exp = nn.Parameter(torch.zeros(bz, cfg.expression_params).float().to(self.device))
        pose = nn.Parameter(torch.zeros(bz, cfg.pose_params).float().to(self.device))
        cam = torch.zeros(bz, cfg.camera_params); cam[:, 0] = 5.
        cam = nn.Parameter(cam.float().to(self.device))
        lights = nn.Parameter(torch.zeros(bz, 9, 3).float().to(self.device))
        e_opt = torch.optim.Adam(
            [shape, exp, pose, cam, tex, lights],
            lr=cfg.e_lr,
            weight_decay=cfg.e_wd
        )
        e_opt_rigid = torch.optim.Adam(
            [pose, cam],
            lr=cfg.e_lr,
            weight_decay=cfg.e_wd
        )

        gt_landmark = landmarks

        # rigid fitting of pose and camera with 51 static face landmarks,
        # this is due to the non-differentiable attribute of contour landmarks trajectory
        for k in range(200):
            losses = {}
            vertices, landmarks2d, landmarks3d = self.flame(shape_params=shape, expression_params=exp, pose_params=pose)
            trans_vertices = util.batch_orth_proj(vertices, cam)
            trans_vertices[..., 1:] = - trans_vertices[..., 1:]
            landmarks2d = util.batch_orth_proj(landmarks2d, cam)
            landmarks2d[..., 1:] = - landmarks2d[..., 1:]
            landmarks3d = util.batch_orth_proj(landmarks3d, cam)
            landmarks3d[..., 1:] = - landmarks3d[..., 1:]

            losses['landmark'] = util.l2_distance(landmarks2d[:, 17:, :2], gt_landmark[:, 17:, :2]) * cfg.w_lmks

            all_loss = 0.
            for key in losses.keys():
                all_loss = all_loss + losses[key]
            losses['all_loss'] = all_loss
            e_opt_rigid.zero_grad()
            all_loss.backward()
            e_opt_rigid.step()

            loss_info = '----iter: {}, time: {}\n'.format(k, datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
            for key in losses.keys():
                loss_info = loss_info + '{}: {}, '.format(key, float(losses[key]))
            if k % 10 == 0:
                print(loss_info)

            if k % 10 == 0:
                grids = {}
                visind = range(bz)  # [0]
                grids['images'] = torchvision.utils.make_grid(images[visind]).detach().cpu()
                grids['landmarks_gt'] = torchvision.utils.make_grid(
                    util.tensor_vis_landmarks(images[visind], landmarks[visind]))
                grids['landmarks2d'] = torchvision.utils.make_grid(
                    util.tensor_vis_landmarks(images[visind], landmarks2d[visind]))
                grids['landmarks3d'] = torchvision.utils.make_grid(
                    util.tensor_vis_landmarks(images[visind], landmarks3d[visind]))

                grid = torch.cat(list(grids.values()), 1)
                grid_image = (grid.numpy().transpose(1, 2, 0).copy() * 255)[:, :, [2, 1, 0]]
                grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8)
                cv2.imwrite('{}/{}.jpg'.format(savefolder, k), grid_image)

        # non-rigid fitting of all the parameters with 68 face landmarks, photometric loss and regularization terms.
        for k in range(200, 1000):
            losses = {}
            vertices, landmarks2d, landmarks3d = self.flame(shape_params=shape, expression_params=exp, pose_params=pose)
            trans_vertices = util.batch_orth_proj(vertices, cam)
            trans_vertices[..., 1:] = - trans_vertices[..., 1:]
            landmarks2d = util.batch_orth_proj(landmarks2d, cam)
            landmarks2d[..., 1:] = - landmarks2d[..., 1:]
            landmarks3d = util.batch_orth_proj(landmarks3d, cam)
            landmarks3d[..., 1:] = - landmarks3d[..., 1:]

            losses['landmark'] = util.l2_distance(landmarks2d[:, :, :2], gt_landmark[:, :, :2]) * cfg.w_lmks
            losses['shape_reg'] = (torch.sum(shape ** 2) / 2) * cfg.w_shape_reg  # *1e-4
            losses['expression_reg'] = (torch.sum(exp ** 2) / 2) * cfg.w_expr_reg  # *1e-4
            losses['pose_reg'] = (torch.sum(pose ** 2) / 2) * cfg.w_pose_reg

            ## render
            albedos = self.flametex(tex) / 255.
            ops = self.render(vertices, trans_vertices, albedos, lights)
            predicted_images = ops['images']
            losses['photometric_texture'] = (image_masks * (ops['images'] - images).abs()).mean() * cfg.w_pho

            all_loss = 0.
            for key in losses.keys():
                all_loss = all_loss + losses[key]
            losses['all_loss'] = all_loss
            e_opt.zero_grad()
            all_loss.backward()
            e_opt.step()

            loss_info = '----iter: {}, time: {}\n'.format(k, datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
            for key in losses.keys():
                loss_info = loss_info + '{}: {}, '.format(key, float(losses[key]))

            if k % 10 == 0:
                print(loss_info)

            # visualize
            if k % 10 == 0:
                grids = {}
                visind = range(bz)  # [0]
                grids['images'] = torchvision.utils.make_grid(images[visind]).detach().cpu()
                grids['landmarks_gt'] = torchvision.utils.make_grid(
                    util.tensor_vis_landmarks(images[visind], landmarks[visind]))
                grids['landmarks2d'] = torchvision.utils.make_grid(
                    util.tensor_vis_landmarks(images[visind], landmarks2d[visind]))
                grids['landmarks3d'] = torchvision.utils.make_grid(
                    util.tensor_vis_landmarks(images[visind], landmarks3d[visind]))
                grids['albedoimage'] = torchvision.utils.make_grid(
                    (ops['albedo_images'])[visind].detach().cpu())
                grids['render'] = torchvision.utils.make_grid(predicted_images[visind].detach().float().cpu())
                shape_images = self.render.render_shape(vertices, trans_vertices, images)
                grids['shape'] = torchvision.utils.make_grid(
                    F.interpolate(shape_images[visind], [224, 224])).detach().float().cpu()


                # grids['tex'] = torchvision.utils.make_grid(F.interpolate(albedos[visind], [224, 224])).detach().cpu()
                grid = torch.cat(list(grids.values()), 1)
                grid_image = (grid.numpy().transpose(1, 2, 0).copy() * 255)[:, :, [2, 1, 0]]
                grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8)

                cv2.imwrite('{}/{}.jpg'.format(savefolder, k), grid_image)

        single_params = {
            'shape': shape.detach().cpu().numpy(),
            'exp': exp.detach().cpu().numpy(),
            'pose': pose.detach().cpu().numpy(),
            'cam': cam.detach().cpu().numpy(),
            'verts': trans_vertices.detach().cpu().numpy(),
            'albedos':albedos.detach().cpu().numpy(),
            'tex': tex.detach().cpu().numpy(),
            'lit': lights.detach().cpu().numpy()
        }
        return single_params

    def run(self, imagepath, landmarkpath):
        # The implementation is potentially able to optimize with images(batch_size>1),
        # here we show the example with a single image fitting
        images = []
        landmarks = []
        image_masks = []

        image_name = os.path.basename(imagepath)[:-4]
        savefile = os.path.sep.join([cfg.save_folder, image_name + '.npy'])

        # photometric optimization is sensitive to the hair or glass occlusions,
        # therefore we use a face segmentation network to mask the skin region out.
        image_mask_folder = './FFHQ_seg/'
        image_mask_path = os.path.sep.join([image_mask_folder, image_name + '.npy'])

        image = cv2.resize(cv2.imread(imagepath), (cfg.cropped_size, cfg.cropped_size)).astype(np.float32) / 255.
        image = image[:, :, [2, 1, 0]].transpose(2, 0, 1)
        images.append(torch.from_numpy(image[None, :, :, :]).to(self.device))

        image_mask = np.load(image_mask_path, allow_pickle=True)
        image_mask = image_mask[..., None].astype('float32')
        image_mask = image_mask.transpose(2, 0, 1)
        image_mask_bn = np.zeros_like(image_mask)
        image_mask_bn[np.where(image_mask != 0)] = 1.
        image_masks.append(torch.from_numpy(image_mask_bn[None, :, :, :]).to(self.device))

        landmark = np.load(landmarkpath).astype(np.float32)
        landmark[:, 0] = landmark[:, 0] / float(image.shape[2]) * 2 - 1
        landmark[:, 1] = landmark[:, 1] / float(image.shape[1]) * 2 - 1
        landmarks.append(torch.from_numpy(landmark)[None, :, :].float().to(self.device))

        images = torch.cat(images, dim=0)
        images = F.interpolate(images, [cfg.image_size, cfg.image_size])
        image_masks = torch.cat(image_masks, dim=0)
        image_masks = F.interpolate(image_masks, [cfg.image_size, cfg.image_size])

        landmarks = torch.cat(landmarks, dim=0)
        savefolder = os.path.sep.join([cfg.save_folder, image_name])

        util.check_mkdir(savefolder)
        # optimize
        single_params = self.optimize(images, landmarks, image_masks, savefolder)
        self.render.save_obj(filename=savefile[:-4]+'.obj',
                             vertices=torch.from_numpy(single_params['verts'][0]).to(self.device),
                             textures=torch.from_numpy(single_params['albedos'][0]).to(self.device)
                             )
        np.save(savefile, single_params)


if __name__ == '__main__':
    image_name = str(sys.argv[1])
    device_name = str(sys.argv[2])
    util.check_mkdir(cfg.save_folder)

    cfg.batch_size = 1
    fitting = PhotometricFitting(device=device_name)

    input_folder = './FFHQ'

    imagepath = os.path.sep.join([input_folder, image_name + '.png'])
    landmarkpath = os.path.sep.join([input_folder, image_name + '.npy'])
    fitting.run(imagepath, landmarkpath)
