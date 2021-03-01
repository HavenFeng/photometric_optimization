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
from models.face_seg_model import BiSeNet
from utils.renderer import Renderer
from utils import util
from face_alignment.detection import sfd_detector as detector
from face_alignment.detection import FAN_landmark

torch.backends.cudnn.benchmark = True


class PhotometricFitting(object):
    def __init__(self, config, device='cuda'):
        self.batch_size = config.batch_size
        self.image_size = config.image_size
        self.cropped_size = config.cropped_size
        self.config = config
        self.device = device
        self.flame = FLAME(self.config).to(self.device)
        self.flametex = FLAMETex(self.config).to(self.device)

        self._setup_renderer()

    def _setup_renderer(self):
        mesh_file = './data/head_template_mesh.obj'
        self.render = Renderer(self.image_size, obj_filename=mesh_file).to(self.device)

    def optimize(self, images, landmarks, image_masks, video_writer):
        bz = images.shape[0]
        shape = nn.Parameter(torch.zeros(bz, self.config.shape_params).float().to(self.device))
        tex = nn.Parameter(torch.zeros(bz, self.config.tex_params).float().to(self.device))
        exp = nn.Parameter(torch.zeros(bz, self.config.expression_params).float().to(self.device))
        pose = nn.Parameter(torch.zeros(bz, self.config.pose_params).float().to(self.device))
        cam = torch.zeros(bz, self.config.camera_params)
        cam[:, 0] = 5.
        cam = nn.Parameter(cam.float().to(self.device))
        lights = nn.Parameter(torch.zeros(bz, 9, 3).float().to(self.device))
        e_opt = torch.optim.Adam(
            [shape, exp, pose, cam, tex, lights],
            lr=self.config.e_lr,
            weight_decay=self.config.e_wd
        )

        gt_landmark = landmarks

        # non-rigid fitting of all the parameters with 68 face landmarks, photometric loss and regularization terms.
        all_train_iter = 0
        all_train_iters = []
        photometric_loss = []
        for k in range(config.max_iter):
            losses = {}
            vertices, landmarks2d, landmarks3d = self.flame(shape_params=shape, expression_params=exp, pose_params=pose)
            trans_vertices = util.batch_orth_proj(vertices, cam)
            trans_vertices[..., 1:] = - trans_vertices[..., 1:]
            landmarks2d = util.batch_orth_proj(landmarks2d, cam)
            landmarks2d[..., 1:] = - landmarks2d[..., 1:]
            landmarks3d = util.batch_orth_proj(landmarks3d, cam)
            landmarks3d[..., 1:] = - landmarks3d[..., 1:]
            losses['landmark'] = util.l2_distance(landmarks2d[:, :, :2], gt_landmark[:, :, :2])

            # render
            albedos = self.flametex(tex) / 255.
            ops = self.render(vertices, trans_vertices, albedos, lights)
            predicted_images = ops['images']
            # losses['photometric_texture'] = (image_masks * (ops['images'] - images).abs()).mean() * config.w_pho
            losses['photometric_texture'] = F.smooth_l1_loss(image_masks * ops['images'],
                                                             image_masks * images) * config.w_pho

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
                all_train_iter += 10
                all_train_iters.append(all_train_iter)
                photometric_loss.append(losses['photometric_texture'])
                print(loss_info)

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
                video_writer.write(grid_image)

        single_params = {
            'shape': shape.detach().cpu().numpy(),
            'exp': exp.detach().cpu().numpy(),
            'pose': pose.detach().cpu().numpy(),
            'cam': cam.detach().cpu().numpy(),
            'verts': trans_vertices.detach().cpu().numpy(),
            'albedos': albedos.detach().cpu().numpy(),
            'tex': tex.detach().cpu().numpy(),
            'lit': lights.detach().cpu().numpy()
        }
        util.draw_train_process("training", all_train_iters, photometric_loss, 'photometric loss')
        # np.save("./test_results/model.npy", single_params)
        return single_params

    def run(self, img, net, rect_detect, landmark_detect, rect_thresh, save_name, video_writer, savefolder):
        # The implementation is potentially able to optimize with images(batch_size>1),
        # here we show the example with a single image fitting
        images = []
        landmarks = []
        image_masks = []
        bbox = rect_detect.extract(img, rect_thresh)
        if len(bbox) > 0:
            crop_image, new_bbox = util.crop_img(img, bbox[0], config.cropped_size)

            # input landmark
            resize_img, landmark = landmark_detect.extract([crop_image, [new_bbox]])
            landmark = landmark[0]
            landmark[:, 0] = landmark[:, 0] / float(resize_img.shape[1]) * 2 - 1
            landmark[:, 1] = landmark[:, 1] / float(resize_img.shape[0]) * 2 - 1
            landmarks.append(torch.from_numpy(landmark)[None, :, :].double().to(self.device))
            landmarks = torch.cat(landmarks, dim=0)

            # input image
            image = cv2.resize(crop_image, (config.cropped_size, config.cropped_size)).astype(np.float32) / 255.
            image = image[:, :, [2, 1, 0]].transpose(2, 0, 1)
            images.append(torch.from_numpy(image[None, :, :, :]).double().to(self.device))
            images = torch.cat(images, dim=0)
            images = F.interpolate(images, [self.image_size, self.image_size])

            # face segment mask
            image_mask = util.face_seg(crop_image, net, self.cropped_size)
            image_masks.append(torch.from_numpy(image_mask).double().to(self.device))
            image_masks = torch.cat(image_masks, dim=0)
            image_masks = F.interpolate(image_masks, [self.image_size, self.image_size])

            # check folder exist or not
            util.check_mkdir(savefolder)
            save_file = os.path.join(savefolder, save_name)

            # optimize
            single_params = self.optimize(images, landmarks, image_masks, video_writer)
            self.render.save_obj(filename=save_file,
                                 vertices=torch.from_numpy(single_params['verts'][0]).to(self.device),
                                 textures=torch.from_numpy(single_params['albedos'][0]).to(self.device)
                                 )
            np.save(save_file, single_params)


if __name__ == '__main__':
    image_path = str(sys.argv[1])
    device_name = str(sys.argv[2])
    config = {
        # FLAME
        'flame_model_path': './model/generic_model.pkl',  # acquire it from FLAME project page
        'flame_lmk_embedding_path': './data/landmark_embedding.npy',
        'face_seg_model': './model/face_seg.pth',
        'seg_class': 19,
        'face_detect_type': "2D",
        'rect_model_path': "./model/s3fd.pth",
        'rect_thresh': 0.5,
        'landmark_model_path': "./model/2DFAN4-11f355bf06.pth.tar",
        'tex_space_path': './model/FLAME_texture.npz',  # acquire it from FLAME project page
        'camera_params': 3,
        'shape_params': 300,
        'expression_params': 100,
        'pose_params': 6,
        'tex_params': 50,
        'use_face_contour': True,
        'max_iter': 2000,
        'cropped_size': 256,
        'batch_size': 1,
        'image_size': 224,
        'e_lr': 0.005,
        'e_wd': 0.0001,
        'savefolder': './test_results/',
        # weights of losses and reg terms
        'w_pho': 8,
        'w_lmks': 1,
        'w_shape_reg': 1e-4,
        'w_expr_reg': 1e-4,
        'w_pose_reg': 0,
    }
    config = util.dict2obj(config)
    save_name = os.path.split(image_path)[1].split(".")[0] + '.obj'
    save_video_name = os.path.split(image_path)[1].split(".")[0] + '.avi'
    video_writer = cv2.VideoWriter(os.path.join(config.savefolder, save_video_name),
                                   cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 16,
                                   (config.image_size, config.image_size * 7))
    util.check_mkdir(config.savefolder)
    fitting = PhotometricFitting(config, device=device_name)
    img = cv2.imread(image_path)
    w_h_scale = util.resize_para(img)

    face_detect = detector.SFDDetector(device_name, config.rect_model_path, w_h_scale)
    face_landmark = FAN_landmark.FANLandmarks(device_name, config.landmark_model_path, config.face_detect_type)

    seg_net = BiSeNet(n_classes=config.seg_class)
    seg_net.cuda()
    seg_net.load_state_dict(torch.load(config.face_seg_model))
    seg_net.eval()

    fitting.run(img, seg_net, face_detect, face_landmark, config.rect_thresh, save_name, video_writer,
                config.savefolder)
