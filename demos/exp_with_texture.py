import os, sys
import cv2
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import datetime
sys.path.append(".")
from utils.renderer import Renderer
from utils import util
from utils.config import cfg
from facial_alignment.detection import sfd_detector as detector
from facial_alignment.detection import FAN_landmark
from models.face_seg_model import BiSeNet
from models.FLAME import FLAME, FLAMETex

torch.backends.cudnn.benchmark = True


class PhotometricFitting(object):
    def __init__(self, device='cuda'):
        self.config = cfg
        self.device = device
        self.flame = FLAME(self.config).to(self.device)
        self.flametex = FLAMETex(self.config).to(self.device)

        self._setup_renderer()

    def _setup_renderer(self):
        self.render = Renderer(cfg.image_size, obj_filename=cfg.mesh_file).to(self.device)

    def optimize(self, images, landmarks, image_masks, all_param, video_writer, first_flag):
        shape_para, tex_para, exp_para, pose_para, cam_para, lights_para = all_param
        e_opt = torch.optim.Adam(
            [shape_para, exp_para, pose_para, cam_para, tex_para, lights_para],
            lr=cfg.e_lr,
            weight_decay=cfg.e_wd
        )
        d_opt = torch.optim.Adam(
            [shape_para, exp_para, pose_para, cam_para],
            lr=cfg.e_lr,
            weight_decay=cfg.e_wd
        )

        gt_landmark = landmarks
        max_iter = 50
        if first_flag:
            max_iter = cfg.max_iter

        tmp_predict = torch.squeeze(images)
        for k in range(0, max_iter):
            losses = {}
            vertices, landmarks2d, landmarks3d = self.flame(shape_params=shape_para, expression_params=exp_para,
                                                            pose_params=pose_para)
            trans_vertices = util.batch_orth_proj(vertices, cam_para)
            trans_vertices[..., 1:] = - trans_vertices[..., 1:]
            landmarks2d = util.batch_orth_proj(landmarks2d, cam_para)
            landmarks2d[..., 1:] = - landmarks2d[..., 1:]
            landmarks3d = util.batch_orth_proj(landmarks3d, cam_para)
            landmarks3d[..., 1:] = - landmarks3d[..., 1:]

            losses['landmark'] = util.l2_distance(landmarks2d[:, :, :2], gt_landmark[:, :, :2])

            # render
            albedos = self.flametex(tex_para) / 255.
            ops = self.render(vertices, trans_vertices, albedos, lights_para)
            tmp_predict = torchvision.utils.make_grid(ops['images'][0].detach().float().cpu())
            # losses['photometric_texture'] = (image_masks * (ops['images'] - images).abs()).mean() * config.w_pho
            if first_flag:
                losses['photometric_texture'] = F.smooth_l1_loss(image_masks * ops['images'],
                                                                 image_masks * images) * cfg.w_pho

            all_loss = 0.
            for key in losses.keys():
                all_loss = all_loss + losses[key]
            losses['all_loss'] = all_loss
            if first_flag:
                e_opt.zero_grad()
                all_loss.backward()
                e_opt.step()
            else:
                d_opt.zero_grad()
                all_loss.backward()
                d_opt.step()
            loss_info = '----iter: {}, time: {}\n'.format(k, datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
            for key in losses.keys():
                loss_info = loss_info + '{}: {}, '.format(key, float(losses[key]))
            print(loss_info)

        # tmp_predict = torchvision.utils.make_grid(ops['images'][0].detach().float().cpu())
        tmp_predict = (tmp_predict.numpy().transpose(1, 2, 0).copy() * 255)[:, :, [2, 1, 0]]
        tmp_predict = np.minimum(np.maximum(tmp_predict, 0), 255).astype(np.uint8)

        tmp_image = torchvision.utils.make_grid(images[0].detach().float().cpu())
        tmp_image = (tmp_image.numpy().transpose(1, 2, 0).copy() * 255)[:, :, [2, 1, 0]]
        tmp_image = np.minimum(np.maximum(tmp_image, 0), 255).astype(np.uint8)
        combine = np.concatenate((tmp_predict, tmp_image), axis=1)
        cv2.imshow("tmp_image", combine)
        cv2.waitKey(1)
        video_writer.write(combine)
        return [shape_para, tex_para, exp_para, pose_para, cam_para, lights_para]

    def run(self, img, net, rect_detect, landmark_detect, all_param, rect_thresh, out, first_flag):
        # The implementation is potentially able to optimize with images(batch_size>1),
        # here we show the example with a single image fitting
        images = []
        landmarks = []
        image_masks = []
        bbox = rect_detect.extract(img, rect_thresh)
        if len(bbox) > 0:
            crop_image, new_bbox = util.crop_img(img, bbox[0], cfg.cropped_size)

            resize_img, landmark = landmark_detect.extract([crop_image, [new_bbox]])
            landmark = landmark[0]
            landmark[:, 0] = landmark[:, 0] / float(resize_img.shape[1]) * 2 - 1
            landmark[:, 1] = landmark[:, 1] / float(resize_img.shape[0]) * 2 - 1
            landmarks.append(torch.from_numpy(landmark)[None, :, :].double().to(self.device))
            landmarks = torch.cat(landmarks, dim=0)

            image = cv2.resize(crop_image, (cfg.cropped_size, cfg.cropped_size)).astype(np.float32) / 255.
            image = image[:, :, ::-1].transpose(2, 0, 1).copy()
            images.append(torch.from_numpy(image[None, :, :, :]).double().to(self.device))
            images = torch.cat(images, dim=0)
            images = F.interpolate(images, [cfg.image_size, cfg.image_size])

            image_mask = util.face_seg(crop_image, net, cfg.cropped_size)
            image_masks.append(torch.from_numpy(image_mask).double().to(self.device))
            image_masks = torch.cat(image_masks, dim=0)
            image_masks = F.interpolate(image_masks, [cfg.image_size, cfg.image_size])

            single_params = self.optimize(images, landmarks, image_masks, all_param, out, first_flag)
            return single_params


if __name__ == '__main__':
    video_path = str(sys.argv[1])
    device_name = str(sys.argv[2])
    util.check_mkdir(cfg.save_folder)
    fitting = PhotometricFitting(device=device_name)
    save_video_name = os.path.split(video_path)[1].split(".")[0] + '.avi'
    video_writer = cv2.VideoWriter(os.path.join(cfg.save_folder, save_video_name),
                                   cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 16,
                                   (cfg.image_size * 2, cfg.image_size))
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()

    if ret:
        w_h_scale = util.resize_para(frame)
        face_detect = detector.SFDDetector(device_name, cfg.rect_model_path)
        face_landmark = FAN_landmark.FANLandmarks(device_name, cfg.landmark_model_path, cfg.face_detect_type)
        seg_net = BiSeNet(n_classes=cfg.seg_class)
        seg_net.cuda()
        seg_net.load_state_dict(torch.load(cfg.face_seg_model))
        seg_net.eval()
        first_flag = True
        shape = nn.Parameter(torch.zeros(cfg.batch_size, cfg.shape_params).float().to(device_name))
        tex = nn.Parameter(torch.zeros(cfg.batch_size, cfg.tex_params).float().to(device_name))
        exp = nn.Parameter(torch.zeros(cfg.batch_size, cfg.expression_params).float().to(device_name))
        pose = nn.Parameter(torch.zeros(cfg.batch_size, cfg.pose_params).float().to(device_name))
        cam = torch.zeros(cfg.batch_size, cfg.camera_params)
        cam[:, 0] = 5.
        cam = nn.Parameter(cam.float().to(device_name))
        lights = nn.Parameter(torch.zeros(cfg.batch_size, 9, 3).float().to(device_name))
        all_params = [shape, tex, exp, pose, cam, lights]
        while ret:
            all_params = fitting.run(frame, seg_net, face_detect, face_landmark, all_params,
                                     cfg.rect_thresh, video_writer, first_flag)
            first_flag = False
            ret, frame = cap.read()
