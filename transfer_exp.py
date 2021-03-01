import os
import sys
import cv2
import torch
import torch.nn as nn
import numpy as np
from face_seg_model import BiSeNet
import util
from face_alignment.detection import sfd_detector as detector
from face_alignment.detection import FAN_landmark
from exp_with_texture import PhotometricFitting


if __name__ == '__main__':
    video_path = str(sys.argv[1])
    basic_model = str(sys.argv[2])
    device_name = str(sys.argv[3])
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
        'max_iter': 400,
        'cropped_size': 256,
        'batch_size': 1,
        'image_size': 224,
        'e_lr': 0.005,
        'e_wd': 0.0001,
        'savefolder': 'E:/data/test_results/',
        # weights of losses and reg terms
        'w_pho': 64,
        'w_lmks': 1,
        'w_shape_reg': 1e-4,
        'w_expr_reg': 1e-4,
        'w_pose_reg': 0,
    }
    config = util.dict2obj(config)
    util.check_mkdir(config.savefolder)
    fitting = PhotometricFitting(config, device=device_name)

    # basic face parameter npy file
    basic_face_data = np.load(basic_model, allow_pickle=True).item()
    shape = nn.Parameter(torch.from_numpy(basic_face_data['shape']).float().to(device_name))
    tex = nn.Parameter(torch.from_numpy(basic_face_data['tex']).float().to(device_name))
    exp = nn.Parameter(torch.from_numpy(basic_face_data['exp']).float().to(device_name))
    pose = nn.Parameter(torch.from_numpy(basic_face_data['pose']).float().to(device_name))
    cam = nn.Parameter(torch.from_numpy(basic_face_data['cam']).float().to(device_name))
    lights = nn.Parameter(torch.from_numpy(basic_face_data['lit']).float().to(device_name))
    all_params = [shape, tex, exp, pose, cam, lights]

    save_video_name = os.path.split(video_path)[1].split(".")[0] + '.avi'
    video_writer = cv2.VideoWriter(os.path.join(config.savefolder, save_video_name),
                                   cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 16,
                                   (config.image_size * 2, config.image_size))

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if ret:
        w_h_scale = util.resize_para(frame)
        face_detect = detector.SFDDetector(device_name, config.rect_model_path, w_h_scale)
        face_landmark = FAN_landmark.FANLandmarks(device_name, config.landmark_model_path, config.face_detect_type)
        seg_net = BiSeNet(n_classes=config.seg_class)
        seg_net.cuda()
        seg_net.load_state_dict(torch.load(config.face_seg_model))
        seg_net.eval()
        while ret:
            all_params = fitting.run(frame, seg_net, face_detect, face_landmark, all_params,
                                     config.rect_thresh, video_writer, False)
            ret, frame = cap.read()
