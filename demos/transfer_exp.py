import os
import sys
import cv2
import torch
import torch.nn as nn
import numpy as np

sys.path.append('.')
from models.face_seg_model import BiSeNet
from utils import util
from utils.config import cfg
from facial_alignment.detection import sfd_detector as detector
from facial_alignment.detection import FAN_landmark
from demos.exp_with_texture import PhotometricFitting


if __name__ == '__main__':
    video_path = str(sys.argv[1])
    basic_model = str(sys.argv[2])
    device_name = str(sys.argv[3])
    util.check_mkdir(cfg.save_folder)
    fitting = PhotometricFitting(device=device_name)

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
        while ret:
            all_params = fitting.run(frame, seg_net, face_detect, face_landmark, all_params,
                                     cfg.rect_thresh, video_writer, False)
            ret, frame = cap.read()
