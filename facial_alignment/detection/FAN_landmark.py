from facial_alignment.detection.models import FAN, ResNetDepth
from .utils import crop, get_preds_fromhm, draw_gaussian
import torch
import numpy as np
import cv2


class FANLandmarks:
    def __init__(self, device, model_path, detect_type):
        # Initialise the face detector
        model_weights = torch.load(model_path)
        self.device = device
        self.detect_type = detect_type
        torch.backends.cudnn.benchmark = True
        self.face_landmark = FAN(4)
        self.face_landmark.load_state_dict(model_weights)
        self.face_landmark.to(device)
        self.face_landmark.eval()
        self.reference_scale = 195.0

        if self.detect_type == "3D":
            self.depth_prediciton_net = ResNetDepth()
            depth_weights = torch.load("D:/model/depth-2a464da4ea.pth.tar")
            depth_dict = {k.replace('module.', ''): v for k, v in depth_weights['state_dict'].items()}
            self.depth_prediciton_net.load_state_dict(depth_dict)
            self.depth_prediciton_net.to(device)
            self.depth_prediciton_net.eval()

    def extract(self, rect_queue):
        # image, face_rect = rect_queue.get(block=True, timeout=10)
        image, face_rect = rect_queue
        landmarks = []
        for i, d in enumerate(face_rect):
            center_x = d[2] - (d[2] - d[0]) / 2.0
            center_y = d[3] - (d[3] - d[1]) / 2.0
            center = torch.FloatTensor([center_x, center_y])
            scale = (d[2] - d[0] + d[3] - d[1]) / self.reference_scale

            inp = crop(image, center, scale)
            inp = torch.from_numpy(inp.transpose((2, 0, 1))).float().to(self.device)
            inp.div_(255.0).unsqueeze_(0)
            with torch.no_grad():
                out = self.face_landmark(inp)[-1]
            out = out.cpu()
            pts, pts_img = get_preds_fromhm(out, center, scale)
            pts, pts_img = pts.view(68, 2) * 4, pts_img.view(68, 2)

            if self.detect_type == "3D":
                heatmaps = np.zeros((68, 256, 256), dtype=np.float32)
                for i in range(68):
                    if pts[i, 0] > 0:
                        heatmaps[i] = draw_gaussian(heatmaps[i], pts[i], 2)
                heatmaps = torch.from_numpy(heatmaps).unsqueeze_(0)

                heatmaps = heatmaps.to(self.device)
                depth_pred = self.depth_prediciton_net(torch.cat((inp, heatmaps), 1)).data.cpu().view(68, 1)
                pts_img = torch.cat((pts_img, depth_pred * (1.0 / (256.0 / (200.0 * scale)))), 1)

            landmarks.append(pts_img.numpy())

        return image, landmarks


