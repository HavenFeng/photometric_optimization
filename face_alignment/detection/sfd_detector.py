from .net_s3fd import s3fd
from .bbox import nms, decode
import torch.nn.functional as F
import numpy as np
import cv2
import torch


class SFDDetector:
    def __init__(self, device_name, model_path, image_info):
        # Initialise the face detector
        device = torch.device(device_name)
        model_weights = torch.load(model_path)
        torch.backends.cudnn.benchmark = True

        self.w, self.h, self.input_scale = image_info
        self.device = device
        self.face_detector = s3fd().to(self.device)
        self.face_detector.load_state_dict(model_weights)
        self.face_detector.eval()

    def pre_process_frame(self, frame):
        img = cv2.resize(frame, (self.h, self.w))
        img = img[..., ::-1]
        img = img - np.array([104, 117, 123])
        img = img.transpose(2, 0, 1)
        img = img.reshape((1,) + img.shape)
        return img

    def detect_rect(self, frame, thresh):
        img = self.pre_process_frame(frame)
        img = torch.from_numpy(img).float().to(self.device)
        with torch.no_grad():
            olist = self.face_detector(img)

        bboxes = []
        olist = [oelem.data.cpu() for oelem in olist]
        for i in range(len(olist) // 2):
            ocls, oreg = olist[i * 2], olist[i * 2 + 1]
            stride = 2 ** (i + 2)  # 4,8,16,32,64,128
            poss = zip(*np.where(ocls[:, 1, :, :] > 0.05))
            for Iindex, hindex, windex in poss:
                axc, ayc = stride / 2 + windex * stride, stride / 2 + hindex * stride
                score = ocls[0, 1, hindex, windex]
                if score > thresh:
                    loc = oreg[0, :, hindex, windex].contiguous().view(1, 4)
                    priors = torch.Tensor([[axc / 1.0, ayc / 1.0, stride * 4 / 1.0, stride * 4 / 1.0]])
                    variances = [0.1, 0.2]
                    box = decode(loc, priors, variances)
                    x1, y1, x2, y2 = box[0] * 1.0
                    bboxes.append([x1, y1, x2, y2, score])
        bboxes = np.array(bboxes)
        return bboxes

    def extract(self, frame, thresh):

        bboxes = self.detect_rect(frame, thresh)
        if len(bboxes) > 0:
            keep = nms(bboxes, 0.3)
            bboxlist = bboxes[keep, :]
            # restore the rect points
            detected_faces = []
            for ltrb in bboxlist:
                l, t, r, b, _ = [x * self.input_scale for x in ltrb]
                bt = b - t
                if min(r - l, bt) < 40:
                    continue
                b += bt * 0.1
                detected_faces.append((l, t, r, b))
        else:
            return []
        # for box in detected_faces:
        #     box = [int(i) for i in box]
        #     x, y, x2, y2 = box
        #     cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 255))
        return detected_faces
