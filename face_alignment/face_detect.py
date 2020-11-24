from face_alignment.detection import sfd_detector as detector
from face_alignment.detection import FAN_landmark
import cv2
import torch
import socket
from TG_thread.FullProcess import FullSwapProcess
from collections import deque
import time
from head_pose_estimation.pose_estimator import PoseEstimator
from head_pose_estimation.stabilizer import Stabilizer
from head_pose_estimation.visualization import *
from head_pose_estimation.misc import *
import multiprocessing as mp
from DFLIMG import *
from pathlib import Path


def resize_para(ori_frame):
    w, h, c = ori_frame.shape
    d = max(w, h)
    scale_to = 640 if d >= 1280 else d / 2
    scale_to = max(64, scale_to)
    input_scale = d / scale_to
    w = int(w / input_scale)
    h = int(h / input_scale)
    image_info = [w, h, input_scale]
    return image_info


def cv2_imwrite(filename, img, *args):
    ret, buf = cv2.imencode(Path(filename).suffix, img, *args)
    if ret == True:
        try:
            with open(filename, "wb") as stream:
                stream.write(buf)
        except:
            pass


video_path = "E:/data/face_video/20200810hsc.mp4"
rect_model_path = "D:/model/s3fd.pth"
landmark_model_path = "D:/model/2DFAN4-11f355bf06.pth.tar"
if __name__ == "__main__":
    # mp.set_start_method("spawn")
    # ctx = mp.get_context("spawn")
    # FullSwapProcess(ctx, video_path, 'cuda', 0.5).start()
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    # frame = cv2.imread("D:/data/face.png")
    # frame = cv2.resize(frame, (640, 360))

    w_h_scale = resize_para(frame)
    device = torch.device("cuda")
    face_detect = detector.SFDDetector(device, rect_model_path, w_h_scale)
    face_landmark = FAN_landmark.FANLandmarks(device, landmark_model_path)
    thresh = 0.5

    # pose_estimator = PoseEstimator(img_size=frame.shape[:2])
    # pose_stabilizers = [Stabilizer(
    #     state_num=2,
    #     measure_num=1,
    #     cov_process=0.01,
    #     cov_measure=0.1) for _ in range(8)]
    # address = ('127.0.0.1', 5066)
    # s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # s.connect(address)
    frame_count = 0
    ts = []
    no_face_count = 0
    prev_boxes = deque(maxlen=5)
    prev_marks = deque(maxlen=5)

    while True:
        # frame_count += 1
        # flip_frame = cv2.flip(frame, 2)
        # t = time.time()
        # facebox = face_detect.extract(flip_frame, thresh)
        #
        # box = facebox[0]
        #
        # if len(box) > 0:
        #     # or every even frame
        #     box = [int(i) for i in box]
        #     face_img = flip_frame[box[1]: box[3], box[0]: box[2]]
        #     transfer_frame, landmarks = face_landmark.extract([flip_frame, facebox])
        #     marks = landmarks[-1]
        #
        #     # x_l, y_l, ll, lu = detect_iris(frame, marks, "left")
        #     # x_r, y_r, rl, ru = detect_iris(frame, marks, "right")
        #     pose = pose_estimator.solve_pose_by_68_points(marks)
        #
        #     # pose_estimator.draw_annotation_box(
        #     #     frame, pose[0], pose[1], color=(128, 255, 128))
        #
        #     steady_pose = []
        #     pose_np = np.array(pose).flatten()
        #     for value, ps_stb in zip(pose_np, pose_stabilizers):
        #         ps_stb.update([value])
        #         steady_pose.append(ps_stb.state[0])
        #     #
        #     # roll = np.clip(-(180 + np.degrees(steady_pose[2])), -50, 50)[0]
        #     # pitch = np.clip(-(np.degrees(steady_pose[1])) - 15, -40, 40)[0]
        #     # yaw = np.clip(-(np.degrees(steady_pose[0])), -50, 50)[0]
        #     roll = float(np.degrees(steady_pose[2]))
        #     pitch = float(np.degrees(steady_pose[1]))
        #     yaw = float(np.degrees(steady_pose[0]))
        #
        #     mouse_open_ratio = np.linalg.norm(marks[62] - marks[66]) / np.linalg.norm(marks[60] - marks[64]) * 2
        #     print(mouse_open_ratio)
        #     # if frame_count > 60:  # send information to unity
        #     #     msg = '%.4f %.4f %.4f %.4f' % \
        #     #           (roll, pitch, yaw, mouse_open_ratio)
        #     #     s.send(bytes(msg, "utf-8"))
        #     cv2.imshow("Preview", frame)
        #     if cv2.waitKey(1) == 27:
        #         break
        #     ret, frame = cap.read()
        #     frame = cv2.resize(frame, (640, 360))

        total_start = cv2.getTickCount()
        rect_start = cv2.getTickCount()
        bbox = face_detect.extract(frame, thresh)
        rect_end = cv2.getTickCount()
        rect_time = (rect_end - rect_start) / cv2.getTickFrequency() * 1000
        print("rect time: ", rect_time, "ms...")
        if len(bbox) > 0:
            frame, landmarks = face_landmark.extract([frame, bbox])
            total_end = cv2.getTickCount()
            total_time = (total_end - total_start) / cv2.getTickFrequency() * 1000
            print("total time: ", total_time, "ms...")
            for land in landmarks:
                # max_x, max_y = land.astype("int32").max(axis=0)
                # min_x, min_y = land.astype("int32").min(axis=0)
                # center_x = (max_x + min_x) // 2
                # center_y = (max_y + min_y) // 2
                # w = int((max_x - min_x) * 1.6)
                # h = int((max_y - min_y) * 2.2)
                # x1 = center_x - h // 2
                # x2 = x1 + h
                # y1 = center_y - h // 3 * 2
                # y2 = y1 + h
                # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255))
                for (x, y) in land:
                    cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
        img_h, img_w, _ = frame.shape
        ratio = 1.6
        if img_w == 1280:
            ratio = 1.4
        elif img_w == 3840:
            ratio = 1.8
        if len(bbox) > 0:
            for box in bbox:
                box = [int(i) for i in box]
                x, y, x2, y2 = box
                center_x = (x + x2) // 2
                center_y = (y + y2) // 2
                h = int((y2 - y) * ratio)
                if center_x < h / 2:
                    h = center_x
                if center_y < h / 2:
                    h = center_y
                if center_x + h / 2 > img_w:
                    h = (img_w - center_x) * 2
                if center_y + h / 2 > img_h:
                    h = (img_h - center_y) * 2

                x1 = max(center_x - h // 2, 0)
                x2 = min(x1 + h, img_w)
                y1 = max(center_y - h // 3 * 2, 0)
                y2 = min(y1 + h, img_h)
                cut_frame = frame[y1:y2, x1:x2, :]
                cut_frame = cv2.resize(cut_frame, (768, 768))
                cv2.imshow("cut_frame", cut_frame)
                cv2.waitKey(1)
                frame_count += 1
                output_filepath = "E:/data/hsc/" + str(frame_count).zfill(5) + ".jpg"
                # cv2_imwrite(output_filepath, cut_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                # cv2.imwrite(output_filepath, cut_frame)
                import os
                if os.path.isfile(output_filepath):
                    dflimg = DFLJPG.load(output_filepath)
                    dflimg.set_source_rect((x1, y1, x2, y2))

                    dflimg.save()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255))
        cv2.imshow("frame", frame)
        cv2.waitKey(1)
        ret, frame = cap.read()
        # frame = cv2.resize(frame, (640, 360))
