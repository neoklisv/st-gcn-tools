import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk

# Copyright (c) OpenMMLab. All rights reserved.
"""
conda activate pyskl
python demo/stgcn_yolov8.py --config configs/stgcn++/stgcn++_ntu120_xsub_hrnet/j.py     --checkpoint http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu120_xsub_hrnet/j.pth
"""
import argparse
import cv2
import os

import mmcv
import numpy as np
import os
import os.path as osp
import shutil
import torch
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
from ultralytics.utils.plotting import Colors
import json
import glob
import queue, threading, time
from playsound import playsound

import mmengine
import torch
from mmengine import DictAction
from mmengine.utils import track_iter_progress

from mmaction.apis import (detection_inference, inference_recognizer,
                           init_recognizer, pose_inference)
from mmaction.registry import VISUALIZERS
from mmaction.utils import frame_extract
from mmengine.utils import ProgressBar

FRAME_TO_PROCESS=50
CAM_INDEX=0

keypoint_array       = []
keypoint_score_array = []
prev_time=0

# bufferless VideoCapture
class VideoCapture:

  def __init__(self, name):
    self.cap = cv2.VideoCapture(name)
    self.cap.set(3, 640)
    self.cap.set(4, 480)
    self.q = queue.Queue()
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()

  # read frames as soon as they are available, keeping only most recent one
  def _reader(self):
    while True:
      ret, frame = self.cap.read()
      if not ret:
        break
      if not self.q.empty():
        try:
          self.q.get_nowait()   # discard previous (unprocessed) frame
        except Queue.Empty:
          pass
      self.q.put(frame)

  def read(self):
    return self.q.get()


def parse_args():
    parser = argparse.ArgumentParser(description='PoseC3D demo')
    parser.add_argument(
        '--config',
        default='configs/stgcn++/stgcn++_ntu120_xsub_hrnet/j.py',
        help='skeleton action recognition config file path')
    parser.add_argument(
        '--checkpoint',
        default=
        'http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu120_xsub_hrnet/j.pths',
        help='skeleton action recognition checkpoint file/url')
    parser.add_argument(
        '--pose-checkpoint',
        default=
        'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth',
        help='human pose estimation checkpoint file/url')
    parser.add_argument('--det-score-thr',
                        type=float,
                        default=0.7,
                        help='the threshold of human detection score')
    parser.add_argument('--label-map',
                        default='tools/data/skeleton/label_map_ntu60.txt',
                        help='label map file')
    parser.add_argument('--device',
                        type=str,
                        default='cuda:0',
                        help='CPU/CUDA device option')
    parser.add_argument(
    '--cfg-options',
    nargs='+',
    action=DictAction,
    default={},
    help='override some settings in the used config, the key-value pair '
    'in xxx=yyy format will be merged into config file. For example, '
    "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    args = parser.parse_args()
    return args

args = parse_args()
yolo_model = YOLO(args.pose_checkpoint)
cap = VideoCapture(CAM_INDEX)

def update_frame():
    global prev_time

    img = cap.read()

    results_yolo = yolo_model(img,verbose=False)
        
    # Extract keypoints and scores
    keypoints       = results_yolo[0].keypoints.xy.cpu().numpy()[0] 
    
    # Append the keypoint information to the arrays
    if len(keypoints) > 0:
        keypoint_scores = results_yolo[0].keypoints.conf.cpu().numpy()
        keypoint_array.append(keypoints)
        keypoint_score_array.append(keypoint_scores[0])
    else:
        keypoint_array.append(np.zeros((17, 2)))
        keypoint_score_array.append(np.zeros(17))

    if len(keypoint_array) > FRAME_TO_PROCESS:
        keypoint_array.pop(0)
        keypoint_score_array.pop(0)

        keypoints_list       = np.array(keypoint_array, dtype=np.float32)
        keypoints_score_list = np.array(keypoint_score_array, dtype=np.float32)

        keypoints_list = keypoints_list.reshape(1, *keypoints_list.shape)
        keypoints_score_list = keypoints_score_list.reshape(1, *keypoints_score_list.shape)

        

        torch.cuda.empty_cache()

        fake_anno = dict(frame_dir='',
                        label=-1,
                        img_shape=(480, 640),
                        original_shape=(480, 640),
                        start_index=0,
                        modality='Pose',
                        total_frames=FRAME_TO_PROCESS)


        fake_anno['keypoint'] = keypoints_list
        fake_anno['keypoint_score'] = keypoints_score_list
        result = inference_recognizer(model, fake_anno).cpu().numpy()

        action_label = label_map[result.pred_label[0]]
        action_prob = result.pred_score[result.pred_label[0]]

        print(f'\nAction: {action_label} Prob: {action_prob}')

        if action_label == 'fall' and action_prob > 0.80:
            playsound('demo/bell.mp3')
            time.sleep(2)
            while (keypoint_array):
                keypoint_array.pop(0)
                keypoint_score_array.pop(0)



    # Convert the frame to an ImageTk object
    img = Image.fromarray(results_yolo[0].plot())
    imgtk = ImageTk.PhotoImage(image=img)
    
    # Update the image in the label
    label.imgtk = imgtk
    label.configure(image=imgtk)

    cur_time = time.time()
    delta = cur_time - prev_time
    print('FPS=',1.0/delta)
    prev_time = cur_time
    root.after(1, update_frame)



if __name__ == '__main__':

 
    config = mmengine.Config.fromfile(args.config)
    config.merge_from_dict(args.cfg_options)

    model = init_recognizer(config, args.checkpoint, args.device)

    # Load label_map
    label_map = [x.strip() for x in open(args.label_map).readlines()]

    # Create a Tkinter window
    root = tk.Tk()
    root.title("Live Camera Feed")

    # Create a label to display the video feed
    label = Label(root)
    label.pack()

    # Start updating the frame
    update_frame()

    # Run the Tkinter event loop
    root.mainloop()

    cap.release()
    cv2.destroyAllWindows()
