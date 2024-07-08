"""
conda activate pyskl
python demo/stgcn_yolov8.py --config configs/stgcn++/stgcn++_ntu120_xsub_hrnet/j.py     --checkpoint http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu120_xsub_hrnet/j.pth
"""
import argparse
import cv2
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
import supervision as sv
import glob

import cv2
import mmcv
import mmengine
import torch
from mmengine import DictAction
from mmengine.utils import track_iter_progress

from mmaction.apis import (detection_inference, inference_recognizer,
                           init_recognizer, pose_inference)
from mmaction.registry import VISUALIZERS
from mmaction.utils import frame_extract
from mmengine.utils import ProgressBar



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

def pose_inference(args, video):
   
    ret = []
    predictions_for_viz = []
    print('Performing Human Pose Estimation for each frame')
    prog_bar = ProgressBar(sv.VideoInfo.from_video_path(video_path=video).total_frames)

    # Initialize keypoint and keypoint_score arrays
    keypoint_array       = []
    keypoint_score_array = []
    for f in sv.get_video_frames_generator(source_path=video):
        prog_bar.update()
        results = yolo_model(f,verbose=False)
            
        # Extract keypoints and scores
        # keypoints       = results[0].keypoints.xyn.cpu().numpy()[0] #NORMALIZED!
        keypoints       = results[0].keypoints.xy.cpu().numpy()[0] # NOT NORMALIZED!
        
        # Append the keypoint information to the arrays
        if len(keypoints) > 0:
            keypoint_scores = results[0].keypoints.conf.cpu().numpy()
            keypoint_array.append(keypoints)
            keypoint_score_array.append(keypoint_scores[0])
        else:
            keypoint_array.append(np.zeros((17, 2)))
            keypoint_score_array.append(np.zeros(17))


        
    # Convert keypoint and keypoint_score arrays to numpy arrays
    keypoint_array       = np.array(keypoint_array, dtype=np.float32)
    keypoint_score_array = np.array(keypoint_score_array, dtype=np.float32)

    keypoint_array = keypoint_array.reshape(1, *keypoint_array.shape)
    keypoint_score_array = keypoint_score_array.reshape(1, *keypoint_score_array.shape)
    return keypoint_array, keypoint_score_array



def main(video_path):


    video_info = sv.VideoInfo.from_video_path(video_path=video_path)
    num_frame = video_info.from_video_path(video_path=video_path).total_frames
    h, w,= video_info.height, video_info.width

    config = mmengine.Config.fromfile(args.config)
    config.merge_from_dict(args.cfg_options)


    model = init_recognizer(config, args.checkpoint, args.device)

    # Load label_map
    label_map = [x.strip() for x in open(args.label_map).readlines()]

    # Get Human detection results
    keypoints_list, keypoints_score_list = pose_inference(args, video_path)



    torch.cuda.empty_cache()

    fake_anno = dict(frame_dir='',
                     label=-1,
                     img_shape=(h, w),
                     original_shape=(h, w),
                     start_index=0,
                     modality='Pose',
                     total_frames=num_frame)


    fake_anno['keypoint'] = keypoints_list
    fake_anno['keypoint_score'] = keypoints_score_list


    result = inference_recognizer(model, fake_anno).cpu().numpy()

    action_label = label_map[result.pred_label[0]]


    print(f'\nAction: {action_label}')

    line = ''
    line += video_path.rsplit('/')[-1] + ','

    results = result.pred_score.tolist()
    results_sorted = []

    for act,prob in enumerate(results):
        results_sorted.append((act,prob))

    results_sorted = sorted(results_sorted, key= lambda x:x[1], reverse=True)

    for act,prob in results_sorted:
        line += '[' + label_map[int(act)] + ',' + str(prob) + '],' 

    line += '\n'
    file = open("demo/log.txt", "a")
    file.write(line)



if __name__ == '__main__':

    list_videos = glob.glob('demo/fall_videos/*.avi')

    for video in list_videos:
        frame_no=0
        print(video)
        main(video)

