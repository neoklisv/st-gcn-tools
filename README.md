# gcn-tools
# st-gcn-tools


For creating the dataset place your videos in 'train/class_name' and 'val/class_name' and simply run "python create_pkl_train_val.py". Example:

train/
  cats/
  dogs/

val/
  cats/
  dogs/


For live inference:

python demo/stgcn_yolov8_live_gui.py --config configs/skeleton/stgcnpp/stgcnpp_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d.py --checkpoint trained_model.pth --pose-checkpoint yolov8s-pose.pt --label-map label_map_trained_model.txt


For video file processing place all videos to be classified to video_files directory and simply run:

python demo/stgcn_yolov8_videos.py --config configs/skeleton/stgcnpp/stgcnpp_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d.py --checkpoint trained_model.pth --pose-checkpoint yolov8s-pose.pt --label-map label_map_trained_model.txt




  
