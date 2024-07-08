import os
import cv2
import numpy as np
import pickle
from ultralytics import YOLO
import random

# Set up the YOLO model for pose estimation
model = YOLO('yolov8m-pose.pt')

# Define the class labels and their corresponding folders
classes = ['fall', 'lying_down','sit_down','sitting', 'stand_up','standing','walking']
class_folders = {
    'fall': 'train/fall',
    'lying_down': 'train/lying_down',
    'sit_down':'train/sit_down',
    'sitting':'train/sitting',
    'stand_up':'train/stand_up',
    'standing':'train/standing',
    'walking': 'train/walking'

}

class_folders_val = {
     'fall': 'val/fall',
    'lying_down': 'val/lying_down',
    'sit_down':'val/sit_down',
    'sitting':'val/sitting',
    'stand_up':'val/stand_up',
    'standing':'val/standing',
    'walking': 'val/walking'
}


# Initialize the data dictionary to store the annotations
data = {
    'split': {
        'xsub_train': [],
        'xsub_val':   []
    },
    'annotations': []
}
count=0
# Process each class folder
for class_idx, class_name in enumerate(classes):
    class_folder = class_folders[class_name]
    video_files = os.listdir(class_folder)
    
    for video_file in video_files:
        video_path = os.path.join(class_folder, video_file)
        video_id = os.path.splitext(video_file)[0]
 
        print('video : ',video_id)
        print(count)
        count+=1
        
        # Open the video file
        video        = cv2.VideoCapture(video_path)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        img_shape    = (int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(video.get(cv2.CAP_PROP_FRAME_WIDTH)))
 
        print('total frames are: ',total_frames)
        print('img_shape is: ',img_shape)
        
        # Initialize keypoint and keypoint_score arrays
        keypoint_array       = []
        keypoint_score_array = []

        if total_frames == 0:
            os.remove(video_path)
            continue
        
        # Process each frame of the video
        for frame_idx in range(total_frames):
            ret, frame = video.read()
            if not ret:
                break
            
            # Perform pose estimation using YOLO
            results = model(frame, verbose=False)
            
            # Extract keypoints and scores
            keypoints       = results[0].keypoints.xy.cpu().numpy()[0]
            
            # Append the keypoint information to the arrays
            if len(keypoints) > 0:
                keypoint_scores = results[0].keypoints.conf.cpu().numpy()
                keypoint_array.append(keypoints)
                keypoint_score_array.append(keypoint_scores[0])
            else:
                keypoint_array.append(np.zeros((17, 2)))
                keypoint_score_array.append(np.zeros(17))
            
            # print(keypoints)
            # print(keypoint_scores)
            # break
        
        # Release the video capture
        video.release()
 
        
        # Convert keypoint and keypoint_score arrays to numpy arrays
        keypoint_array       = np.array(keypoint_array, dtype=np.float32)
        keypoint_score_array = np.array(keypoint_score_array, dtype=np.float32)
 
        keypoint_array = keypoint_array.reshape(1, *keypoint_array.shape)
        keypoint_score_array = keypoint_score_array.reshape(1, *keypoint_score_array.shape)
 
        # print('keypoints array shape is: ',keypoint_array.shape)
        # print('keypoints scores array shape is: ',keypoint_score_array.shape)
        
        # Create an annotation dictionary for the video
        annotation = {
            'frame_dir': video_id,
            'total_frames': total_frames,
            'img_shape': img_shape,
            'original_shape': img_shape,
            'label': class_idx,
            'keypoint': keypoint_array,
            'keypoint_score': keypoint_score_array,
        }
 
        # print(annotation[0])
        
        # Append the annotation to the list
        data['annotations'].append(annotation)
        # break
    # break

count=0
# Process each class folder
for class_idx, class_name in enumerate(classes):
    class_folder = class_folders_val[class_name]
    video_files = os.listdir(class_folder)
    
    for video_file in video_files:
        video_path = os.path.join(class_folder, video_file)
        video_id = os.path.splitext(video_file)[0]
 
        print('video : ',video_id)
        print(count)
        count+=1
        
        # Open the video file
        video        = cv2.VideoCapture(video_path)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        img_shape    = (int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(video.get(cv2.CAP_PROP_FRAME_WIDTH)))
 
        print('total frames are: ',total_frames)
        print('img_shape is: ',img_shape)
        
        # Initialize keypoint and keypoint_score arrays
        keypoint_array       = []
        keypoint_score_array = []

        if total_frames == 0:
            os.remove(video_path)
            continue
        
        # Process each frame of the video
        for frame_idx in range(total_frames):
            ret, frame = video.read()
            if not ret:
                break
            
            # Perform pose estimation using YOLO
            results = model(frame)
            
            # Extract keypoints and scores
            keypoints       = results[0].keypoints.xy.cpu().numpy()[0]
            
            
            # Append the keypoint information to the arrays
            if len(keypoints) > 0:
                keypoint_scores = results[0].keypoints.conf.cpu().numpy()
                keypoint_array.append(keypoints)
                keypoint_score_array.append(keypoint_scores[0])
            else:
                keypoint_array.append(np.zeros((17, 2)))
                keypoint_score_array.append(np.zeros(17))

        # Release the video capture
        video.release()
 
        
        # Convert keypoint and keypoint_score arrays to numpy arrays
        keypoint_array       = np.array(keypoint_array, dtype=np.float32)
        keypoint_score_array = np.array(keypoint_score_array, dtype=np.float32)
 
        keypoint_array = keypoint_array.reshape(1, *keypoint_array.shape)
        keypoint_score_array = keypoint_score_array.reshape(1, *keypoint_score_array.shape)
        
        # Create an annotation dictionary for the video
        annotation = {
            'frame_dir': video_id,
            'total_frames': total_frames,
            'img_shape': img_shape,
            'original_shape': img_shape,
            'label': class_idx,
            'keypoint': keypoint_array,
            'keypoint_score': keypoint_score_array,
        }
 
        # Append the annotation to the list
        data['annotations'].append(annotation)
        # break
    # break
 
# Split the dataset into train and val sets (you can adjust the split ratio as needed)
all_videos = []
 
# Collect all video identifiers
for class_name, folder in class_folders.items():
    video_files = [os.path.splitext(f)[0] for f in os.listdir(folder) if f.endswith('.avi')]
    all_videos.extend(video_files)
 
# Shuffle the list of videos
random.shuffle(all_videos)


all_videos_val = []
 
# Collect all video identifiers
for class_name, folder in class_folders_val.items():
    video_files = [os.path.splitext(f)[0] for f in os.listdir(folder) if f.endswith('.avi')]
    all_videos_val.extend(video_files)
 
# Shuffle the list of videos
random.shuffle(all_videos_val)
 
 

# Create the split
data['split'] = {
    'xsub_train': all_videos,
    'xsub_val': all_videos_val
}
 
# Save the data dictionary as a pickle file
output_file = 'fall_dataset_train_val_not_normalized.pkl'
with open(output_file, 'wb') as f:
    pickle.dump(data, f)