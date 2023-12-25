import cv2
import os
import numpy as np
import time
from PIL import Image
from spatial_transforms import (Compose, Normalize, Resize, CenterCrop,
                                CornerCrop, MultiScaleCornerCrop,
                                RandomResizedCrop, RandomHorizontalFlip,
                                ToTensor, ScaleValue, ColorJitter,
                                PickFirstChannels)


def get_spatial_transform(sample_size=224, train_crop_min_scale=0.8, train_crop_min_ratio=0.6, mean=[0.4345,0.4051,0.3775], std=[0.2768,0.2713,0.2737], value_scale=1):

    spatial_transform = []                                 
    spatial_transform.append(
            RandomResizedCrop(
                sample_size, (train_crop_min_scale, 1.0),
                (train_crop_min_ratio, 1.0 / train_crop_min_ratio)))
    # normalize = Normalize(mean, std)
    spatial_transform.append(ColorJitter(brightness=0.3, contrast=0.5, saturation=0.5, hue=0.05))
    spatial_transform.append(ToTensor())
    # spatial_transform.append(ScaleValue(value_scale))
    # spatial_transform.append(normalize)
    spatial_transform = Compose(spatial_transform)

    return spatial_transform

def process_video(file_path):
    cap = cv2.VideoCapture(file_path)

    aug_param = get_spatial_transform()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)

        frame = aug_param(frame)

        # Convert back to NumPy array and BGR for OpenCV display
        frame = frame.numpy().transpose(1, 2, 0) * 255
        frame = frame.astype(np.uint8)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        cv2.imshow('Transformed Video', frame)
        time.sleep(0.05)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Parameters
root_dir = 'E:\DATA_ALL\Dataset_V6\Data_Labeled'

for subdir, dirs, files in os.walk(root_dir):
        count = 0
        for file in files:
            if count > 5:
                 break
            
            if file.endswith('.mp4'):
                file_path = os.path.join(subdir, file)
                process_video(file_path)
                count += 1