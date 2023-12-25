import os
import cv2
from moviepy.editor import *
from moviepy.editor import VideoFileClip, clips_array
from moviepy import *
import numpy as np

# crop params
start_point = (10, 280)
end_point = (640, 480)
size = 224
angle = -2

from moviepy.editor import VideoFileClip, clips_array


def view_crop_rotate(mp4_file_path, mp4_output):
    clip = VideoFileClip(mp4_file_path)

    clip = clip.rotate(angle)
    
    # new_clip = clip.crop(x1=start_point[0],y1=start_point[1],width=end_point[0]-start_point[0],height=end_point[1]-start_point[1])
    # new_clip = new_clip.resize((size,size))
    # processed_clip_path = "aabghilgrggh"
    # os.makedirs(processed_clip_path, mode=0o777)
    # for i, frame in enumerate(new_clip.iter_frames(fps=10), start=1):
    #     if i < 8:
    #         continue
    #     img = ImageClip(frame)
    #     img.save_frame(processed_clip_path+f"\\image_{i-8:05d}.jpg")


    images = []
    for i, frame in enumerate(clip.iter_frames(fps=10), start=1):
        if i < 8:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        color = (0, 255, 0)
        thickness = 2
        cv2.rectangle(frame, start_point, end_point, color, thickness)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        images.append(frame)

    final_clip = ImageSequenceClip(images, fps=10)
    final_clip.write_videofile(mp4_output, codec='libx264')

view_crop_rotate('E:\DATA_ALL\\Dataset_LongTerm\\20231209_OP_full\\20231209-00-00-03.mp4',"aaaab.mp4")

