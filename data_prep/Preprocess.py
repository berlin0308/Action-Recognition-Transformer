from moviepy.editor import *
from moviepy import *
import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# labeled data path
DATA_LABELED_ROOT =      'E:\DATA_ALL\\Dataset_LongTerm\\20231210_OP_full'
# new folder for preprocessed data
DATA_PREPROCESSED_ROOT = 'E:\\DATA_ALL\\Dateset_day_inf\\1210_O_1440'

PREPROCESS_RESULT = ["mp4", "jpgs"][0]

# crop params
start_point = (10, 10)
end_point = (640, 280)
size = 224
angle = -2

def iterate_mp4_files(root_path):
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith(".mp4"):
                file_path = os.path.join(root, file)
                yield file_path

def process_mp4_files(mp4_file_path):
    
    print("Read: "+mp4_file_path)
    processed_clip_path = DATA_PREPROCESSED_ROOT+"\\"+os.path.splitext(os.path.basename(mp4_file_path))[0]

    clip = VideoFileClip(mp4_file_path)
    
    clip = clip.rotate(angle)
    clip = clip.crop(x1=start_point[0],y1=start_point[1],width=end_point[0]-start_point[0],height=end_point[1]-start_point[1])
    clip = clip.resize((size,size))

    if PREPROCESS_RESULT == "jpgs":

        os.makedirs(processed_clip_path, mode=0o777)
        
        total_frames = int(clip.fps * clip.duration)
        assert total_frames == 32
        # assert total_frames == 40

        for i, frame in enumerate(clip.iter_frames()):
            # if total_frames == 40:
            #     if i < 8:
            #         continue
            img = ImageClip(frame)
            img.save_frame(processed_clip_path+f"\\image_{i:05d}.jpg")

    if PREPROCESS_RESULT == "mp4":
        total_frames = int(clip.fps * clip.duration)
        assert total_frames == 40

        clip = clip.subclip(8/clip.fps)
        clip.write_videofile(os.path.join(DATA_PREPROCESSED_ROOT,mp4_file_path[-21:]), codec='libx264')
    
os.makedirs(DATA_PREPROCESSED_ROOT, mode=0o777, exist_ok=True)

with ThreadPoolExecutor(max_workers=6) as executor:
    for mp4_file_path in iterate_mp4_files(DATA_LABELED_ROOT):
        executor.submit(process_mp4_files, mp4_file_path)

# # build new path for preprocessed data
# for labeled_class in ['\\NL','\\AL','\\NS','\\AS','\\FD','\\DR','\\RM','\\X']:
#     os.makedirs(DATA_PREPROCESSED_ROOT+labeled_class, mode=0o777)


# for labeled_class in ['\\NL','\\AL','\\NS','\\AS','\\FD','\\DR','\\RM','\\X']:
#     CLASS_ROOT = DATA_LABELED_ROOT + labeled_class
#     print("Start to process -- "+CLASS_ROOT)
#     for mp4_file_path in iterate_mp4_files(CLASS_ROOT):
#         print("Read: "+mp4_file_path)
#         processed_clip_path = DATA_PREPROCESSED_ROOT+labeled_class+"\\"+os.path.splitext(os.path.basename(mp4_file_path))[0]

#         clip = VideoFileClip(mp4_file_path)
        
#         # clip = clip.rotate(angle)
#         # clip = clip.crop(x1=start_point[0],y1=start_point[1],width=end_point[0]-start_point[0],height=end_point[1]-start_point[1])
#         clip = clip.resize((size,size))
        
#         if PREPROCESS_RESULT == "jpgs":

#             os.makedirs(processed_clip_path, mode=0o777)
            
#             total_frames = int(clip.fps * clip.duration)
#             assert total_frames == 32
#             # assert total_frames == 40

#             for i, frame in enumerate(clip.iter_frames()):
#                 # if total_frames == 40:
#                 #     if i < 8:
#                 #         continue
#                 img = ImageClip(frame)
#                 img.save_frame(processed_clip_path+f"\\image_{i:05d}.jpg")

#         if PREPROCESS_RESULT == "mp4":
#             total_frames = int(clip.fps * clip.duration)
#             assert total_frames == 40

#             clip = clip.subclip(8/clip.fps)
#             clip.write_videofile(os.path.join(DATA_PREPROCESSED_ROOT + labeled_class,mp4_file_path[-21:]), codec='libx264')






