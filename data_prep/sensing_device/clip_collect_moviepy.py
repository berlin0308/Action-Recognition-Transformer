import moviepy.editor as mp
from moviepy.video.fx.crop import crop
import numpy as np
import cv2
import time
import datetime
import os

# width = 640
# height = 480
fps = 10
duration = 4
# screen_size = (320,240)

cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap.set(cv2.CAP_PROP_FPS, fps)
#print(cap.get(cv2.CAP_PROP_EXPOSURE))
cap.set(cv2.CAP_PROP_EXPOSURE, 10)
print(cap.get(cv2.CAP_PROP_EXPOSURE))
#mp4_save_dir = " C:\\Users\\BERLIN CHEN\\Desktop\\CALF\\DATA_PREP\\Unused_Test_Data\\collect_mp4\\"

def make_frame(t):
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
    else:
        return np.zeros((10, 10, 3))
    
def recording(mp4_save_dir):
    
    clip = mp.VideoClip(make_frame, duration=duration)
    # clip = clip.resize(screen_size)
    # clip = clip.fx(crop,x1=0,y1=0,x2=100,y2=100)

    start_time = datetime.datetime.now()
    mp4_save_path = mp4_save_dir + '{}.mp4'.format(start_time.strftime("%Y%m%d-%H-%M-%S"))
    clip.write_videofile(mp4_save_path, fps=fps)
    end_time = datetime.datetime.now()

    elapsed_time = end_time - start_time
    print('Actual write file time: '+str(elapsed_time.total_seconds())+" s")

    mp4_size = os.path.getsize(mp4_save_path)
    print("mp4 size: "+str(mp4_size/1024)+" KB")
    return mp4_save_path


# update_time = datetime.datetime.now()
# while True:
#     now = datetime.datetime.now()
#     if now-update_time > datetime.timedelta(seconds=5):
#         update_time = now
#         mp4_save_path = recording(mp4_save_dir)
#         print("mp4 saved: "+mp4_save_path)
