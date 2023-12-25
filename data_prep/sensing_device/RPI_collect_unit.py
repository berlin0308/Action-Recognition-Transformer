import cv2
#import imagezmq
import datetime
import clip_collect_moviepy
from moviepy.editor import *
import cpu_temp_check

"""
Publisher -- Run in RPI

Collect a mp4 clip per [TimeInterval]
Save mp4 clips in [mp4_save_dir] folder
Transmit frames in clips by NAT (TCP/ImageZMQ)
"""


mp4_save_dir = "/home/berlinpi/Desktop/Sensing_Device/collect_mp4/"
 
#ImageZMQ_Sender = imagezmq.ImageSender(connect_to='tcp://140.112.94.129:2222') #port: 2222

def to_postfix(x):
    if len(str(x))==1: 
        return "__"+str(x)
    if len(str(x))==2: 
        return "_"+str(x)
    return str(x)


now = datetime.datetime.now()
     
date_dir = str(now.strftime("%Y%m%d")) + '/'

if not os.path.exists(mp4_save_dir+date_dir):
    os.makedirs(mp4_save_dir+date_dir)
            
print("\n------------------------\nStart recording !")
            
mp4_save_path = clip_collect_moviepy.recording(mp4_save_dir+date_dir) # record and save mp4
print("mp4 recorded: " + mp4_save_path)
        
     #   print("\nStart sending...")
saved_mp4 = VideoFileClip(mp4_save_path)
cpu_temp_check.check_cpu_temp(76)
       # for i, frame in enumerate(saved_mp4.iter_frames()):
        #    each_frame = ImageClip(frame)
            # ImageZMQ_Sender.send_image(mp4_save_path[:-5] + to_postfix(i), frame)
      #      print("RPI publishes: {}/{}".format(i,int(saved_mp4.duration*saved_mp4.fps)))
       # print(str(int(saved_mp4.duration*saved_mp4.fps))+" images have been sent !\n------------------------")
