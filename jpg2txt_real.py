#from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
#from moviepy.editor import VideoFileClip
import os
import shutil
import random
import argparse
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument("--path", type=Path, help='path')
parser.add_argument("--name", type=str, help='name')
args = parser.parse_args()
count = -1
subs = 0
fuck = 0

for path, subdirs, class_name in os.walk("calf_data/Dataset_V8/"):
    # print(args.path)
    if count == -1:
        hey = subdirs
        count = 0
        continue
    if subdirs == []:
        continue

    # if count==6:
    #     continue
    # print(hey[count])
    # print(subdirs)
    name_count = 0

    txtname = './'+"calf_data/V8_10_0_txt/"+hey[count]+".txt"
    for filename in subdirs:
        name_count += 1
        #print(filename)

    if name_count==8:
        continue

    ntrain = int(name_count*0.9)
    nval = name_count - ntrain
    ntest = 0

    ntrain = name_count
    nval = 0
    ntest = 0

    list1 = [1]*int(ntrain)+[2]*int(nval) + [3]*int(ntest)
    print("train: {}, validation: {}, test: {}".format(ntrain, nval, ntest))
    random.shuffle(list1)
    with open(txtname, "w") as a:
        for i, filename in enumerate(subdirs):
            a.write(str(filename) + ".mp4" + " " + str(list1[i]) + os.linesep)
    print(name_count)

    count += 1







