from moviepy.editor import *
import os

def delete_images(start_folder, image_range_start, image_range_end):
    for root, dirs, files in os.walk(start_folder):
        for file in files:
            if file.endswith('.jpg'):
                image_number = int(file.split('_')[1].split('.')[0])
                if image_range_start <= image_number <= image_range_end:
                    file_path = os.path.join(root, file)
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")

def rename_images(start_folder):
    for root, dirs, files in os.walk(start_folder):
        for file in files:
            if file.endswith('.jpg'):
                parts = file.split('_')
                if len(parts) == 2 and parts[0] == 'image' and parts[1].startswith('000') and parts[1].endswith('.jpg'):
                    image_number = int(parts[1][3:-4])  # 提取数字部分并转换为整数
                    if image_number >= 8:
                        new_image_number = image_number - 8
                        new_file_name = f"image_{new_image_number:05d}.jpg"  # 格式化新的文件名
                        old_file_path = os.path.join(root, file)
                        new_file_path = os.path.join(root, new_file_name)
                        os.rename(old_file_path, new_file_path)
                        print(f"Renamed: {old_file_path} to {new_file_path}")

def rename_folder(folder_path):
    jpg_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
    for i, jpg_file in enumerate(jpg_files, start=1):
        # 新的檔案名稱
        new_name = f'image_{i:03d}.jpg'
        
        # 建立舊檔案路徑和新檔案路徑
        old_path = os.path.join(folder_path, jpg_file)
        new_path = os.path.join(folder_path, new_name)
        
        # 重命名檔案
        os.rename(old_path, new_path)
        print(f'Renamed: {jpg_file} -> {new_name}')

def process_clip_paths(root_path):
    # 获取root path下的所有class path
    class_paths = [os.path.join(root_path, class_dir) for class_dir in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, class_dir))]

    for class_path in class_paths:
        # 获取class path下的所有clip path
        clip_paths = [os.path.join(class_path, clip_dir) for clip_dir in os.listdir(class_path) if os.path.isdir(os.path.join(class_path, clip_dir))]
        print(clip_paths)
        for clip_path in clip_paths:
            # 获取clip path下的所有jpg文件
            jpg_files = [os.path.join(clip_path, jpg_file) for jpg_file in os.listdir(clip_path) if jpg_file.endswith('.jpg')]

            if len(jpg_files) == 40:
                # 删除前8个jpg文件
                for i in range(8):
                    os.remove(jpg_files[i])

                # 重命名剩余的jpg文件
                for i, jpg_file in enumerate(jpg_files[8:]):
                    new_name = f"image_{i:05d}.jpg"
                    os.rename(jpg_file, os.path.join(clip_path, new_name))

def delete_every_two_mp4(root_folder):
    # Walk through the directory
    for root, dirs, files in os.walk(root_folder):
        # Filter and sort the MP4 files
        mp4_files = sorted([file for file in files if file.lower().endswith('.mp4')])
        print(mp4_files)

        # Iterate over the MP4 files and delete every second one
        for i in range(1, len(mp4_files), 2):
            os.remove(os.path.join(root, mp4_files[i]))


def iterate_mp4_files(root_path):
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith(".mp4"):
                file_path = os.path.join(root, file)
                yield file_path

def preprocess_day():

    # labeled data path
    DATA_ROOT =      "E:\\DATA_ALL\\Dateset_day_inf\\1103_K_1440"
    # new folder for preprocessed data
    DATA_PREPROCESSED_ROOT = "E:\\DATA_ALL\\Dateset_day_inf\\1103_K_jpgs"

    PREPROCESS_RESULT = ["mp4", "jpgs"][1]

    # crop params
    start_point = (30, 80)
    end_point = (640, 460)
    size = 224
    angle = -2

    for mp4_file_path in iterate_mp4_files(DATA_ROOT):
        print("Read: "+mp4_file_path)
        processed_clip_path = DATA_PREPROCESSED_ROOT+"\\"+os.path.splitext(os.path.basename(mp4_file_path))[0]

        clip = VideoFileClip(mp4_file_path)
        clip = clip.rotate(angle)
        clip = clip.crop(x1=start_point[0],y1=start_point[1],width=end_point[0]-start_point[0],height=end_point[1]-start_point[1])
        clip = clip.resize((size,size))
        
        if PREPROCESS_RESULT == "jpgs":

            os.makedirs(processed_clip_path, mode=0o777)
            
            total_frames = int(clip.fps * clip.duration)
            # assert total_frames == 32
            assert total_frames == 40

            for i, frame in enumerate(clip.iter_frames()):
                if total_frames == 40:
                    if i < 8:
                        continue
                img = ImageClip(frame)
                img.save_frame(processed_clip_path+f"\\image_{i:05d}.jpg")

        if PREPROCESS_RESULT == "mp4":
            total_frames = int(clip.fps * clip.duration)
            assert total_frames == 40

            clip = clip.subclip(8/clip.fps)
            clip.write_videofile(os.path.join(DATA_PREPROCESSED_ROOT,mp4_file_path[-21:]), codec='libx264')








# root_folder = "E:\\DATA_ALL\\Dataset_F\\Data_Preprocessed_224_32"
# process_clip_paths(root_folder)
# rename_folder("c:\\Users\\ASUS\\Desktop\\2023-intro-AI\\HW1_Embedding_ImageBind\\Dog")

# delete_every_two_mp4("E:\\DATA_ALL\\Dateset_day_inf\\1103_K_1440")

preprocess_day()