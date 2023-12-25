import torch
import cv2
import os
from PIL import Image
import numpy as np
import torch.nn.functional as F
from spatial_transforms import (Compose, Normalize, Resize, CenterCrop,
                                CornerCrop, MultiScaleCornerCrop,
                                RandomResizedCrop, RandomHorizontalFlip,
                                ToTensor, ScaleValue, ColorJitter,
                                PickFirstChannels)
from model import (generate_model, load_pretrained_model, make_data_parallel,
                   get_fine_tuning_parameters)
from models import resnet2p1d
import datetime


# crop params
x1 = 30
y1 = 50
w = 590
h = 280

# resize params
size = 112

def predict_behavior(clip, model):
    model.eval()
    
    spatial_transform = get_spatial_transform()
    clip = preprocessing(clip, spatial_transform)
    
    with torch.no_grad():
        outputs = model(clip)
        # apply softmax and move from gpu to cpu
        outputs = F.softmax(outputs, dim=1).cpu()
        # print(outputs)
        # get best class
        score, class_prediction = torch.max(outputs, 1)

    return score[0], class_prediction[0]

def show_predict_result(clip_path,predicted_class,probability):
    print("\nShowing the result...")
    classes = ['AL','AS','DR','FD','NL','NS']
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    complete_file_name = "c:/Users/BERLIN CHEN/Desktop/CALF/3D_RESNET_V1_2/samples/predict_{}.mp4".format(datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S"))
    out = cv2.VideoWriter(complete_file_name, fourcc, 10.0, (112, 112))
    
    if clip_path[-4:] == ".mp4":
        frames = mp4_to_frames(clip_path)
        for frame in frames:
            image = np.array(frame)
            cv2.putText(image,classes[predicted_class], (10, 105), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 1, cv2.LINE_AA)
            out.write(image)
        out.release()

            
    # else:
    #     for file_name in os.listdir(clip_path):
    #         if file_name.endswith(".jpg"):
    #             image_path = os.path.join(clip_path, file_name)
    #             image = Image.open(image_path)
    #             image = np.array(image)
    #             cv2.putText(image,classes[predicted_class], (100, 600), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), 2, cv2.LINE_AA)
    #             out.write(image)
    #     out.release()
    print("Predict result saved: "+complete_file_name)

def mp4_to_frames(filename):
    cap = cv2.VideoCapture(filename)
    frames = []
    while True:
        ret, frame = cap.read()
        # print("size: "+str(frame.shape))

        if not ret:
            break
        frame = frame[y1:y1+h, x1:x1+w]
        frame = cv2.resize(frame,(112,112))
        frames.append(frame)
    
    # print("clip length: "+str(len(frames)))
    # print("cropped size: "+str(frames[0].shape))
    
    cap.release()
    return frames

def preprocessing(clip, spatial_transform):
    # Applying spatial transformations
    if spatial_transform is not None:
        spatial_transform.randomize_parameters()
        # Before applying spatial transformation you need to convert your frame into PIL Image format
        # (its not the best way, but works)
        clip = [spatial_transform(Image.fromarray(np.uint8(img)).convert('RGB')) for img in clip]
    # Rearange shapes to fit model input
    clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
    clip = torch.stack((clip,), 0)
    return clip


def get_spatial_transform():
    sample_size = 112
    normalize = get_normalize_method() # Normalize without using mean and std
    spatial_transform = [Resize(sample_size)]
    
    spatial_transform.append(CenterCrop(sample_size)) #if we apply center crop of predicting
    
    spatial_transform.append(ToTensor())
    spatial_transform.extend([ScaleValue(1), normalize])
    spatial_transform = Compose(spatial_transform)
    return spatial_transform

def get_normalize_method():
    return Normalize([0.4345, 0.4051, 0.3775],[0.2768, 0.2713, 0.2737])

def resume_model(resume_path):
    model = generate_model()
    print('loading checkpoint {} model...\n'.format(resume_path))
    arch = '{}-{}'.format("resnet2p1d", 18) # original: opt.svm
    checkpoint = torch.load(resume_path, map_location='cpu')
    
    # print(checkpoint['arch'])
    # model.load_state_dict(checkpoint['state_dict'])
    # if hasattr(model, 'module'):
    #     model.module.load_state_dict(checkpoint['state_dict'])
    # else:
    model.load_state_dict(checkpoint['state_dict'],strict=False)  # "strict=False" revised 

    return model

def generate_model():
    model = resnet2p1d.generate_model(model_depth=18,
                                          n_classes=6,
                                          n_input_channels=3,
                                          shortcut_type='B',
                                          conv1_t_size=7,
                                          conv1_t_stride=1,
                                          no_max_pool=False,
                                          widen_factor=1.0)
    return model


if __name__ == '__main__':

    resume_path = "C:\\Users\\BERLIN CHEN\\Desktop\\CALF\\3D_RESNET_V1_2\\results\\save_50.pth"
    clip_path = "C:\\Users\\BERLIN CHEN\\Desktop\\CALF\\3D_RESNET_V1_2\\samples\\NS_20230524-17-47-23.mp4"

    # if clip_path[-4:] == ".mp4":

    frames = mp4_to_frames(clip_path)
    # else:
    #     cap = cv2.VideoCapture(clip_path)
    #     frames = []
    #     for file_name in os.listdir(clip_path):
    #         if file_name.endswith(".jpg"):
    #             image_path = os.path.join(clip_path, file_name)
    #             image = Image.open(image_path)
    #             frames.append(image)

    print("\nPredict the clip: {}".format(clip_path))
    print("\nBy the R(2+1)D model: {}".format(resume_path))
    R2p1_Model = resume_model(resume_path)
    probability, predicted_class = predict_behavior(frames,R2p1_Model)

    classes = ['AL','AS','DR','FD','NL','NS']
    print("\nscore: {}, prediction: {}".format(probability, classes[predicted_class]))

    show_predict_result(clip_path,predicted_class,probability)
