import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from PIL import Image
from predict import (resume_model, predict_behavior)
import cv2


classes = ['AL','AS','DR','FD','NL','NS']
num_classes = 6
modified_classes = ['NL','AL','NS','AS','FD','DR']
modified_classes_official = ['Non-active Lying','Active Lying','Non-active Standing','Active Standing','Feeding','Drinking']

confusion_mat = np.zeros((num_classes, num_classes), dtype=int)

def class_id_modified(class_id):
    if class_id==0: # AL
        return 1
    if class_id==1: # AS
        return 3
    if class_id==2: # DR
        return 5
    if class_id==3: # FD
        return 4
    if class_id==4: # NL
        return 0
    if class_id==5: # NS
        return 2

def predict(clip_path):
    frames = []
    for file_name in os.listdir(clip_path):
        # print("Predicting: "+str(clip_path))
        if file_name.endswith(".jpg"):
            image_path = os.path.join(clip_path, file_name)
            image = Image.open(image_path)
            # image.show()
            frames.append(image)
        
        probability, predicted_class = predict_behavior(frames,R2p1_Model)
        # print("\nscore: {}, prediction: {}".format(probability, classes[predicted_class]))
        
    return predicted_class

# root path
root_path = "/home/ubuntu/CALF/3D_RESNET_V1_2/calf_data/jpgs"

# trained model
resume_path = "/home/ubuntu/CALF/3D_RESNET_V1_2/results/save_50.pth"
R2p1_Model = resume_model(resume_path)
os.environ['CUDA_VISIBLE_DEVICES']='0'

for class_path in os.listdir(root_path):
    class_dir = os.path.join(root_path, class_path)
    if os.path.isdir(class_dir):
        ground_truth = classes.index(class_path)
        print("Ground truth: "+str(ground_truth)+" "+str(class_path))
        
        # count = 0
        for clip_path in os.listdir(class_dir):
            # count += 1
            # if count==1:
            #     break
            clip_dir = os.path.join(class_dir, clip_path)
            if os.path.isdir(clip_dir):
                prediction = int(predict(clip_dir))
                print("Prediction: "+str(prediction)+" "+str(classes[prediction]))
                confusion_mat[class_id_modified(ground_truth)][class_id_modified(prediction)] += 1

print("Confusion Matrix:")
print(confusion_mat)
plt.figure(figsize=(10, 8))
plt.imshow(confusion_mat, cmap='Blues')
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted',fontsize=14)
plt.ylabel('Actual',fontsize=14)
tick_marks = np.arange(num_classes)
plt.xticks(tick_marks, modified_classes_official,rotation=30)
plt.yticks(tick_marks, modified_classes_official)
plt.show()
plt.savefig('/home/ubuntu/CALF/3D_RESNET_V1_2/plots/confusion_matrix.png', format='png')
