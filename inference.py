import time
import json
from collections import defaultdict
import cv2
import numpy
import statistics
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from utils import AverageMeter
import csv
from spatial_transforms import (Compose, Normalize, Resize, CenterCrop,
                                CornerCrop, MultiScaleCornerCrop,
                                RandomResizedCrop, RandomHorizontalFlip,
                                ToTensor, ScaleValue, ColorJitter,
                                PickFirstChannels)
import datetime
import os
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


def get_video_results(outputs, class_names, output_topk):
    sorted_scores, locs = torch.topk(outputs,
                                     k=min(output_topk, len(class_names)))

    video_results = []
    for i in range(sorted_scores.size(0)):
        video_results.append({
            'label': class_names[locs[i].item()],
            'score': sorted_scores[i].item()
        })

    return video_results


def inference(data_loader, model, result_path, class_names, no_average,
              output_topk, time_feature=False):
    print('\ninference time\n')

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    results = {'results': defaultdict(list)}

    end_time = time.time()

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            data_time.update(time.time() - end_time)

            video_ids, segments = zip(*targets)
            outputs = model(inputs)
            outputs = F.softmax(outputs, dim=1).cpu()

            print(inputs,type(inputs))
            print(outputs,type(outputs))


            for j in range(outputs.size(0)):
                results['results'][video_ids[j]].append({
                    'segment': segments[j],
                    'output': outputs[j]
                })

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            print('[{}/{}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                      i + 1,
                      len(data_loader),
                      batch_time=batch_time,
                      data_time=data_time))

    inference_results = {'results': {}}
    if not no_average:
        for video_id, video_results in results['results'].items():
            video_outputs = [
                segment_result['output'] for segment_result in video_results
            ]
            video_outputs = torch.stack(video_outputs)
            average_scores = torch.mean(video_outputs, dim=0)
            inference_results['results'][video_id] = get_video_results(
                average_scores, class_names, output_topk)
    else:
        for video_id, video_results in results['results'].items():
            inference_results['results'][video_id] = []
            for segment_result in video_results:
                segment = segment_result['segment']
                result = get_video_results(segment_result['output'],
                                           class_names, output_topk)
                inference_results['results'][video_id].append({
                    'segment': segment,
                    'result': result
                })

    with result_path.open('w') as f:
        json.dump(inference_results, f)


def get_normalize_method(mean, std, no_mean_norm, no_std_norm):
    if no_mean_norm:
        if no_std_norm:
            return Normalize([0, 0, 0], [1, 1, 1])
        else:
            return Normalize([0, 0, 0], std)
    else:
        if no_std_norm:
            return Normalize(mean, [1, 1, 1])
        else:
            return Normalize(mean, std)


def get_spatial_transform(opt, kind):
    normalize = get_normalize_method(opt.mean, opt.std, opt.no_mean_norm,
                                 opt.no_std_norm)
    
    spatial_transform = [Resize(opt.sample_size)]
    spatial_transform.append(ToTensor())
    spatial_transform.extend([ScaleValue(opt.value_scale), normalize])
    spatial_transform = Compose(spatial_transform)
    return spatial_transform


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

def predict(clip, model, spatial_transform):
    # Set mode eval mode
    model.eval()
    # do some preprocessing steps
    clip = preprocessing(clip, spatial_transform)
    # don't calculate grads
    with torch.no_grad():
        # apply model to input
        outputs = model(clip)
        # apply softmax and move from gpu to cpu
        outputs = F.softmax(outputs, dim=1).cpu()

        # print(outputs)
        # get best class
        score, class_prediction = torch.max(outputs, 1)
        # print("score: {}, class prediction: {}".format(score, class_prediction))
    return score[0], class_prediction[0], outputs[0]


def inf_day_main(opt, model):

    import os
    from result_day.csv_util import csv_append_result
    from moviepy.editor import VideoFileClip

    classes = ['AL','AS','DR','FD','NL','NS']

    # day_root_path = "Data_Preprocessed_Evaluated_Final/AL"
    day_root_path = "./Dataset_Inf_day/1101_L_2880"
    # day_root_path = "./Dataset_Inf_day/1101_I_2880"
    result_csv_path = "./inferencing_result/inf_day_"

    MODE = ['preprocessed_folder_jpgs', 'unprocessed_mp4'][1]
    print("Current MODE: " + MODE)

    # preprocess parameters
    start_point = (20, 50)
    end_point = (630, 350)
    size = 224

    if MODE == 'preprocessed_folder_jpgs':
        for i, clip_name in enumerate(os.listdir(day_root_path)):
            if i == 0:
                result_csv_path += clip_name[:8]
                result_csv_path += '.csv'

            clip_path = os.path.join(day_root_path,clip_name)

            clip = []
            for image_file in os.listdir(clip_path):

                if image_file[:9] == "image_000":
                    image_path = os.path.join(clip_path, image_file)
                    # print(image_path)
                    image = cv2.imread(image_path)
                    clip.append(image)
        
            score, predicted_class, probs = predict(clip, model, get_spatial_transform(opt, "resnet"), )
            print("score: {}, prediction: {}, class name: {}\nprobs: {}".format(score, predicted_class, classes[predicted_class], probs))

            #csv_append_result(time=clip_name, action=predicted_class, score=probability, csv_path=result_csv_path)

    if MODE == 'unprocessed_mp4':
        print(day_root_path)
        for i, mp4_file_path in enumerate(iterate_mp4_files(day_root_path)):

            print("Read: "+mp4_file_path)
            clip_name = mp4_file_path.split('\\')[1][:17]
            
            if i == 0:
                result_csv_path += clip_name[:8]
                result_csv_path += '.csv'

            mv_clip = VideoFileClip(mp4_file_path)
            mv_clip = mv_clip.crop(x1=start_point[0],y1=start_point[1],width=end_point[0]-start_point[0],height=end_point[1]-start_point[1])
            mv_clip = mv_clip.resize((size,size))

            clip = []
            for i, frame in enumerate(mv_clip.iter_frames(fps=10)):
                if i < 8:
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                # img = np.ndarray(img)

                clip.append(frame)

            score, predicted_class, probs = predict(clip, model, get_spatial_transform(opt, "resnet"), )
            print("score: {}, prediction: {}, class name: {}\nprobs: {}".format(score, predicted_class, classes[predicted_class], probs))
            #csv_append_result(time=clip_name, action=predicted_class, score=probability, csv_path=result_csv_path)


    print("--------------------------------\nEnd of predictions")
    print("Result:", result_csv_path)

def inf_test_data(opt, model):
    
    root_path = "calf_data/Data_Preprocessed_J_crop_jpgs"

    num_classes = 7
    classes = ['AL','AS','DR','FD','NL','NS','RM']
    modified_classes_official = ['Non-active Lying','Active Lying','Non-active Standing','Active Standing','Feeding','Drinking','Ruminating']


    confusion_mat = np.zeros((num_classes, num_classes), dtype=int)

    y_true = []
    y_pred = []

    for class_path in classes:
        class_dir = os.path.join(root_path, class_path)
        print(class_dir)
        if os.path.isdir(class_dir):
            ground_truth = classes.index(class_path)
            print("Ground truth: "+str(ground_truth)+" "+str(class_path))
            
            for clip_path in os.listdir(class_dir):
                clip_dir = os.path.join(class_dir, clip_path)
                print(clip_dir)

                clip = []
                for image_file in os.listdir(clip_dir):
                    if image_file[:9] == "image_000":
                        image_path = os.path.join(clip_dir, image_file)
                        # print(image_path)
                        image = cv2.imread(image_path)
                        clip.append(image)
            
                score, predicted_class, probs = predict(clip, model, get_spatial_transform(opt, "resnet"), )
                print("score: {}, prediction: {}, class name: {}\n".format(score, predicted_class, classes[predicted_class]))

                y_true.append(ground_truth)
                y_pred.append(predicted_class)
                confusion_mat[class_id_modified(ground_truth)][class_id_modified(predicted_class)] += 1

    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)
    print(f'Weighted F1-score: {weighted_f1} \nAccuracy:{accuracy}')


    np.set_printoptions(suppress=True)
    np.savetxt(str(opt.result_path)+'/test_J_matrix.txt', confusion_mat, fmt='%d')
    print("Confusion Matrix:")
    print(confusion_mat)

def inf_day_eval(opt, model, csv=False):

    day_root_path = "calf_data/1112_K_jpgs"
    ground_truth_path = "calf_data/1112_Labeled_Preprocessed_mp4"

    if csv:
        from result_day.csv_util import (create_new_csv, csv_append_result)
        result_csv_path = "result_day/inf_day_" + "20231112" + ".csv"
        create_new_csv(result_csv_path)


    classes = ['AL','AS','DR','FD','NL','NS','RM','X']
    
    confusion_mat = np.zeros((8, 8), dtype=int)
    y_true = []
    y_pred = []
    
    for i, clip_name in enumerate(sorted(os.listdir(day_root_path),key=str)):

        clip_path = os.path.join(day_root_path,clip_name)
        print("\n"+clip_path)

        # fetch ground truth of the clip
        truth = -1
        for i, class_path in enumerate(classes):
            class_dir = os.path.join(ground_truth_path, class_path)
            # print(os.path.join(class_dir, clip_name))
            # if os.path.isdir(os.path.join(class_dir, "20231112"+clip_name)):
            if os.path.exists(os.path.join(class_dir, "20231112-"+clip_name+".mp4")):
                truth = i
                break
        
        assert (truth != -1)
        print("ground truth: ", truth)

        # make prediction
        clip = []
        for image_file in os.listdir(clip_path):

            if image_file[:9] == "image_000":
                image_path = os.path.join(clip_path, image_file)
                # print(image_path)
                image = cv2.imread(image_path)
                clip.append(image)
    
        score, predicted_class, probs = predict(clip, model, get_spatial_transform(opt, "resnet"), )
        # print("score: {}, prediction: {}, class name: {}\nprobs: {}".format(score, predicted_class, classes[predicted_class], probs))
        print("prediction: ", int(predicted_class))

        if truth != 7:
            y_true.append(truth)
            y_pred.append(predicted_class)
        
        confusion_mat[class_id_modified(truth)][class_id_modified(int(predicted_class))] += 1
        if csv:
            csv_append_result(csv_path=result_csv_path, time=clip_name, action=predicted_class, score=score, truth=truth, probs=probs)
    

    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)
    print(f'Weighted F1-score: {weighted_f1} \nAccuracy:{accuracy}')

    np.set_printoptions(suppress=True)
    np.savetxt(str(opt.result_path)+'/1112_day_matrix.txt', confusion_mat, fmt='%d')
    print("Confusion Matrix:")
    print(confusion_mat)

def iterate_mp4_files(root_path):
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith(".mp4"):
                file_path = os.path.join(root, file)
                yield file_path


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
    if class_id==6: # RM
        return 6
    if class_id==7: # X
        return 7
    
