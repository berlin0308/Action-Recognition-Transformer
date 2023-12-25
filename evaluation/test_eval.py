import json
import argparse
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np


def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def load_ground_truth(ground_truth_path, subset):
    with ground_truth_path.open('r') as f:
        data = json.load(f)

    class_labels_map = get_class_labels(data)

    ground_truth = []
    for video_id, v in data['database'].items():
        if subset != v['subset']:
            continue
        this_label = v['annotations']['label']
        ground_truth.append((video_id, class_labels_map[this_label]))

    return ground_truth, class_labels_map


def load_result(result_path, top_k, class_labels_map):
    with result_path.open('r') as f:
        data = json.load(f)

    result = {}
    for video_id, v in data['results'].items():
        labels_and_scores = []
        for this_result in v:
            label = class_labels_map[this_result['label']]
            score = this_result['score']
            labels_and_scores.append((label, score))
        labels_and_scores.sort(key=lambda x: x[1], reverse=True)
        result[video_id] = list(zip(*labels_and_scores[:top_k]))[0]
    return result


def remove_nonexistent_ground_truth(ground_truth, result):
    exist_ground_truth = [line for line in ground_truth if line[0] in result]

    return exist_ground_truth


def evaluate(ground_truth_path, result_path, subset, top_k, ignore):
    print('load ground truth')
    ground_truth, class_labels_map = load_ground_truth(ground_truth_path,
                                                       subset)
    print('number of ground truth: {}'.format(len(ground_truth)))

    print('load result')
    result = load_result(result_path, top_k, class_labels_map)
    print('number of result: {}'.format(len(result)))

    n_ground_truth = len(ground_truth)
    ground_truth = remove_nonexistent_ground_truth(ground_truth, result)
    if ignore:
        n_ground_truth = len(ground_truth)

    print('calculate top-{} accuracy'.format(top_k))
    correct = [1 if line[1] in result[line[0]] else 0 for line in ground_truth]
    accuracy = sum(correct) / n_ground_truth

    print('top-{} accuracy: {}'.format(top_k, accuracy))

    # Calculate confusion matrix
    y_true = [line[1] for line in ground_truth]
    y_pred = [result[line[0]][0] for line in ground_truth]  # Assuming top-1 prediction
    conf_matrix = confusion_matrix(y_true, y_pred, labels=list(class_labels_map.values()))

    # Calculate other performance metrics using the confusion matrix if needed
    # precision, recall, f1-score, ... 
    metrics = classification_report(y_true, y_pred)
    print(metrics)


    # Save confusion matrix to a text file
    np.savetxt('conf_matrix.txt', conf_matrix, fmt='%d', delimiter='\t')




    return accuracy


if __name__ == '__main__':
    
    
    accuracy = evaluate(Path('./r2p1d18_b8_lr4e-3_0912/calf_annotation.json'), Path('./r2p1d18_b16_lr1e-2_layer3_0915/test.json'), ['testing','validation'][0],
                        1, False)

    with (Path('top-1_accuracy.txt')).open('w') as f:
        f.write(str(accuracy))