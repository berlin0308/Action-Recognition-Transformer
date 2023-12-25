import numpy as np
import matplotlib.pyplot as plt

classes_default = ['Active\nLying','Active\nStanding','Drinking','Feeding','Non-active\nLying','Non-active\nStanding','Ruminating','X']
classes_final = ['Non-active\nLying','Active\nLying','Non-active\nStanding','Active\nStanding','Feeding','Drinking','Ruminating','X']


def load_matrix_from_txt(filename):
    matrix = np.loadtxt(filename)
    accuracy = np.trace(matrix) / np.sum(matrix)

    true_positives = np.diag(matrix)
    false_positives = np.sum(matrix, axis=0) - true_positives
    false_negatives = np.sum(matrix, axis=1) - true_positives

    precision = np.divide(true_positives, true_positives + false_positives, out=np.zeros_like(true_positives), where=(true_positives + false_positives) != 0)
    recall = np.divide(true_positives, true_positives + false_negatives, out=np.zeros_like(true_positives), where=(true_positives + false_negatives) != 0)

    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10) # adding a small constant to avoid division by zero

    # Calculating weighted F1-score
    weights = np.sum(matrix, axis=1)
    weighted_f1_score = np.sum(f1_scores * weights) / np.sum(weights)

    print(f'\n\nTop-1 Accuracy: {accuracy} \nWeighted F1-score: {weighted_f1_score}\n\n')

    return matrix.astype(int)


"""
filename: A confusion matrix txt of the inference result ( modified classes )
"""
if __name__ == '__main__':

    # filename = '1103_day_matrix.txt'
    # filename = '1112_day_matrix.txt'
    # filename = '1112_day_matrix.txt'
    filename = 'test_J_matrix.txt'

    loaded_matrix = load_matrix_from_txt(filename)
    print(loaded_matrix)

    

    num_classes = 7

    new_conf_matrix = np.zeros((num_classes, num_classes))

    for i in range(num_classes):
        for j in range(num_classes):
            new_i = classes_final.index(classes_default[i])
            new_j = classes_final.index(classes_default[j])
            
            new_conf_matrix[new_i, new_j] = loaded_matrix[i][j]

    print(new_conf_matrix)

    conf_matrix = []
    for row in loaded_matrix:
        new_row = list(row / float(sum(row)))
        conf_matrix.append(new_row)

    print(conf_matrix)

    # color = ['#46959num_classes', '#5BA199', '#BBC6C8','#E5E3E4','#E0D5BE','#DDBEAA']

    plt.figure(figsize=(10, 8))
    plt.imshow(conf_matrix, cmap='Greys')
    plt.title('Confusion Matrix')
    # plt.colorbar(shrink=0.9)
    plt.xlabel('Predicted',fontsize=14)
    plt.ylabel('Actual',fontsize=14)
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, classes_final[:num_classes]) #,rotation=30
    plt.yticks(tick_marks, classes_final[:num_classes])

    for i in range(num_classes):
        for j in range(num_classes):
            if i == j:
                plt.text(j, i, str(format(round(conf_matrix[i][j],4)*100 ,'.2f')), ha='center', va='center', color='white', fontsize=12)
            else:
                plt.text(j, i, str(format(round(conf_matrix[i][j],4)*100 ,'.2f')), ha='center', va='center', color='black', fontsize=12)

    plt.show()
    plt.savefig('confusion_matrix_V8.png', format='png')


    
    plt.figure(figsize=(10, 8))
    plt.imshow(loaded_matrix, cmap='Greys')
    plt.title('Confusion Matrix')
    # plt.colorbar(shrink=0.9)
    plt.xlabel('Predicted',fontsize=14)
    plt.ylabel('Actual',fontsize=14)
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, classes_final[:num_classes], rotation=30) #,rotation=30
    plt.yticks(tick_marks, classes_final[:num_classes])

    for i in range(num_classes):
        for j in range(num_classes):
            if i==0 and j==0:
                plt.text(j, i, str(loaded_matrix[i][j]), ha='center', va='center', color='white', fontsize=12)
            else:
                plt.text(j, i, str(loaded_matrix[i][j]), ha='center', va='center', color='black', fontsize=12)

    plt.show()