import matplotlib.pyplot as plt
import numpy as np

modified_classes = ['NL','AL','NS','AS','FD','DR','RM']
modified_classes_official = ['Non-active\nLying','Active\nLying','Non-active\nStanding','Active\nStanding','Feeding','Drinking','Ruminating']


Dataset_B = [52, 41, 42, 95, 32, 36, 1]
Dataset_Plus = [28, 20, 30, 189, 4, 4, 8]
Dataset_D = [8, 6, 4, 12, 6, 4, 0]
Dataset_E =  [25, 14, 8, 25, 5, 9, 4]
Dataset_F = [25, 17, 6, 26, 38, 11, 2]
Dataset_G = [48, 29, 38, 30, 24, 16, 21]
Dataset_H = [56, 49, 58, 64, 146, 63, 13]
Dataset_I = [48, 38, 71, 83, 63, 59, 65]
Dataset_J = [88, 39, 48, 62, 95, 73, 27]
Dataset_K = [40, 62, 8, 53, 47, 40, 84]
Dataset_M = []
Dataset_N = []

calf_names = ['B','C','D','E','F','G','H','I','K']

Accum = [0,0,0,0,0,0,0]
for i in range(9): 
    dataset = [Dataset_B, Dataset_Plus, Dataset_D, Dataset_E, Dataset_F, Dataset_G, Dataset_H, Dataset_I, Dataset_K][i]
    color = ['bisque', 'palegoldenrod', 'burlywood','antiquewhite','tan','navajowhite','oldlace','beige','cornsilk'][i]
    plt.bar(modified_classes_official, dataset, bottom=Accum, color=color, label=calf_names[i])
    Accum = np.add(Accum, dataset)

for i in range(7): 
    plt.text(i, Accum[i] + 3, Accum[i], fontsize=11, horizontalalignment='center')
        
plt.ylim(0,800)
plt.legend()

plt.xlabel("Classes")
plt.ylabel("Video data")
plt.show()
