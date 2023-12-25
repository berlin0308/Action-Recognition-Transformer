import matplotlib.pyplot as plt
import numpy as np

# labels = ['AL','AS','DR','FD','NL','NS']
# sizes = [2.5,2.0,0.9,1.0, 15.5, 2.0]  

modified_classes_official = ['Non-active\nLying','Active\nLying','Non-active\nStanding','Active\nStanding','Feeding','Drinking']
sizes = [15.5, 2.5, 2.0, 2.0, 1.0, 0.9]  
color = ['#469597', '#5BA199', '#BBC6C8','#E5E3E4','#E0D5BE','#DDBEAA','#EEDFD6']


plt.pie(sizes, labels=modified_classes_official, autopct='%1.1f%%', startangle=90, colors=color)
plt.axis('equal') 
# plt.title('A day of a calf')

plt.show()
