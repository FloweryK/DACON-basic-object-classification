import os
import cv2
import numpy as np
from config import CLASSES
from utils import save, load

def save_total_data(data_dir):
    total_data = []

    for class_name in CLASSES.keys():
        class_dir = os.path.join(data_dir, class_name)

        for file_name in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file_name)
            print(file_path)

            img = cv2.imread(file_path)
            total_data.append(img)
    
    save(total_data, "total_data.list")
    

# save_total_data(os.path.join('data', 'train'))
total_data = load("total_data.list")
total_data = np.array(total_data)

# calculate statistics
print(total_data.shape)                     # (50000, 32, 32, 3)
print(np.mean(total_data, axis=(0, 1, 2)))  # [113.86283869 122.93301916 125.26884795]
print(np.std(total_data, axis=(0, 1, 2)))   # [66.46033586 61.90929531 62.5340599 ]