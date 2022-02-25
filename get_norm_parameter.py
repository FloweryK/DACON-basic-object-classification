import os
import cv2
import numpy as np
from config import CLASSES


def calculate_mean_var(data_dir):
    total_data = []

    for class_name in CLASSES.keys():
        class_dir = os.path.join(data_dir, class_name)

        for file_name in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file_name)
            print(file_path)

            img = cv2.imread(file_path)
            total_data.append(img)
    
    
    # calculate statistics
    total_data = np.array(total_data)
    print(total_data.shape)                     # (50000, 32, 32, 3)
    print(np.mean(total_data, axis=(0, 1, 2)))  # [113.86283869 122.93301916 125.26884795]
    print(np.std(total_data, axis=(0, 1, 2)))   # [66.46033586 61.90929531 62.5340599 ]


if __name__ == "__main__":
    dir = os.path.join("data", "train")
    calculate_mean_var(dir)