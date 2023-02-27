import cv2
from functions import calculate_IOU, save, load_classifier
import numpy as np
from least_square import calculate_least_square, calculate_neighborhood_weight
import pandas as pd

"""load data"""

#flag is True : multi-layer
#flag is False : Binary-Layer
#classifier : the number of classes

grayImg = cv2.imread('datasets/Emily-In-Paris-gray.png', 0).astype(np.float64)
scribbles = cv2.imread('datasets/Emily-In-Paris-scribbles.png').astype(np.float64)
classifier, class_name = load_classifier('datasets/Emily-In-Paris_class_names.txt')

img_row, img_col = grayImg.shape
size = img_row*img_col
kernel_size = 3
save_name = "original_multi_kernel-3_"
flag = False


"""Task1: Calculate neighborhood weights"""
"""weight_function : original, option1 """
weight_mat, lables = calculate_neighborhood_weight('original', grayImg, scribbles, kernel_size, classifier)


"""Task2: Calculate least square solution"""
res = calculate_least_square(weight_mat, lables)
save(res, img_row, img_col,flag, save_name)


"""Task3: Calculate IOU"""
iou, m_iou = calculate_IOU('datasets/Emily-In-Paris-gt.png', classifier, res, flag)

dic = {'class' : class_name, 'IOU' : iou, 'mIOU' : m_iou}
df = pd.DataFrame(dic)
df.to_csv(save_name+"IOU.csv")
print(df)
print("m_lou : ", m_iou)


print("done")

