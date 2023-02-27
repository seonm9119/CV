from scipy.signal import convolve2d
import numpy as np
import cupy as cp
import cv2
from PIL import Image


def calculate_IOU(name, classfier, res, flag):


    res1 = cp.asnumpy(res)
    gtImg = cv2.imread(name)
    gtImg = gtImg[:, :, 2].flatten()


    iou_lables = []

    for i in classfier:
        if i!=0:
            if flag == False:
                gt_lable = np.where(gtImg == i-1, True, False)
                seg_lable = np.where(res1 == i-1, True, False)
            else:
                gt_lable = np.where(gtImg == i, True, False)
                seg_lable = np.where(res1 == i - 1, True, False)

            inter_lable = sum(np.logical_and(gt_lable, seg_lable))
            sum_lable = sum(gt_lable)
            iou_lables.append(inter_lable / sum_lable)


    m_iou = np.array(iou_lables).mean()

    return iou_lables, m_iou


def compute_statics(image, kernel_size):

    kernel = np.ones((kernel_size, kernel_size))
    kernel[int(kernel_size / 2), int(kernel_size / 2)] = 0

    neighbor_sum = convolve2d(image,
                              kernel,
                              mode='same',
                              boundary='fill',
                              fillvalue=0)

    num_neighbor = convolve2d(np.ones(image.shape),
                              kernel, mode='same',
                              boundary='fill',
                              fillvalue=0)

    neighbor_mean = neighbor_sum / num_neighbor
    neighbor_variation = (image - neighbor_mean) ** 2 / num_neighbor

    neighbor_variation[np.where(neighbor_variation == 0)] = 1e-1

    return neighbor_mean, neighbor_variation


def weight_identifier(name):
    globs = globals()
    return globs.get(name)


def original(neighbor_pixels, r, variation, mean):
    return abs(neighbor_pixels - r)


def option1(neighbor_pixels, r, variation, mean):
    tmp = (r - neighbor_pixels) ** 2
    tmp = -1*tmp / (2 * variation)
    return np.exp(tmp).round(6)


def save(res, row, col, flag, name):
    res1 = cp.asnumpy(res)
    res_img = res1.reshape((row, col))
    im = Image.fromarray(res_img.astype('uint8'), 'L')
    im.save( name +"res_img.png")

    def transform_img(flag):
        """flag is true : multi
           flag is false : binary"""

        if flag == True:
            seg_img = np.zeros((row,col,3))
            for x in range(row):
                for y in range(col):
                    seg_img[x][y][:]=switch(res_img[x][y])
            return seg_img
        else:
            seg_img = res_img.copy()
            seg_img *= 255
            return seg_img

    seg_img = transform_img(flag)
    im = Image.fromarray(seg_img.astype('uint8'))
    im.save(name + "seg_img.png")


def load_classifier(name):
    classifier = []
    class_name = []

    with open(name, 'r') as f:
        for line in f:
            tmp = line.strip().split()
            class_name.append(tmp[0])
            classifier.append(int(tmp[1].strip('[').strip(']').split(',')[0]))

    return classifier, class_name

def switch(label):

    color = {7: np.array([0, 0, 0]),
             0: np.array([171, 242, 0]),
             1: np.array([255, 255, 90]),
             2: np.array([1, 0, 255]),
             3: np.array([0, 216, 255]),
             4: np.array([255, 255, 255]),
             5: np.array([255, 0, 221]),
             6: np.array([255, 0, 0])}.get(label, np.array([0, 0, 0]))

    return color

