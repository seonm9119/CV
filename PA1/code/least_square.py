import numpy as np
import cupy as cp
from cupyx.scipy.sparse.linalg import lsqr
from cupyx.scipy.sparse import csc_matrix
from cupyx.scipy.sparse import eye
import functions as f

def calculate_neighborhood_weight(name, image, scribbles, kernel_size, classifier):
    weight_function = f.weight_identifier(name)
    img_f = image.flatten()
    img_row, img_col = image.shape
    size = img_row * img_col

    lables = []
    for i in classifier:
        if i!=0:
            lables.append(np.where(scribbles[:, :, 2] == i, 1, 0).flatten())

    
    lables = cp.array(lables, dtype=cp.float64)


    ii = int(kernel_size / 2)
    idces = np.arange(0, size).reshape((img_row, img_col))

    """compute neighborhood means, neighborhood variation"""
    neighbor_mean, neighbor_variation = f.compute_statics(image, kernel_size)

    k = 0;
    row = [];
    col = [];
    data = []
    flag = np.sum(lables, axis=0)


    cnt = 0
    for x in range(img_row):
        for y in range(img_col):

            if flag[k] == 0:
                cnt += 1
                idx = idces[0 if x - ii < 0 else x - ii:x + ii + 1,
                      0 if y - ii < 0 else y - ii:y + ii + 1].flatten().tolist()
                idx.remove(k)

                """compute weights"""
                weight = weight_function(img_f[idx], image[x][y], neighbor_variation[x][y], neighbor_mean[x][y])
                weight[np.where(weight == 0)] = 1e-1

                sum_weight = sum(weight)
                if sum_weight == 0:
                    sum_weight = 1e-1

                weight = weight / sum_weight


                row += [k for _ in idx]
                col += idx
                data += weight.flatten().tolist()

            k += 1


    data = cp.array(data)
    row = cp.array(row)
    col = cp.array(col)
    weight_mat = csc_matrix((data, (row, col)), shape=(size, size))

    return eye(size) - weight_mat, lables

def calculate_least_square(weight_mat, lables):

    cost_mat = weight_mat
    lables = cp.copy(lables)


    res = []
    for i in range(lables.shape[0]):
        res.append(lsqr(cost_mat, lables[i])[0])

    res = cp.array(res, dtype=cp.float64)
    res = cp.argmax(res, axis=0)

    return res

