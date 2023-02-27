import cv2
import numpy as np
from numpy.linalg import svd
import random
import matlab.engine

def match_features(keypoint1,keypoint2,descriptor1,descriptor2):

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    match = flann.knnMatch(descriptor1, descriptor2, k=2)

    matches = []
    point1 = []
    point2 = []

    for m, n in match:
        point1.append(keypoint1[m.queryIdx].pt)
        point2.append(keypoint2[m.trainIdx].pt)
        matches.append(m)

    point1 = np.asarray(point1)
    point2 = np.asarray(point2)
    matches = np.asarray(matches)

    return point1, point2, matches

def decompose_essential_matrix(E):

    U, sdiag, V = svd(E)
    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])

    t = -np.dot(U, np.array([0, 0, 1]).T).reshape((3, 1))
    R = np.dot(np.dot(U, W.T), V)

    mat1 = np.hstack((np.identity(3), np.zeros((3, 1))))
    mat2 = np.hstack((R, t))

    return mat1, mat2

def generate_3D_points(mat1,mat2, point1, point2, K):
    projmat1 = np.dot(K, mat1)
    projmat2 = np.dot(K, mat2)

    matA = np.zeros((4, 4))
    point3d = np.zeros((3, point1.shape[0]))
    projmat = np.array([projmat1, projmat2])
    projpt = np.array(([point1, point2]))

    for i in range(point1.shape[0]):
        for j in range(2):

            x = projpt[j][i][0]
            y = projpt[j][i][1]

            for k in range(4):
                matA[j * 2 + 0][k] = x * projmat[j][2][k] - projmat[j][0][k]
                matA[j * 2 + 1][k] = y * projmat[j][2][k] - projmat[j][1][k]

        U, sdiag, V = svd(matA)

        point3d[0][i] = V[3][0] / V[3][3]
        point3d[1][i] = V[3][1] / V[3][3]
        point3d[2][i] = V[3][2] / V[3][3]

    point3d = point3d.T

    return point3d

def get_colors(img1, img2, point1, point2):
    color = []
    for i in range(point1.shape[0]):
        cx1 = int(point1[i][1])
        cx2 = int(point2[i][1])

        cy1 = int(point1[i][0])
        cy2 = int(point2[i][0])

        x1 = 0 if cx1 - 1 < 0 else cx1 - 1
        y1 = 0 if cy1 - 1 < 0 else cy1 - 1

        x2 = 0 if cx2 - 1 < 0 else cx2 - 1
        y2 = 0 if cy2 - 1 < 0 else cy2 - 1



        R = (np.mean(img1[x1: cx1 + 2, y1: cy1 + 2, 0]) + np.mean(img2[x2:cx2 + 2, y2:cy2 + 2, 0])) / 2
        G = (np.mean(img1[x1: cx1 + 2, y1: cy1 + 2, 1]) + np.mean(img2[x2:cx2 + 2, y2:cy2 + 2, 1])) / 2
        B = (np.mean(img1[x1: cx1 + 2, y1: cy1 + 2, 2]) + np.mean(img2[x2:cx2 + 2, y2:cy2 + 2, 2])) / 2
        color.append(np.hstack((R, G, B)))


    color = np.asarray(color, dtype=np.int64)

    return color



def five_points_with_ransac(point1,point2, t , K, p ,s):

    point1 = point1.T
    point2 = point2.T

    eng = matlab.engine.start_matlab()

    # You don't have to set the seed.
    random.seed(42)

    row = point1.shape[0]
    col = point1.shape[1]

    if row == 2:
        point1 = np.vstack((point1, np.ones((1, col))))
        point2 = np.vstack((point2, np.ones((1, col))))

    N = 1000
    count = 0
    bestscore = 0
    bestE = []
    bestInliers = []

    while N > count:
        idx = random.sample(range(0, col - 1), s)

        def convert_matlab(point1, point2):

            x1 = [list(point1[0]), list(point1[1]), list(point1[2])]
            x2 = [list(point2[0]),list(point2[1]), list(point2[2])]

            pt1 = matlab.double(x1)
            pt2 = matlab.double(x2)

            return pt1, pt2

        # Step4. Find essential matrix candidates using the given five-points algorithm.
        pt1, pt2 = convert_matlab(point1[:,idx], point2[:,idx])
        Evec = eng.calibrated_fivepoint(pt1,pt2)
        Evec = np.asarray(Evec)
        nsol = Evec.shape[1]
        Emat = Evec.reshape((3,3,nsol)).T

        if Emat.shape[0]==0:
            continue

        # Step5. Find the distance of each machined point using the essential matrix candidates calculated in step 4.
        def compute_distance(Emat,x1,x2, K):
            bestE = Emat[0].T
            bestInliers = []
            ninliers = 0

            for E in Emat:
                inv_K = np.linalg.inv(K)
                inv_KtE = np.dot(inv_K.T, E.T)
                F = np.dot(inv_KtE, inv_K)

                Fx1 = np.dot(F, x1)
                Ftx2 = np.dot(F.T, x2)
                x2tFx1 = sum(x2 * Fx1)
                d = (x2tFx1 ** 2) * ((1/((Fx1[0, :] ** 2) + (Fx1[1, :] ** 2)))+((1/(Ftx2[0, :] ** 2) + (Ftx2[1, :] ** 2))))

                n_d = (d - np.min(d)) / (np.max(d) - np.min(d))
                inliers = np.where(n_d < t)

                if len(inliers[0]) > ninliers:
                    ninliers = len(inliers[0])
                    bestE = E.T
                    bestInliers = inliers

            return bestInliers, bestE


        inliers, E = compute_distance(Emat, point1, point2, K)

        if len(inliers[0]) > bestscore:
            bestscore = len(inliers[0])
            bestInliers = inliers
            bestE = E

            outlier_ratio = len(inliers[0])/point1.shape[1]
            es = 1 - outlier_ratio**s
            N = np.log(1-p)/np.log(1-es)


        count += 1

    return bestInliers,bestE

