import cv2
import numpy as np
from SavePLY import SavePLY
from functions import decompose_essential_matrix, generate_3D_points, get_colors, match_features, five_points_with_ransac

K = np.array([[1506.070, 0, 1021.043],
              [0, 1512.965, 698.031],
              [0, 0, 1]])


# Step1. Load the input images.
img1 = cv2.imread('datasets/dataset_two view/sfm01.jpg')
img2 = cv2.imread('datasets/dataset_two view/sfm02.jpg')


# Step2. Extract features from both images.
sift = cv2.xfeatures2d.SIFT_create()
keypoint1, descriptor1 = sift.detectAndCompute(img1, None)
keypoint2, descriptor2 = sift.detectAndCompute(img2, None)

kp_img1 = cv2.drawKeypoints(img1, keypoint1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
kp_img2 = cv2.drawKeypoints(img2, keypoint2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite("save/kp_img1.jpg", kp_img1)
cv2.imwrite("save/kp_img2.jpg", kp_img2)


# Step3. Match features between two images.
point1, point2, matches = match_features(keypoint1,keypoint2,descriptor1,descriptor2)
match_img = cv2.drawMatches(img1, keypoint1, img2, keypoint2, matches, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite("save/match_img.jpg", match_img)


#Step4-5. Estimate Essential matrix E with RANSAC (clibrated_fivepoint)
inlier,E = five_points_with_ransac(point1,point2, 1e-4, K, 0.99 ,5)
inlier_img = cv2.drawMatches(img1, keypoint1, img2, keypoint2, matches[inlier], None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite("save/inliner_img.jpg", inlier_img)


# Step6. Decompose essential matrix E to camera matrix.
mat1, mat2 = decompose_essential_matrix(E)


# Step7. Generate 3D points by implementing triangulation.
point3d = generate_3D_points(mat1,mat2, point1, point2, K)
point3d_inlier = generate_3D_points(mat1,mat2, point1[inlier], point2[inlier], K)


# Step8. Generate ply file.
color = get_colors(img1, img2, point1, point2)
color_inlier = get_colors(img1, img2, point1[inlier], point2[inlier])

X = np.hstack([point3d, color])
X_inlier = np.hstack([point3d_inlier, color_inlier])

SavePLY("save/sift_3d.ply", X)
SavePLY("save/inlier_3d.ply", X_inlier)
print("done")


