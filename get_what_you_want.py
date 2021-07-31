import cv2 as cv
import numpy as np
from scipy import io
import math

P_L = np.array([[10, 0, 90, 80, 70, 60, 50, 40],
                [100, 100, 200, 300, 400, 200, 250, 300],
                [700, 730, 720, 710, 750, 760, 780, 700]])

KKK = io.loadmat(
    '/home/yanze/Desktop/Image-matching-and-ranging-based-on-binocular-stereo-vision-master/source code/Calib_Results_stereo.mat')
R = KKK['R']
T = KKK['T']

KK_left = KKK['KK_left']
KK_right = KKK['KK_right']

P_R = np.dot(R, P_L) + T
LEFT_camera = np.dot(KK_left, P_L) / P_L[2, :]
pixel_left = LEFT_camera[0:2, :]
pixel_left.dtype = 'float64'
pixel_left = pixel_left.T
RIGHT_camera = np.dot(KK_right, P_R) / P_R[2, :]
pixel_right = RIGHT_camera[0: 2, :]
pixel_right.dtype = 'float64'
pixel_right = pixel_right.T
Pose1 = [10., 10., 10., 5., 1, 5., 7.]
Px = Pose1[0]
Py = Pose1[1]
Pz = Pose1[2]
t_true = np.array([[Px], [Py], [Pz]])
rota = Pose1[3] * math.pi / 180.
rotb = Pose1[4] * math.pi / 180.
rotc = Pose1[5] * math.pi / 180.
Rx = np.array([[1, 0, 0], [0, math.cos(rotc), -math.sin(rotc)], [0, math.sin(rotc), math.cos(rotc)]])
Ry = np.array([[math.cos(rotb), 0, math.sin(rotb)], [0, 1, 0], [-math.sin(rotb), 0, math.cos(rotb)]])
Rz = np.array([[math.cos(rota), -math.sin(rota), 0], [math.sin(rota), math.cos(rota), 0], [0, 0, 1]])
R_true = np.dot(np.dot(Rz, Ry), Rx)
print(R_true)
# T_true = np.array([[R_true, t_true], [0, 0, 0, 1]])

P_L2 = np.dot(R_true, P_L) + t_true
P_R2 = np.dot(R, P_L2) + T
LEFT_camera2 = np.dot(KK_left, P_L2) / P_L2[2, :]
RIGHT_camera2 = np.dot(KK_right, P_R2) / P_R2[2, :]
pixel_left2 = LEFT_camera2[0:2, :]
pixel_left2.dtype = 'float64'
pixel_left2 = pixel_left2.T
pixel_right2 = RIGHT_camera2[0:2, :]
pixel_right2.dtype = 'float64'
pixel_right2 = pixel_right2.T
KK_left.dtype = 'float64'
pts1 = np.array(
    [[2827.40125706007, 2648.21992634424], [3225.82870222552, 3864.79525915776], [3841.3974357474, 1445.2378066086],
     [4559.45049270162, 3925.27562650571], [5115.82228040332, 2446.34926247009], [4559.45049270162, 3925.27562650571]])
pts2 = np.array(
    [[3430.57566731374, 2802.04116751756], [3835.39250729576, 4012.73987030132], [4460.72414071088, 1613.14709322255],
     [5184.72527459039, 4076.97090464259], [5746.37631816432, 2614.77259877965], [5184.72527459039, 4076.97090464259]])
print(type(pts2))
print(type(pixel_left))
E, MASK = cv.findEssentialMat(pixel_left, pixel_left2, KK_left, method=cv.LMEDS, threshold=1)
points, R1, t1, mas = cv.recoverPose(E, pixel_left, pixel_left2, KK_left)
print(type(R1))
P_L3 = np.dot(R1, P_L) + t1
ttt = P_L3 - P_L2


