# from pathlib import Path
# from pystripe.core import imread_tif_raw, imsave_tif
# from process_images import align_images
# from numpy import zeros
# from skimage.transform import resize
# color1 = Path(r"Y:\3D_stitched\20220331_SW220203_04_LS_6x_1000z\Ex_561_Em_600_tif")
# color2 = Path(r"Y:\3D_stitched\20220331_SW220203_04_LS_6x_1000z\Ex_642_Em_680_tif")
# file = "img_003000.tif"
#
# img1 = imread_tif_raw(color1/file)
# img2 = imread_tif_raw(color2/file)
# img1 = resize(img1, img2.shape, preserve_range=True, anti_aliasing=True)
# img1 = align_images(img1, img2)
# shape = img1.shape  # (max(img1.shape[0], img2.shape[0]), max(img1.shape[1], img2.shape[1]))
#
# multi_channel_img = zeros((shape[0], shape[1], 3), dtype=img2.dtype)
# multi_channel_img[:, :, 0] = img1
# multi_channel_img[:, :, 1] = img2

# images = {"g": img1, "r": img2}
# for idx, color in enumerate("rgb"):  # the order of letters should be "rgb" here
#     image = images.get(color, None)
#     if image is not None:
#         image_shape = image.shape
#         if image_shape != shape:
#             if image_shape[0] <= shape[0] or image_shape[1] <= shape[1]:
#                 padded_image = zeros(shape, dtype=image.dtype)
#                 padded_image[:image_shape[0], :image_shape[1]] = image
#                 image = padded_image
#             else:
#                 image = image[:shape[0], :shape[1]]
#         multi_channel_img[:, :, idx] = image

# imsave_tif(Path(r"D:\merged_destripe_sobel_destripe.tif"), multi_channel_img, compression=("ZLIB", 1))

# for file1 in color1.iterdir():
#     file2 = color2 / file1.name
#     if file1.is_file() and file2.exists() and file2.is_file():
#         img1 = imread_tif_raw(file1)
#         img2 = imread_tif_raw(file2)
#         img_aligned = align_images(img2, img1)


# # !/usr/bin/python
# # -*- coding: utf-8 -*-
# ''' Phase Correlation based image matching and registration libraries
# '''
# __author__ = "Yoshi Ri"
# __copyright__ = "Copyright 2017, The University of Tokyo"
# __credits__ = ["Yoshi Ri"]
# __license__ = "BSD"
# __version__ = "1.0.1"
# __maintainer__ = "Yoshi Ri"
# __email__ = "yoshiyoshidetteiu@gmail.com"
# __status__ = "Production"
#
# # Phase Correlation to Estimate Pose
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt  # matplotlibの描画系
# import math
# import sys
#
#
# class imregpoc:
#     def __init__(self, iref, icmp, *, threshold=0.06, alpha=0.5, beta=0.8, fitting='WeightedCOG'):
#         self.orig_ref = iref.astype(np.float32)
#         self.orig_cmp = icmp.astype(np.float32)
#         self.th = threshold
#         self.orig_center = np.array(self.orig_ref.shape) / 2.0
#         self.alpha = alpha
#         self.beta = beta
#         self.fitting = fitting
#
#         self.param = [0, 0, 0, 1]
#         self.peak = 0
#         self.affine = np.float32([1, 0, 0, 0, 1, 0]).reshape(2, 3)
#         self.perspective = np.float32([1, 0, 0, 0, 1, 0, 0, 0, 0]).reshape(3, 3)
#
#         # set ref, cmp, center
#         self.fft_padding()
#         self.match()
#
#     def define_fftsize(self):
#         refshape = self.orig_ref.shape
#         cmpshape = self.orig_cmp.shape
#         if not refshape == cmpshape:
#             print("The size of two input images are not equal! Estimation could be inaccurate.")
#         maxsize = max(max(refshape), max(cmpshape))
#         # we can use faster fft window size with scipy.fftpack.next_fast_len
#         return maxsize
#
#     def padding_image(self, img, imsize):
#         pad_img = np.pad(img, [(0, imsize[0] - img.shape[0]), (0, imsize[1] - img.shape[1])], 'constant')
#         return pad_img
#
#     def fft_padding(self):
#         maxsize = self.define_fftsize()
#         self.ref = self.padding_image(self.orig_ref, [maxsize, maxsize])
#         self.cmp = self.padding_image(self.orig_cmp, [maxsize, maxsize])
#         self.center = np.array(self.ref.shape) / 2.0
#
#     def fix_params(self):
#         # If you padded to right and lower, perspective is the same with original image
#         self.param = self.warp2poc(perspective=self.perspective, center=self.orig_center)
#
#     def match(self):
#         height, width = self.ref.shape
#         self.hanw = cv2.createHanningWindow((width, height), cv2.CV_64F)
#
#         # Windowing and FFT
#         G_a = np.fft.fft2(self.ref * self.hanw)
#         G_b = np.fft.fft2(self.cmp * self.hanw)
#
#         # 1.1: Frequency Whitening
#         self.LA = np.fft.fftshift(np.log(np.absolute(G_a) + 1))
#         self.LB = np.fft.fftshift(np.log(np.absolute(G_b) + 1))
#         # 1.2: Log polar Transformation
#         cx = self.center[1]
#         cy = self.center[0]
#         self.Mag = width / math.log(width)
#         self.LPA = cv2.logPolar(self.LA, (cy, cx), self.Mag, flags=cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
#         self.LPB = cv2.logPolar(self.LB, (cy, cx), self.Mag, flags=cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
#
#         # 1.3:filtering
#         LPmin = math.floor(self.Mag * math.log(self.alpha * width / 2.0 / math.pi))
#         LPmax = min(width, math.floor(self.Mag * math.log(width * self.beta / 2)))
#         assert LPmax > LPmin, 'Invalid condition!\n Enlarge lpmax tuning parameter or lpmin_tuning parameter'
#         Tile = np.repeat([0.0, 1.0, 0.0], [LPmin - 1, LPmax - LPmin + 1, width - LPmax])
#         self.Mask = np.tile(Tile, [height, 1])
#         self.LPA_filt = self.LPA * self.Mask
#         self.LPB_filt = self.LPB * self.Mask
#
#         # 1.4: Phase Correlate to Get Rotation and Scaling
#         Diff, peak, self.r_rotatescale = self.PhaseCorrelation(self.LPA_filt, self.LPB_filt)
#         theta1 = 2 * math.pi * Diff[1] / height;  # deg
#         theta2 = theta1 + math.pi;  # deg theta ambiguity
#         invscale = math.exp(Diff[0] / self.Mag)
#         # 2.1: Correct rotation and scaling
#         b1 = self.Warp_4dof(self.cmp, [0, 0, theta1, invscale])
#         b2 = self.Warp_4dof(self.cmp, [0, 0, theta2, invscale])
#
#         # 2.2 : Translation estimation
#         diff1, peak1, self.r1 = self.PhaseCorrelation(self.ref, b1)  # diff1, peak1 = PhaseCorrelation(a,b1)
#         diff2, peak2, self.r2 = self.PhaseCorrelation(self.ref, b2)  # diff2, peak2 = PhaseCorrelation(a,b2)
#         # Use cv2.phaseCorrelate(a,b1) because it is much faster
#
#         # 2.3: Compare peaks and choose true rotational error
#         if peak1 > peak2:
#             Trans = diff1
#             peak = peak1
#             theta = -theta1
#         else:
#             Trans = diff2
#             peak = peak2
#             theta = -theta2
#
#         if theta > math.pi:
#             theta -= math.pi * 2
#         elif theta < -math.pi:
#             theta += math.pi * 2
#
#         # Results
#         self.param = [Trans[0], Trans[1], theta, 1 / invscale]
#         self.peak = peak
#         self.perspective = self.poc2warp(self.center, self.param)
#         self.affine = self.perspective[0:2, :]
#         self.fix_params()
#
#     def match_new(self, newImg):
#         self.cmp_orig = newImg
#         self.fft_padding()
#         height, width = self.cmp.shape
#         cy, cx = height / 2, width / 2
#         G_b = np.fft.fft2(self.cmp * self.hanw)
#         self.LB = np.fft.fftshift(np.log(np.absolute(G_b) + 1))
#         self.LPB = cv2.logPolar(self.LB, (cy, cx), self.Mag, flags=cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
#         self.LPB_filt = self.LPB * self.Mask
#         # 1.4: Phase Correlate to Get Rotation and Scaling
#         Diff, peak, self.r_rotatescale = self.PhaseCorrelation(self.LPA_filt, self.LPB_filt)
#         theta1 = 2 * math.pi * Diff[1] / height;  # deg
#         theta2 = theta1 + math.pi;  # deg theta ambiguity
#         invscale = math.exp(Diff[0] / self.Mag)
#         # 2.1: Correct rotation and scaling
#         b1 = self.Warp_4dof(self.cmp, [0, 0, theta1, invscale])
#         b2 = self.Warp_4dof(self.cmp, [0, 0, theta2, invscale])
#
#         # 2.2 : Translation estimation
#         diff1, peak1, self.r1 = self.PhaseCorrelation(self.ref, b1)  # diff1, peak1 = PhaseCorrelation(a,b1)
#         diff2, peak2, self.r2 = self.PhaseCorrelation(self.ref, b2)  # diff2, peak2 = PhaseCorrelation(a,b2)
#         # Use cv2.phaseCorrelate(a,b1) because it is much faster
#
#         # 2.3: Compare peaks and choose true rotational error
#         if peak1 > peak2:
#             Trans = diff1
#             peak = peak1
#             theta = -theta1
#         else:
#             Trans = diff2
#             peak = peak2
#             theta = -theta2
#
#         if theta > math.pi:
#             theta -= math.pi * 2
#         elif theta < -math.pi:
#             theta += math.pi * 2
#
#         # Results
#         self.param = [Trans[0], Trans[1], theta, 1 / invscale]
#         self.peak = peak
#         self.perspective = self.poc2warp(self.center, self.param)
#         self.affine = self.perspective[0:2, :]
#         self.fix_params()
#
#     def poc2warp(self, center, param):
#         cx, cy = center
#         dx, dy, theta, scale = param
#         cs = scale * math.cos(theta)
#         sn = scale * math.sin(theta)
#
#         Rot = np.float32([[cs, sn, 0], [-sn, cs, 0], [0, 0, 1]])
#         center_Trans = np.float32([[1, 0, cx], [0, 1, cy], [0, 0, 1]])
#         center_iTrans = np.float32([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
#         cRot = np.dot(np.dot(center_Trans, Rot), center_iTrans)
#         Trans = np.float32([[1, 0, dx], [0, 1, dy], [0, 0, 1]])
#         Affine = np.dot(cRot, Trans)
#         return Affine
#
#     def warp2poc(self, center, perspective):
#         cx, cy = center
#         Center = np.float32([[1, 0, cx], [0, 1, cy], [0, 0, 1]])
#         iCenter = np.float32([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
#
#         pocmatrix = np.dot(np.dot(iCenter, perspective), Center)
#         dxy = np.dot(np.linalg.inv(pocmatrix[0:2, 0:2]), pocmatrix[0:2, 2])
#         scale = np.sqrt(pocmatrix[0, 0] ** 2 + pocmatrix[0, 1] ** 2)
#         theta = np.arctan2(pocmatrix[0, 1], pocmatrix[0, 0])
#         return [dxy[0], dxy[1], theta, scale]
#
#     # Warp Image based on poc parameter
#     def Warp_4dof(self, Img, param):
#         center = np.array(Img.shape) / 2
#         rows, cols = Img.shape
#         Affine = self.poc2warp(center, param)
#         outImg = cv2.warpPerspective(Img, Affine, (cols, rows), cv2.INTER_LINEAR)
#         return outImg
#
#     def SubpixFitting(self, mat):
#         if self.fitting == 'COG':
#             TY, TX = self.CenterOfGravity(mat)
#         elif self.fitting == 'WeightedCOG':
#             TY, TX = self.WeightedCOG(mat)
#         elif self.fitting == 'Parabola':
#             TY, TX = self.Parabola(mat)
#         else:
#             print("Undefined subpixel fitting method! Use weighted center of gravity method instead.")
#             TY, TX = self.WeightedCOG(mat)
#
#         return [TY, TX]
#
#     # Get peak point
#     def CenterOfGravity(self, mat):
#         hei, wid = mat.shape
#         if hei != wid:  # if mat size is not square, there must be something wrong
#             print("Skip subpixel estimation!")
#             return [0, 0]
#         Tile = np.arange(wid, dtype=float) - (wid - 1.0) / 2.0
#         Tx = np.tile(Tile, [hei, 1])  # Ty = Tx.T
#         Sum = np.sum(mat)
#         # print(mat)
#         Ax = np.sum(mat * Tx) / Sum
#         Ay = np.sum(mat * Tx.T) / Sum
#         return [Ay, Ax]
#
#     # Weighted Center Of Gravity
#     def WeightedCOG(self, mat):
#         if mat.size == 0:
#             print("Skip subpixel estimation!")
#             Res = [0, 0]
#         else:
#             peak = mat.max()
#             newmat = mat * (mat > peak / 10)  # discard information of lower peak
#             Res = self.CenterOfGravity(newmat)
#         return Res
#
#     # Parabola subpixel fitting
#     def Parabola(self, mat):
#         hei, wid = mat.shape
#         boxsize = 3
#         cy = int((hei - 1) / 2)
#         cx = int((wid - 1) / 2)
#         bs = int((boxsize - 1) / 2)
#         box = mat[cy - bs:cy - bs + boxsize, cx - bs:cx - bs + boxsize]
#         # [x^2 y ^2 x y 1]
#         Tile = np.arange(boxsize, dtype=float) - bs
#         Tx = np.tile(Tile, [boxsize, 1])
#         Ty = Tx.T
#         Ones = np.ones((boxsize * boxsize, 1), dtype=float)
#         x = Tx.reshape(boxsize * boxsize, 1)
#         y = Ty.reshape(boxsize * boxsize, 1)
#         x2 = x * x
#         y2 = y * y
#         A = np.concatenate((x2, y2, x, y, Ones), 1)
#         # data = A^+ B
#         data = np.dot(np.linalg.pinv(A), box.reshape(boxsize * boxsize, 1))
#         # xmax = -c/2a, ymax = -d/2b, peak = e - c^2/4a - d^2/4b
#         a, b, c, d, e = data.squeeze()
#         Ay = -d / 2.0 / b
#         Ax = -c / 2.0 / a
#         self.peak = e - c * c / 4.0 / a - d * d / 4.0 / b
#         return [Ay, Ax]
#
#     # Phase Correlation
#     def PhaseCorrelation(self, a, b):
#         height, width = a.shape
#         # dt = a.dtype # data type
#         # Windowing
#
#         # FFT
#         G_a = np.fft.fft2(a * self.hanw)
#         G_b = np.fft.fft2(b * self.hanw)
#         conj_b = np.ma.conjugate(G_b)
#         R = G_a * conj_b
#         R /= np.absolute(R)
#         r = np.fft.fftshift(np.fft.ifft2(R).real)
#         # Get result and Interpolation
#         DY, DX = np.unravel_index(r.argmax(), r.shape)
#         # Subpixel Accuracy
#         boxsize = 5
#         box = r[DY - int((boxsize - 1) / 2):DY + int((boxsize - 1) / 2) + 1,
#               DX - int((boxsize - 1) / 2):DX + int((boxsize - 1) / 2) + 1]  # x times x box
#         # subpix fitting
#         self.peak = r[DY, DX]
#         TY, TX = self.SubpixFitting(box)
#         sDY = TY + DY
#         sDX = TX + DX
#         # Show the result
#         return [math.floor(width / 2) - sDX, math.floor(height / 2) - sDY], self.peak, r
#
#     def MoveCenter(self, Affine, center, newcenter):
#         dx = newcenter[1] - center[1]
#         dy = newcenter[0] - center[0]
#         center_Trans = np.float32([[1, 0, dx], [0, 1, dy], [0, 0, 1]])
#         center_iTrans = np.float32([[1, 0, -dx], [0, 1, -dy], [0, 0, 1]])
#         newAffine = center_iTrans.dot(Affine.dot(center_Trans))
#         return newAffine
#
#     def getParam(self):
#         return self.param
#
#     def getPeak(self):
#         return self.peak
#
#     def getAffine(self):
#         return self.affine
#
#     def getPerspective(self):
#         return self.perspective
#
#     def showRotatePeak(self):
#         plt.imshow(self.r_rotatescale, vmin=self.r_rotatescale.min(), vmax=self.r_rotatescale.max(), cmap='gray')
#         plt.show()
#
#     def showTranslationPeak(self):
#         plt.subplot(211)
#         plt.imshow(self.r1, vmin=self.r1.min(), vmax=self.r1.max(), cmap='gray')
#         plt.subplot(212)
#         plt.imshow(self.r2, vmin=self.r2.min(), vmax=self.r2.max(), cmap='gray')
#         plt.show()
#
#     def showLPA(self):
#         plt.imshow(self.LPA, vmin=self.LPA.min(), vmax=self.LPA.max(), cmap='gray')
#         plt.show()
#
#     def showLPB(self):
#         plt.imshow(self.LPB, vmin=self.LPB.min(), vmax=self.LPB.max(), cmap='gray')
#         plt.show()
#
#     def showMAT(self, MAT):
#         plt.figure()
#         plt.imshow(MAT, vmin=MAT.min(), vmax=MAT.max(), cmap='gray')
#         plt.show()
#
#     def saveMat(self, MAT, name):
#         cv2.imwrite(name, cv2.normalize(MAT, MAT, 0, 255, cv2.NORM_MINMAX))
#
#     def isSucceed(self):
#         if self.peak > self.th:
#             return 1
#         return 0
#
#     # function around mosaicing
#     def convertRectangle(self, perspective=None):
#         if perspective == None:
#             perspective = self.perspective
#         height, width = self.orig_cmp.shape
#         rectangles = np.float32([[0, 0, 0, width - 1, height - 1, 0, height - 1, width - 1]]).reshape(1, 4, 2)
#         converted_rectangle = cv2.perspectiveTransform(rectangles, np.linalg.inv(perspective))
#         xmax = math.ceil(converted_rectangle[0, :, 0].max())
#         xmin = math.floor(converted_rectangle[0, :, 0].min())
#         ymax = math.ceil(converted_rectangle[0, :, 1].max())
#         ymin = math.floor(converted_rectangle[0, :, 1].min())
#         return [xmin, ymin, xmax, ymax]
#
#     def stitching(self, perspective=None):
#         if perspective == None:
#             perspective = self.perspective
#         xmin, ymin, xmax, ymax = self.convertRectangle()
#         hei, wid = self.orig_ref.shape
#         sxmax = max(xmax, wid - 1)
#         sxmin = min(xmin, 0)
#         symax = max(ymax, hei - 1)
#         symin = min(ymin, 0)
#         swidth, sheight = sxmax - sxmin + 1, symax - symin + 1
#         xtrans, ytrans = 0 - sxmin, 0 - symin
#         Trans = np.float32([1, 0, xtrans, 0, 1, ytrans, 0, 0, 1]).reshape(3, 3)
#         newTrans = np.dot(Trans, np.linalg.inv(perspective))
#         warpedimage = cv2.warpPerspective(self.orig_cmp, newTrans, (swidth, sheight),
#                                           flags=cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
#         warpedimage[ytrans:ytrans + hei, xtrans:xtrans + wid] = self.orig_ref
#         plt.figure()
#         plt.imshow(warpedimage, vmin=warpedimage.min(), vmax=warpedimage.max(), cmap='gray')
#         plt.show()
#
#
# class TempMatcher:
#     def __init__(self, temp, descriptor='ORB'):
#
#         # switch detector and matcher
#         self.detector = self.get_des(descriptor)
#         self.bf = self.get_matcher(descriptor)  # self matcher
#
#         if self.detector == 0:
#             print("Unknown Descriptor! \n")
#             sys.exit()
#
#         if len(temp.shape) > 2:  # if color then convert BGR to GRAY
#             temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
#
#         self.template = temp
#         # self.imsize = np.shape(self.template)
#         self.kp1, self.des1 = self.detector.detectAndCompute(self.template, None)
#         self.kpb, self.desb = self.kp1, self.des1
#         self.flag = 0  # homography estimated flag
#         self.scalebuf = []
#         self.scale = 0
#         self.H = np.eye(3, dtype=np.float32)
#         self.dH1 = np.eye(3, dtype=np.float32)
#         self.dH2 = np.eye(3, dtype=np.float32)
#         self.matches = []
#         self.inliers = []
#         self.center = np.float32([temp.shape[1], temp.shape[0]]).reshape([1, 2]) / 2
#
#     def get_des(self, name):
#         return {
#             'ORB': cv2.ORB_create(nfeatures=500, scoreType=cv2.ORB_HARRIS_SCORE),
#             'AKAZE': cv2.AKAZE_create(),
#             'KAZE': cv2.KAZE_create(),
#             'SIFT': cv2.xfeatures2d.SIFT_create(),
#             'SURF': cv2.xfeatures2d.SURF_create()
#         }.get(name, 0)
#
#     def get_matcher(self, name):  # Binary feature or not
#         return {  # Knnmatch do not need crossCheck
#             'ORB': cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False),
#             'AKAZE': cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False),
#             'KAZE': cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False),
#             'SIFT': cv2.BFMatcher(),
#             'SURF': cv2.BFMatcher()
#         }.get(name, 0)
#
#     '''
#     Do matching based on the descriptor choosed in the constructor.
#     Input 1. Compared Image
#     Input 2. showflag for matched image
#     '''
#
#     def match(self, img, showflag=0):
#         if len(img.shape) > 2:  # if color then convert BGR to GRAY
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#         self.cmp = img
#         kp2, des2 = self.detector.detectAndCompute(img, None)
#         print('Matched Points Number:' + str(len(kp2)))
#         if len(kp2) < 5:
#             return [0, 0, 0, 1], 0, 0
#
#         matches = self.bf.knnMatch(self.des1, des2, k=2)
#         good = []
#         pts1 = []
#         pts2 = []
#
#         count = 0
#         for m, n in matches:
#             if m.distance < 0.5 * n.distance:
#                 good.append([m])
#                 pts2.append(kp2[m.trainIdx].pt)
#                 pts1.append(self.kp1[m.queryIdx].pt)
#                 count += 1
#
#         pts1 = np.float32(pts1)
#         pts2 = np.float32(pts2)
#
#         self.flag = 0
#         self.show = img
#         self.matches.append(count)  # ?
#         self.inliner = 0
#
#         if count > 4:
#             self.H, self.mask = cv2.findHomography(pts1 - self.center, pts2 - self.center, cv2.RANSAC, 3.0)
#             self.inliner = np.count_nonzero(self.mask)
#
#         if showflag:
#             img3 = cv2.drawMatchesKnn(self.template, self.kp1, img, kp2, good, None, flags=2)
#             plt.imshow(img3, cmap='gray')
#
#         param = self.getpoc()
#         return param, count, self.inliner
#
#     def getPerspective(self):
#         hei, wid = self.template.shape
#         cy, cx = hei / 2, wid / 2
#         Trans = np.float32([1, 0, cx, 0, 1, cy, 0, 0, 1]).reshape(3, 3)
#         iTrans = np.float32([1, 0, -cx, 0, 1, -cy, 0, 0, 1]).reshape(3, 3)
#         return np.dot(Trans, np.dot(self.H, iTrans))
#
#     def getpoc(self):
#         h, w = self.template.shape
#         # Affine = MoveCenterOfImage(self.H,[0,0],[w/2,h/2])
#         Affine = self.H
#
#         if Affine is None:
#             return [0, 0, 0, 1]
#
#         # Extraction
#         A2 = Affine * Affine
#         scale = math.sqrt(np.sum(A2[0:2, 0:2]) / 2.0)
#         theta = math.atan2(Affine[0, 1], Affine[0, 0])
#
#         theta = theta * 180.0 / math.pi
#
#         Trans = np.dot(np.linalg.inv(Affine[0:2, 0:2]), Affine[0:2, 2:3])
#         return [Trans[0], Trans[1], theta, scale]
#
#     def convertRectangle(self, perspective=None):
#         if perspective == None:
#             perspective = self.H
#         height, width = self.cmp.shape
#         rectangles = np.float32([[0, 0, 0, width - 1, height - 1, 0, height - 1, width - 1]]).reshape(1, 4, 2)
#         converted_rectangle = cv2.perspectiveTransform(rectangles, np.linalg.inv(perspective))
#         xmax = math.ceil(converted_rectangle[0, :, 0].max())
#         xmin = math.floor(converted_rectangle[0, :, 0].min())
#         ymax = math.ceil(converted_rectangle[0, :, 1].max())
#         ymin = math.floor(converted_rectangle[0, :, 1].min())
#         return [xmin, ymin, xmax, ymax]
#
#     def warp(self, perspective=None):
#         if perspective == None:
#             perspective = self.getPerspective()
#         xmin, ymin, xmax, ymax = self.convertRectangle(self.getPerspective())
#         hei, wid = self.template.shape
#         sxmax = max(xmax, wid - 1)
#         sxmin = min(xmin, 0)
#         symax = max(ymax, hei - 1)
#         symin = min(ymin, 0)
#         swidth, sheight = sxmax - sxmin + 1, symax - symin + 1
#         xtrans, ytrans = 0 - sxmin, 0 - symin
#         Trans = np.float32([1, 0, xtrans, 0, 1, ytrans, 0, 0, 1]).reshape(3, 3)
#         newTrans = np.dot(Trans, np.linalg.inv(perspective))
#         warpedimage = cv2.warpPerspective(self.cmp, newTrans, (swidth, sheight),
#                                           flags=cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
#         warpedimage[ytrans:ytrans + hei, xtrans:xtrans + wid] = self.template
#         return warpedimage
#
#     def stitching(self, perspective=None):
#         if perspective == None:
#             perspective = self.getPerspective()
#         xmin, ymin, xmax, ymax = self.convertRectangle(self.getPerspective())
#         hei, wid = self.template.shape
#         sxmax = max(xmax, wid - 1)
#         sxmin = min(xmin, 0)
#         symax = max(ymax, hei - 1)
#         symin = min(ymin, 0)
#         swidth, sheight = sxmax - sxmin + 1, symax - symin + 1
#         xtrans, ytrans = 0 - sxmin, 0 - symin
#         Trans = np.float32([1, 0, xtrans, 0, 1, ytrans, 0, 0, 1]).reshape(3, 3)
#         newTrans = np.dot(Trans, np.linalg.inv(perspective))
#         warpedimage = cv2.warpPerspective(self.cmp, newTrans, (swidth, sheight),
#                                           flags=cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
#         warpedimage[ytrans:ytrans + hei, xtrans:xtrans + wid] = self.template
#
#         plt.figure()
#         plt.imshow(warpedimage, vmin=warpedimage.min(), vmax=warpedimage.max(), cmap='gray')
#         plt.show()
#
#
# if __name__ == "__main__":
#     # Read image
#     ref = cv2.imread('imgs/ref.png', 0)
#     cmp = cv2.imread('imgs/cmp.png', 0)
#     plt.imshow(ref, cmap="gray")
#
#     # reference parameter (you can change this)
#     match = imregpoc(ref, cmp)
#     print(match.peak, match.param)
#     match_para = imregpoc(ref, cmp, fitting='Parabola')
#     print(match_para.peak, match_para.param)
#     match_cog = imregpoc(ref, cmp, fitting='COG')
#     print(match_cog.peak, match_cog.param)
#
#     match.stitching()
#     match_para.stitching()
#     match_cog.stitching()
#
#     center = np.array(ref.shape) / 2
#     persp = match.poc2warp(center, [-5.40E+01, -2.00E+00, 9.72E+01 / 180 * math.pi, 6.03E-01])
#     match.stitching(persp)
#     # Save particular Image
#     # match.saveMat(match.LPA,'LPA.png')
#     # match.saveMat(match.LPA_filt,'LPA_filt.png')
#     # match.saveMat(match.LA,'LA.png')


# def align_images(image, template):
#     from skimage.registration import optical_flow_tvl1
#     from skimage.transform import warp
#     from skimage.filters import sobel, prewitt, scharr, roberts, farid
#     from sklearn.preprocessing import minmax_scale
#     import numpy as np
#     import matplotlib.pyplot as plt
#     from pystripe.core import filter_streaks
#
#     # --- Compute the optical flow
#     print("step 1")
#     sigma = (256, 256)  # foreground, background
#     template = filter_streaks(template, sigma, level=0, wavelet="db10", crossover=10, threshold=-1)
#     image = filter_streaks(image, sigma, level=0, wavelet="db10", crossover=10, threshold=-1)
#     template_g: ndarray = minmax_scale(sobel(template), feature_range=(0, 255))
#     image_g: ndarray = minmax_scale(sobel(image), feature_range=(0, 255))
#     template_g = filter_streaks(template_g, sigma, level=0, wavelet="db10", crossover=10, threshold=-1)
#     image_g = filter_streaks(image_g, sigma, level=0, wavelet="db10", crossover=10, threshold=-1)
#     plt.imshow(template, cmap='gray', vmin=0, vmax=255)
#     plt.show()
#     plt.imshow(template_g, cmap='gray', vmin=0, vmax=255)
#     plt.show()
#     plt.imshow(image, cmap='gray', vmin=0, vmax=255)
#     plt.show()
#     plt.imshow(image_g, cmap='gray', vmin=0, vmax=255)
#     plt.show()
#     print("step 2")
#     v, u = optical_flow_tvl1(
#         template_g,
#         image_g,
#         attachment=15,
#         tightness=0.3,
#         num_warp=5,
#         num_iter=10,
#         tol=1e-4,
#         prefilter=False,
#         dtype=np.float32
#     )
#     print("step 3")
#
#     # --- Use the estimated optical flow for registration
#     nr, nc = template.shape
#     row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')
#     aligned = warp(image, np.array([row_coords + v, col_coords + u]), mode='edge', preserve_range=True)
#     print("step 4")
#
#     # return the aligned image
#     return aligned


from tsv.volume import TSVVolume
from tsv.convert import convert_to_2D_tif
from pathlib import Path

# stitched_tif_path = Path(r"Y:\3D_stitched\20220506_15_48_26_SW220310_02_R_Hem_LS_15x_1000z_stitched\Ex_488_Em_525_tif")
# stitched_tif_path.mkdir(exist_ok=True)
# shape: convert_to_2D_tif(
#     TSVVolume.load(
#         Path(r"Y:\3D_stitched\20220506_15_48_26_SW220310_02_R_Hem_LS_15x_1000z_stitched\Ex_488_Em_525_xml_import_step_5.xml")
#     ),
#     str(stitched_tif_path / "img_{z:06d}.tif"),
#     compression=("ZLIB", 1),
#     cores=1,
#     dtype='uint8',
#     resume=True
# )

# path = Path(r"E:\20220506_15_48_26_SW220310_02_R_Hem_LS_15x_1000z_lightsheet_cleaned_tif_bitshift.b2.g2.r2_downsampled\Ex_488_Em_525\205890\205890_131640")
# for idx in range(7800):
#     file = path / f"{idx:05}0.tif"
#     if not file.exists():
#         print(file)


from pystripe.core import batch_filter
# from multiprocessing import freeze_support
# from process_images import inspect_for_missing_tiles_get_files_list
# if __name__ == "__main__":
#     freeze_support()
#     inspect_for_missing_tiles_get_files_list(
#         Path(r"E:\20220506_15_48_26_SW220310_02_R_Hem_LS_15x_1000z_lightsheet_cleaned_tif_bitshift.b2.g2.r2_downsampled\Ex_561_Em_600")
#     )

# for z_folder in Path(r"G:\sm20220216-40X_20220426_112014_100ms-3.7TB").iterdir():
#     if z_folder.is_dir():
#         y_folder_names = list(set([f.name.split("_")[0] for f in z_folder.glob("*.TIF")]))
#         for name in y_folder_names:
#             y_folder = z_folder / name
#             y_folder.mkdir(exist_ok=True)
#             for file in z_folder.glob(f"{name}_*.TIF"):
#                 file.rename(y_folder / file.name)

# from logging import basicConfig, DEBUG, debug, INFO, info
# from sys import stderr
# import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle
# basicConfig(stream=stderr, level=INFO)
#
# source = Path(r"E:\sm20220216_NEW_40X_2000z")
# file_suffix = ".TIF"
# file_suffix_length = len(file_suffix)
# tile_size = (2048, 2048)  # (y, x)
# overlap_ratio = 0.05
# metadata = []
#
# for z_folder in source.iterdir():
#     if z_folder.is_dir():
#         debug(z_folder.name)
#         x_n_max, y_n = 0, 0
#         y_min, y_max = float('inf'), float('-inf')
#         x_min, x_max = float('inf'), float('-inf')
#
#         fig, ax = plt.subplots()
#         ax.plot([-49661, 33898], [325593, 387034], alpha=0)
#
#         for y_folder in z_folder.iterdir():
#             if y_folder.is_dir():
#                 y_n += 1
#                 x_n = 0
#                 y_axis = int(y_folder.name[1:])
#                 y_min, y_max = min(y_min, y_axis), max(y_max, y_axis)
#                 prefix_length = len(y_folder.name + "_X")
#                 debug(f"{y_folder.name}, {y_axis}")
#                 for x_file in y_folder.glob("*"+file_suffix):
#                     x_n += 1
#                     x_axis = int(x_file.name[prefix_length:-file_suffix_length])
#                     x_min, x_max = min(x_min, x_axis), max(x_max, x_axis)
#                     debug(f"\t, {x_file.name}, {x_axis}")
#
#                     ax.add_patch(Rectangle((x_axis, y_axis), 2048, 2048, fill=True, alpha=0.5))
#
#                 debug(f"\tx={x_n}")
#                 x_n_max = max(x_n, x_n_max)
#         debug(f"y={y_n}")
#         metadata += [{
#             "z_folder": z_folder.name,
#             "y_min": y_min,
#             "y_max": y_max,
#             "x_min": x_min,
#             "x_max": x_max,
#             "y_n": y_n,
#             "x_n": x_n_max,
#         }]
#         plt.savefig(f"{source/z_folder.name}.jpg")
#         # plt.show()
#         info(metadata[-1])

# info(f"X: {x_min}, {x_max}, {x_n_max}, {x_max-x_min}, {x_n_max*(tile_size[1]-tile_size[1]*overlap_ratio)}")
# info(f"Y: {y_min}, {y_max}, {y_n_max}, {y_max-y_min}, {y_n_max*(tile_size[0]-tile_size[0]*overlap_ratio)}")

# for num in range(1, 8000):
#     if not Path(r"Y:\3D_stitched_LS\20220512_16_48_06_SW220303_04_15x_LS_1000z_stitched\Ex_488_Em_525_tif\deconvolved").joinpath(f"deconv_{num:06}.tif").exists():
#         print(num)

from xml.etree import ElementTree
# tree = ElementTree.parse(r"D:\20220622_17_23_43_SW220414_02_LS_15x_1000Z_MIP_stitched\Ex_488_Em_525_MIP_xml_import_step_5")
# root = tree.getroot()
from parallel_image_processor import parallel_image_processor
from process_images import process_stitched_tif
parallel_image_processor(
    process_stitched_tif,
    source=TSVVolume.load(
        Path(r"/qnap/SmartSPIM_Data/2022_08_18/20220818_12_24_18_SW220405_05_LS_15x_1000z_stitched/Ex_488_Em_525_MIP_xml_import_step_5.xml")
    ),
    destination=Path(r"/data/test"),
    args=(),
    kwargs={"rotation": 0},
    max_processors=12,
    progress_bar_name="tsv",
    compression=("ZLIB", 0)
)
