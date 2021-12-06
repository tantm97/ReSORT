from __future__ import print_function
import cv2
import numpy as np
from skimage import transform as trans
import torch

def alignment(im, facial_landmarks, dst_w=112, dst_h=112):
    # facial5points = []
    # for i in range(len(landmarks)):
    #     for k in range(2):
    #         facial5points.append(landmarks[i][k])
    # facial5points = np.reshape(facial5points, (2, 5))
    
    # default_square = True
    # inner_padding_factor = 0.0001
    # outer_padding = (0, 0)
    # output_size = (128,128)
    # reference_5pts = get_reference_facial_points(
    #     output_size, inner_padding_factor, outer_padding, default_square)

    # # dst_img = warp_and_crop_face(raw, facial5points, reference_5pts, crop_size)
    # dst_img = warp_and_crop_face(im, facial5points, reference_pts=reference_5pts, crop_size=output_size)
    # return dst_img
    reference_facial_points = np.array([
                [38.2946, 51.6963],
                [73.5318, 51.5014],
                [56.0252, 71.7366],
                [41.5493, 92.3655],
                [70.7299, 92.2041] ], dtype=np.float32)
    # reference_facial_points = np.array([
    #     [46.2946, 59.6963],
    #     [81.5318, 59.5014],
    #     [64.0252, 79.7366],
    #     [49.5493, 100.3655],
    #     [78.7299, 100.2041] ], dtype=np.float32)
    tform = trans.SimilarityTransform()
    tform.estimate(np.array(facial_landmarks), reference_facial_points)
    M = tform.params[0:2,:]
    aligned_face = cv2.warpAffine(im, M, (dst_w,dst_h), borderValue = 0.0)

    return aligned_face


# def preprocess_image(im, landmarks):
#     if im is None:
#         return None
#     image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#     image = cv2.resize(image, (128,128))
#     # image = self.alignment(image, landmarks, 128, 128)
#     image = image[:, :, None]
#     image = image.transpose((2, 0, 1))
#     image = image[:, np.newaxis, :, :]
#     image = image.astype(np.float32, copy=False)
#     image -= 127.5
#     image /= 127.5
#     return image

def preprocess_image(im, use_cpu):
    if im is None:
        return None

    check_cuda_available = False
    if not use_cpu:
        if torch.cuda.is_available():
            # print("Cuda is available.")
            check_cuda_available = True
        # else:
        #     print("Cuda is not available.")

    img = cv2.resize(im, (112,112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    if check_cuda_available:
        device = torch.cuda.current_device()
        img = img.to(device)
    img.div_(255).sub_(0.5).div_(0.5)
    return img



def check_frontal_face(facial_landmarks, thresh_dist_low = 0.7, thresh_dist_high = 1.3, thresh_high_std = 0.5):
    if (facial_landmarks[2][0] < facial_landmarks[0][0] or facial_landmarks[2][1] < facial_landmarks[0][1]
        or facial_landmarks[2][0] < facial_landmarks[3][0] or facial_landmarks[2][1] > facial_landmarks[3][1]
        or facial_landmarks[2][0] > facial_landmarks[1][0] or facial_landmarks[2][1] < facial_landmarks[1][1]
        or facial_landmarks[2][0] > facial_landmarks[4][0] or facial_landmarks[2][1] > facial_landmarks[4][1]):

        return False
    
    wide_dist = np.linalg.norm(np.array(facial_landmarks[0]) - np.array(facial_landmarks[1]))
    high_dist = np.linalg.norm(np.array(facial_landmarks[0]) - np.array(facial_landmarks[3]))
    dist_rate = high_dist / wide_dist

    # cal std
    vec_A = np.array(facial_landmarks[0]) - np.array(facial_landmarks[2])
    vec_C = np.array(facial_landmarks[3]) - np.array(facial_landmarks[2])
    dist_A = np.linalg.norm(vec_A)
    dist_C = np.linalg.norm(vec_C)

    # cal rate
    high_rate = dist_A / dist_C
    high_ratio_std = np.fabs(high_rate - 1.1)  # smaller is better
    
    if(dist_rate < thresh_dist_low or dist_rate > thresh_dist_high or high_ratio_std > thresh_high_std):
        return False
    return True
