import numpy as np
import cv2
from skimage.metrics import structural_similarity as compare_ssim

def ssim(imageA, imageB):
    imageA = np.array(imageA * 255, dtype=np.uint8).transpose(1,2,0)
    imageB = np.array(imageB * 255, dtype=np.uint8).transpose(1,2,0)

    # convert the images to grayscale BGR2GRAY
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    (grayScore, diff) = compare_ssim(grayA, grayB, full=True)
    return grayScore