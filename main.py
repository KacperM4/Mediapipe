from tkinter.tix import IMAGE
import cv2
from matplotlib import image
import mediapipe as mp
import time
import numpy as np

def converttoHSV(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
  


def converttogray(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('M ',image)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

def mask(image):    mask = np.zeros(image.shape[:2], dtype="uint8")
            # creating a rectangle on the mask
      # where the pixels are valued at 255
        # The kernel to be used for dilation purpose
    kernel = np.ones((5, 5), np.uint8)
          
    # converting the image to HSV format
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)    # defining the lower and upper values of HSV,
    # this will detect yellow colour
    Lower_hsv = np.array([20, 15, 15])
    Upper_hsv = np.array([210, 210, 210])