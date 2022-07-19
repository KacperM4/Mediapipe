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
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #cv2.imshow('hsv', hsv)
    # defining the lower and upper values of HSV,
    # this will detect yellow colour
    Lower_hsv = np.array([20, 15, 15])
    # creating the mask by eroding,morphing,
    # dilating process
    Mask = cv2.inRange(hsv, Lower_hsv, Upper_hsv)
    Mask = cv2.erode(Mask, kernel, iterations=1)
    Mask = cv2.morphologyEx(Mask, cv2.MORPH_OPEN, kernel)
    Mask = cv2.dilate(Mask, kernel, iterations=1)
        
    # Inverting the mask by
    # performing bitwise-not operation
    mask = cv2.bitwise_not(Mask)
  
    # Displaying the image
    cv2.imshow("Mask", mask)

           # performing a bitwise_and with the image and the mask
    masked = cv2.bitwise_and(image, image, mask=mask)
    cv2.imshow("Mask applied to Image", masked)
    return masked

def jasnosc(image):    kernel = np.array([[0, -1, 0],
                   [-1, 7,-1],
                   [0, -1, 0]])
    image= cv2.filter2D(src=image, ddepth=-10, kernel=kernel)
    cv2.imshow('M ',image)
    return image
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.