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
    # pass by reference.def splitchannels(image):
    b,g,r = cv2.split(image)
    cv2.imshow("Model Blue Image", b)
    cv2.imshow("Model Green Image", g)
    cv2.imshow("Model Red Image", r)

    image = cv2.merge([r, g, b])

    cv2.imshow('Merged', image)
    return image

cap = cv2.VideoCapture("D:\\Tenis_04_07\\Wideonly\\zverev.mp4")
print((int(cap.get(4)), int(cap.get(3))))


out = cv2.VideoWriter("D:/Tenis_04_07/Wideonly/mediapipe/test.mp4", cv2.VideoWriter_fourcc(*'MPG4'), 24, (int(cap.get(3)), int(cap.get(4))))


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose

TENNIS_POSE_CONNECTIONS = frozenset([(11, 12), (11, 13),
                              (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
                              (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
                              (18, 20), (11, 23), (12, 24), (23, 24), (23, 25),
                              (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
                              (29, 31), (30, 32), (27, 31), (28, 32)])
allPoints = []with mp_pose.Pose(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.8) as pose:  frame=0

  
  while cap.isOpened():
  #while frame < 30:
    frame = frame + 1
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
    
      break
    