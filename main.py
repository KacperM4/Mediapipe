from tkinter.tix import IMAGE
import cv2
from matplotlib import image
import mediapipe as mp
import time
import numpy as np

def converttoHSV(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
  

