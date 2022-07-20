import cv2
from matplotlib import image
import mediapipe as mp
import numpy as np

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
  

def mask(image):
    mask = np.zeros(image.shape[:2], dtype="uint8")
    kernel = np.ones((5, 5), np.uint8)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([20, 15, 15])
    upper_hsv = np.array([210, 210, 210])
    Mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    Mask = cv2.erode(Mask, kernel, iterations=1)
    Mask = cv2.morphologyEx(Mask, cv2.MORPH_OPEN, kernel)
    Mask = cv2.dilate(Mask, kernel, iterations=1)
    mask = cv2.bitwise_not(Mask)
    cv2.imshow("Mask", mask)
    masked = cv2.bitwise_and(image, image, mask)
    cv2.imshow("Mask applied to Image", masked)
    return masked

def brightness(image):
    kernel = np.array([[0, -1, 0],
                   [-1, 7,-1],
                   [0, -1, 0]])
    image= cv2.filter2D(src=image, ddepth=-10, kernel=kernel)
    cv2.imshow('brightness',image)
    return image

def split_channels(image):

    b,g,r = cv2.split(image)
    cv2.imshow("Model Blue Image", b)
    cv2.imshow("Model Green Image", g)
    cv2.imshow("Model Red Image", r)

    image = cv2.merge([r, g, b])

    cv2.imshow('Merged', image)
    return image

def pose_landmarks(image, pose):
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            results.pose_landmarks.landmark[6].x = -1
            results.pose_landmarks.landmark[6].y = -1
            results.pose_landmarks.landmark[8].x = -1
            results.pose_landmarks.landmark[8].y = -1
            results.pose_landmarks.landmark[5].x = -1
            results.pose_landmarks.landmark[5].y = -1
            results.pose_landmarks.landmark[4].x = -1
            results.pose_landmarks.landmark[4].y = -1
            results.pose_landmarks.landmark[0].x = -1
            results.pose_landmarks.landmark[0].y = -1
            results.pose_landmarks.landmark[1].x = -1
            results.pose_landmarks.landmark[1].y = -1
            results.pose_landmarks.landmark[2].x = -1
            results.pose_landmarks.landmark[2].y = -1
            results.pose_landmarks.landmark[3].x = -1
            results.pose_landmarks.landmark[3].y = -1
            results.pose_landmarks.landmark[7].x = -1
            results.pose_landmarks.landmark[7].y = -1
            results.pose_landmarks.landmark[9].x = -1
            results.pose_landmarks.landmark[9].y = -1
            results.pose_landmarks.landmark[10].x = -1
            results.pose_landmarks.landmark[10].y = -1
    
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                TENNIS_POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles
                .get_default_tennis_style(),
                connection_drawing_spec=mp_drawing_styles
                .get_default_tennis_style())

            return image

def main():

    wideo_capture = cv2.VideoCapture('D:\\Tenis_04_07\\Wideonly\\zverev.mp4')
    out = cv2.VideoWriter('D:/Tenis_04_07/Wideonly/mediapipe/test.mp4', cv2.VideoWriter_fourcc(*'MPG4'),
    24, (int(wideo_capture.get(3)), int(wideo_capture.get(4))))


    with mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.8) as pose:
        frame=0
  
        while wideo_capture.isOpened():
        #while frame < 30:
            frame = frame + 1
            success, image = wideo_capture.read()
            if not success:
                print("Ignoring empty camera frame.")
                break
    
            image = cv2.resize(image, (700, 450))
            #image = brightness(image)
            image = split_channels(image)
            #image = mask(image)  
            image = pose_landmarks(image, pose)          
            cv2.imshow('MediaPipe ',image)
            out.write(image)
            if cv2.waitKey(100) & 0xFF == 27:
                break
        wideo_capture.release()   
   
if __name__ == '__main__':
    main()