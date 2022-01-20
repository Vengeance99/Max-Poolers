import numpy as np
import cv2
import time
import mediapipe as mp

import moviepy.editor as mo

path_model = "models/"

model_name = "model-f6b98070.onnx";
model = cv2.dnn.readNet(path_model + model_name)


if (model.empty()):
    print("Could not load the neural net! - Check path")
model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
 
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture('C:\Users\Dell\Downloads\window.mp4')
pTime = 0
clip = mo.VideoFileClip("C:\Users\Dell\Downloads\window.mp4")
audio=clip.audio
while cap.isOpened():

    ret, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 10, 70)
    results1 = hands.process(imgRGB)
    results = pose.process(imgRGB)
    ret, mask = cv2.threshold(canny, 70, 255, cv2.THRESH_BINARY)
    cv2.imshow('Video feed', mask)
    imgHeight, imgWidth, channels = img.shape
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            print(id, lm)
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

    if results1.multi_hand_landmarks:
        for handLms in results1.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)
                cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
 
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    start = time.time()


    blob = cv2.dnn.blobFromImage(imgRGB, 1/255., (384,384), (123.675, 116.28, 103.53), True, False)

    model.setInput(blob)

    output = model.forward()
    
    output = output[0,:,:]
    output = cv2.resize(output, (imgWidth, imgHeight))

    output = cv2.normalize(output, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)


    end = time.time()
    fps = 1 / (end-start)

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,(255, 0, 0), 3)

    cv2.imshow('image', img)
    cv2.imshow('Depth Map', output)


    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

 

cap.release()
cv2.destroyAllWindows()
