import cv2 
from google.colab.patches import cv2_imshow
img = []
cntt=0
vidcap = cv2.VideoCapture('/content/drive/MyDrive/Colab Notebooks/WhatsApp Video 2022-01-20 at 11.01.55 PM.mp4') 
def getFrame(sec): 
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000) 
    hasFrames,image = vidcap.read() 
    if hasFrames: 
        # fps = vidcap.get(cv2.CAP_PROP_FPS)
        # print(fps)
        # cntt+=1
        img.append(image)
    return hasFrames 
sec = 0 
frameRate = 1
success = getFrame(sec) 
while success: 
    cntt+=1
    sec = sec + frameRate 
    sec = round(sec, 2) 
    success = getFrame(sec) 


import cv2
import mediapipe as mp
import time

# cap = cv2.VideoCapture(1)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0
i=0
while i<cntt:
    imgRGB = cv2.cvtColor(img[i], cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)
                h, w, c = img[i].shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                # if id == 4:
                cv2.circle(img[i], (cx, cy), 15, (255, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(img[i], handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
    #             (255, 0, 255), 3)

    cv2_imshow(img[i])
    cv2.waitKey(1)
    i+=1
