import cv2
from playsound import playsound
import time
import os
import HandTrackingModule as htm
import pyttsx3

engine = pyttsx3.init()
album = {"1": "Prototype 1/Sound/one.mp3", 
         "2": "Prototype 1/Sound/two.mp3", 
         "3": "Prototype 1/Sound/three.mp3", 
         "4": "Prototype 1/Sound/four.mp3", 
         "5": "Prototype 1/Sound/five.mp3"}

wCam, hCam = 640, 480

cap = cv2.VideoCapture(2)
cap.set(3, wCam)
cap.set(4, hCam)

folderPath = r"C:\Users\phonl\OneDrive\Documents\AI\Motion analysis\FingerData"
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    # print(f'{folderPath}/{imPath}')
    new_size = (200, 200)
    image = cv2.resize(image, new_size)
    overlayList.append(image)

print(len(overlayList))
pTime = 0

detector = htm.handDetector(detectionCon=1)

tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    # print(lmList)

    if len(lmList) != 0:
        fingers = []

        # Thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 Fingers
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        # print(fingers)
        totalFingers = fingers.count(1)
        #engine.say(totalFingers)
        #engine.runAndWait()
        print(totalFingers)
        #playsound(album[str(totalFingers)])
        h, w, c = overlayList[totalFingers - 1].shape 
        img[0:h, 0:w] = overlayList[totalFingers - 1]

        #cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
        #cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
    #end of program