import cv2
import mediapipe
mpHand=mediapipe.solutions.hands

camera=cv2.VideoCapture(0)

hands=mpHand.Hands() # el objesini yaratıyoruz

mpDraw=mediapipe.solutions.drawing_utils  # frame üzerine elli çizmeye yarıyor
wait=False

while True:
    ret,frame=camera.read()
    frame=cv2.flip(frame,1)

    if ret==False:
        break

    frameRGB=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    hlsm=hands.process(frameRGB)  # El in içinde olan noktaları şeçecektir

    height, width, channel = frame.shape

    #print(hlsm.multi_hand_landmarks) # el üzerindeki 21 noktasının min max versiyonunu veriyor bir dizi olarak
    if hlsm.multi_hand_landmarks:
        #print(len(hlsm.multi_hand_landmarks)) # framde kaç tane el olduğunu bize söylüyor
        for handlandmarks in hlsm.multi_hand_landmarks: # Aldığımız değerleri göstermeye çaılşalımfaram üzerinde
            for fingerNun,landmark in enumerate(handlandmarks.landmark):
                positionX,positionY=int(landmark.x*width),int(landmark.y*height)

                cv2.putText(frame,str(fingerNun),(positionX,positionY),cv2.FONT_HERSHEY_SIMPLEX,1,(200,255,0),2) # el üzerinde 21 el ekleminin sayılarını yazar üzerine

                if fingerNun >20 and landmark.y< handlandmarks.landmark[2].y: # burda yapmak istediğimiz şey el işareti ile onay verdiğimizde ne yagerektiğidir
                    break
                if fingerNun == 20 and landmark.y> handlandmarks.landmark[2].y:

                    wait=True

            mpDraw.draw_landmarks(frame,handlandmarks,mpHand.HAND_CONNECTIONS) # frame üzerinde gösterelim

    cv2.imshow("Windows",frame)
    if wait:
        cv2.waitKey(3000)
        break


    if cv2.waitKey(1) & 0xFF==ord("q"):
        break