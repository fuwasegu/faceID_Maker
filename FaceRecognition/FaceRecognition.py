# coding: utf-8

import cv2 

font = cv2.FONT_HERSHEY_SIMPLEX

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cascade_path_human = 'haarcascade_frontalface_default.xml'
    cascade_path_hirosugu = "cascade_hirosugu.xml"

    cascade_hirosugu = cv2.CascadeClassifier(cascade_path_hirosugu)
    cascade_human = cv2.CascadeClassifier(cascade_path_human)

    color = (255,0,0)

    while True:
        ret, frame = cap.read()
    
        facerect_human = cascade_human.detectMultiScale(frame, scaleFactor=1.7, minNeighbors=4, minSize=(100,100))
        facerect_hirosugu = cascade_hirosugu.detectMultiScale(frame, scaleFactor=1.7, minNeighbors=4, minSize=(100,100))
        facerect_chiaya = cascade_chiaya.detectMultiScale(frame, scaleFactor=1.7, minNeighbors=4, minSize=(100,100))

        if len(facerect_human) > 0:
            for rect in facerect_human:
                cv2.rectangle(frame, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), (255,255,255), thickness=2)


        if len(facerect_hirosugu) > 0:
            for rect in facerect_hirosugu:
                cv2.rectangle(frame, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), color, thickness=2)
                cv2.putText(frame, 'Hirosugu Takeshita', tuple(rect[0:2]), font, 2,(0,0,0),2,cv2.LINE_AA)

    
        cv2.imshow("frame", frame)

        # qキーを押すとループ終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
