import cv2
from Lego import Lego


VWIDTH = 640
VHIGH = 480
cap=cv2.VideoCapture(0)
ret = cap.set(3, VWIDTH)
ret = cap.set(4, VHIGH)

while 1:
    _, frame = cap.read()

    imgH,imgW, _ = frame.shape

    l = Lego(frame)
    logo_box = l.get_logo_box()

    cv2.drawContours(frame, [logo_box], -1, (0, 255, 0), 2)

    try:
        l.get_information_part()
        cv2.imshow('info', l._information)
    except:
        pass


    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
#     out.release()
cv2.destroyAllWindows()
