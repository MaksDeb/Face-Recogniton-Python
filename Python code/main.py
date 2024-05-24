import cv2
import numpy as np

fun = None
img = None
i = 0
object_cascade = None

f = 105
n = 5
s = 50

list_haarcascade = ['haarcascade_frontalface_alt2.xml']


def set_object():
    global i, object_cascade, list_haarcascade
    object_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + list_haarcascade[i])
    print(list_haarcascade[i])
    i += 1
    if i >= len(list_haarcascade):
        i = 0


def bar_fun(x):
    global f, n, s
    f = cv2.getTrackbarPos('scaleFactor', 'bar')
    n = cv2.getTrackbarPos('minNeighbors', 'bar')
    s = cv2.getTrackbarPos('minSize', 'bar')


def main():
    global swich_lib, fun, img, object_cascade, f, n, s
    cap = cv2.VideoCapture("moj_film.mp4")
    set_object()
    ret, frame = cap.read()
    cv2.imshow('bar', frame)
    cv2.createTrackbar('scaleFactor', 'bar', 101, 1000, bar_fun)
    #cv2.createTrackbar('minNeighbors', 'bar', 1, 500, bar_fun)
    cv2.createTrackbar('minNeighbors', 'bar', 41, 500, bar_fun)
    #cv2.createTrackbar('minSize', 'bar', 1, 500, bar_fun)
    cv2.createTrackbar('minSize', 'bar', 56, 500, bar_fun)
    count = 1
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = object_cascade.detectMultiScale(
                gray, scaleFactor=f / 100, minNeighbors=n, minSize=(s, s))

            if len(faces) > 0:
                x, y, w, h = faces[0]
                img = frame[y:y + h, x:x + w]
                img = cv2.resize(img, (200, 200),
                                 interpolation=cv2.INTER_LINEAR)
                if frame_count % 2 == 0:
                    cv2.imwrite(f"different_faces/different_face{count}.png", img)
                    count = count + 1

            cv2.imshow('bar', frame)

        key = cv2.waitKey(1)
        if key == ord('n'):
            set_object()
        elif key == 27:
            cv2.destroyAllWindows()
            break
        frame_count += 1
    print("KONIEC")


if __name__ == '__main__':
    main()
