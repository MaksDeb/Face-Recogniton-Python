import tensorflow as tf
import cv2
import numpy as np

model = tf.keras.models.load_model('models/wmapro4.h5')


def detect_faces(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces


def predict_face(image):
    image = cv2.resize(image, (200, 200))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #image = np.array(image, dtype=np.float32) / 255.0
    image = np.array(image, dtype=np.float32)
    image = image.reshape(1, 200, 200, 3)
    prediction = model.predict(image)
    rounded_prediction = np.round(prediction).astype(int)
    print(rounded_prediction)
    return np.argmax(prediction, axis=1)


def read_from_camera():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if ret:
            faces = detect_faces(frame)
            for (x, y, w, h) in faces:
                face_image = frame[y:y+h, x:x+w]
                label = predict_face(face_image)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                if label == 0:
                    cv2.putText(frame, 'Inna twarz', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                else:
                    cv2.putText(frame, 'ja', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()


read_from_camera()
