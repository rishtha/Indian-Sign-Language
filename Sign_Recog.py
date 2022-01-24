import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

model = tf.keras.models.load_model(r'D:\miniproject\111SignRecognition\sample_train.h5')
model.summary()
data_dir = r'D:\miniproject\111SignRecognition\dataset'
labels = sorted(os.listdir(data_dir))
labels[-1] = 'Nothing'
print(labels)

cap = cv2.VideoCapture(0)

while(True):
    _ , frame = cap.read()
    cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 5)

    roi = frame[100:400, 100:400]
    img = cv2.resize(roi, (50, 50))
    cv2.imshow('Output', roi)

    img = img/255

    prediction = model.predict(img.reshape(1, 50, 50, 3))
    char_index = np.argmax(prediction)

    confidence = round(prediction[0, char_index]*100, 1)
    predicted_char = labels[char_index]

    font = cv2.FONT_HERSHEY_TRIPLEX
    fontScale = 1
    color = (0, 255, 255)
    thickness = 2

    if confidence > 98:
        msg = predicted_char +', Conf: '+str(confidence) +' %'
        cv2.putText(frame, msg, (80, 80), font, fontScale, color, thickness)
        print(predicted_char)
        cv2.imshow('Output1', frame)

        if cv2.waitKey(2) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()