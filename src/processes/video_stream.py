import cv2 as cv
import os
import numpy as np
from keras.models import load_model

import src.config as config
from src.utilities.face_segmenter import FaceSegmenter

if __name__ == '__main__':
    model = load_model(os.path.join(config.MODELS_DIR, 'model.h5'))
    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        cap.open()

    while True:
        ret, frame = cap.read()
        if ret:
            # TODO: Smile recognition
            images, faces = FaceSegmenter.segment(frame)
            labels = []
            for image in images:
                image = cv.resize(image, (64, 64))
                image = np.expand_dims(image, axis=0)
                label = model.predict_classes(image)
                labels.append(label)
            counter = 0
            for x, y, w, h in faces:
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv.putText(frame, str(labels[counter]), (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                counter += 1
            cv.imshow('frame', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
