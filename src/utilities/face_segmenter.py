import numpy as np
import cv2 as cv


class FaceSegmenter:

    @classmethod
    def segment(cls, image):
        face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

        gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        img_cropped_list = []

        for x, y, w, h in faces:
            cv.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_color = image[y:y + h, x:x + w]
            img_cropped_list.append(roi_color)

        return img_cropped_list, faces
