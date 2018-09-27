import cv2 as cv

import src.config as config


class FaceSegmenter:
    @classmethod
    def segment(cls, image):
        face_cascade = cv.CascadeClassifier(config.XML_FACE)

        gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        img_cropped_list = []
        for x, y, w, h in faces:
            roi_color = image[y:y + h, x:x + w]
            img_cropped_list.append(roi_color)

        return img_cropped_list, faces
