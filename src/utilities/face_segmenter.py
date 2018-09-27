import cv2 as cv

import src.config as config
# import imutils

# import dlib

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

    # @classmethod
    # def align_face(cls, image, img_size):
    #     # Initialize dlib's face detector (HOG-based) and then create
    #     # the facial landmark predictor and the face aligner.
    #     detector = dlib.get_frontal_face_detector()
    #     predictor = dlib.shape_predictor(predictor_path)
    #     fa = FaceAligner(predictor, desiredFaceWidth=img_size)
    #
    #     # convert it to grayscale.
    #     gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    #
    #     rects = detector(gray, 2)
    #
    #     # Loop over the face detections.
    #     for rect in rects:
    #         # Extract the ROI of the *original* face, then align the face using facial landmarks.
    #         (x, y, w, h) = rect_to_bb(rect)
    #         faceAligned = fa.align(image, gray, rect)
    #
    #     return faceAligned