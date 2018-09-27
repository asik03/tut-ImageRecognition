import cv2 as cv

from src.utilities.face_segmenter import FaceSegmenter

if __name__ == '__main__':
    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        cap.open()

    while True:
        ret, frame = cap.read()
        if ret:
            # TODO: Smile recognition
            _, faces = FaceSegmenter.segment(frame)
            for x, y, w, h in faces:
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv.imshow('frame', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
