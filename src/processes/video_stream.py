import cv2

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        cap.open()

    while True:
        ret, frame = cap.read()
        if ret:
            # TODO: Face segmentation
            # TODO: Smile recognition
            cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
