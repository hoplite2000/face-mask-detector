from tensorflow import keras
import numpy as np
import cv2
import imutils
import time
import h5py
from imutils.video import VideoStream


def detect_and_predict(frame, facenet, masknet):
    (h, w) = frame.shape[:2]
    # create a blob for frame
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    # detect faces
    facenet.setInput(blob)
    detections = facenet.forward()

    # initialize the list of faces, their corresponding locations and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (X, Y, eX, eY) = box.astype('int')
            # ensure whether box is inside frame
            (X, Y) = (max(0, X), max(0, Y))
            (eX, eY) = (min(w - 1, eX), min(h - 1, eY))

            # extract face
            face = frame[Y:eY, X:eX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = keras.preprocessing.image.img_to_array(face)
            face = keras.applications.mobilenet_v2.preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            # append face and its location
            faces.append(face)
            locs.append((X, Y, eX, eY))

    # prediction
    if len(faces) > 0:
        preds = masknet.predict(faces)

    return (locs, preds)

#main
prototxtpath = './face_detector/deploy.prototxt'
caffemodelpath = './face_detector/res10_300x300_ssd_iter_140000.caffemodel'

# face detector
print("[INFO] loading face detector model...")
facenet = cv2.dnn.readNet(prototxtpath, caffemodelpath)

# mask classifier
print("[INFO] loading face mask detector model...")
masknet = keras.models.load_model('./model/model.h5')

# video stream
print("[INFO] starting video stream...")
cam = VideoStream(0).start()
time.sleep(1)

while True:
    frame = cam.read()
    frame = cv2.flip(frame, 1)
    frame = imutils.resize(frame, width=680)
    (locs, preds) = detect_and_predict(frame, facenet, masknet)

    # loop over the detected face locations and their corresponding locations
    for (box, pred) in zip(locs, preds):
        (X, Y, eX, eY) = box
        (mask, withoutmask) = pred

        # determine the class label and color we'll use to draw the bounding box and text
        label = "Mask" if mask > withoutmask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # include the probability in the label
        #label = "{}: {:.2f}%".format(label, max(mask, withoutmask) * 100)

        # display the label and bounding box rectangle on the output frame
        cv2.putText(frame, label, (X, Y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (X, Y), (eX, eY), color, 2)
        
    # show the output frame
    cv2.imshow("Cam", frame)
    key = cv2.waitKey(30)
    if key == 27:
        break

cv2.destroyAllWindows()
cam.stop()
