# python recognize_faces_image.py --encodings encoding.pickle --detection-method hog --image C:\Users\Anonymous\Documents\Visual_Studio_files\PyFiles\Face_Recognition\Test_Images/9.jpg

import face_recognition
import argparse
import cv2
import pickle

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
                help="path to serialized db of facial encodings")
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
ap.add_argument("-d", "--detection-method", type=str, default="hog",
                help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

data = pickle.loads(open(args["encodings"], "rb").read())

image = cv2.imread(args["image"])
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

boxes = face_recognition.face_locations(rgb, model=args["detection_method"])
enocodings = face_recognition.face_encodings(rgb, boxes)

names = []

for encoding in enocodings:
    matches = face_recognition.compare_faces(data["encodings"], encoding)
    name = "Unknown"

    if True in matches:
        matchedIdx = [i for (i, b) in enumerate(matches) if b]
        counts = {}

        for i in matchedIdx:
            name = data["names"][i]
            counts[name] = counts.get(name, 0)+1

        name = max(counts, key=counts.get)

    names.append(name)

for ((top, right, bottom, left), name) in zip(boxes, names):
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    y = top-15 if top - 15 > 15 else top+15
    cv2.putText(image, name, (left, y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

image = cv2.resize(image, (0, 0), fx=0.2, fy=0.2)
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
