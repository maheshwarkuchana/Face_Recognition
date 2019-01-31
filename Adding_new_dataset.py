from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os
import tqdm


ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True, help="Path to encoding.pickle")
ap.add_argument("-i", "--dataset", required=True, help="Path to Dataset")
ap.add_argument("-d", "--detection-method", type=str, default="hog",
                help="face detection model to use either 'hog' or 'cnn'")
args = vars(ap.parse_args())

data = pickle.loads(open(args["encodings"], "rb").read())

imagePaths = list(paths.list_images(args["dataset"]))

for (i, imagePath) in enumerate(imagePaths):
    print("[INFO] processing image {}/{}".format(i+1, len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]

    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    boxes = face_recognition.face_locations(
        rgb, model=args["detection_method"])
    encodings = face_recognition.face_encodings(rgb, boxes)

    for encoding in encodings:
        data["encodings"].append(encoding)
        data["names"].append(name)

print("[INFO] serializing encodings")
# data = {"encodings": knownEncodings, "names": knownNames}
f = open(args["encodings"], "wb")
f.write(pickle.dumps(data))
f.close()
