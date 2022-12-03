#Import Libraries 
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from scipy.spatial.distance import cosine
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
import sys
import time

# extract a single face from a given photograph
class FaceVerify(object):
    detector = MTCNN()
    model = VGGFace(model='resnet50', include_top=False,input_shape=(224, 224, 3), pooling='avg')

    @staticmethod
    def adjust_gamma(image, gamma=0.5):
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    @staticmethod
    def histogram_equalization(image_path):
        pixels = plt.imread(image_path)
        # convert from RGB color-space to YCrCb
        ycrcb_img = cv2.cvtColor(pixels, cv2.COLOR_BGR2YCrCb)
        # equalize the histogram of the Y channel
        ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])
        # convert back to RGB color-space from YCrCb
        equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)
        return equalized_img

    @staticmethod
    def extract_face(filename, detector=detector, required_size=(224, 224)):
        # pixels = plt.imread(filename)
        colhe = FaceVerify.histogram_equalization(filename)
        adjusted = FaceVerify.adjust_gamma(colhe, gamma=0.5)
        # create the detector, using default weights
        # detect faces in the image
        results = detector.detect_faces(adjusted)
        # extract the bounding box from the first face
        x1, y1, width, height = results[0]['box']
        x2, y2 = x1 + width, y1 + height
        # extract the face
        face = adjusted[y1:y2, x1:x2]
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = asarray(image)
        return face_array
# resize pixels to the model size

    @staticmethod
    def get_embeddings(face, model=model):
        # extract faces
        # convert into an array of samples
        samples = asarray(face, 'float32')
        # prepare the face for the model, e.g. center pixels
        samples = preprocess_input(samples, version=2)
        samples = samples[np.newaxis, :]
        # create a vggface model
        model = VGGFace(model='resnet50', include_top=False,input_shape=(224, 224, 3), pooling='avg')
        # perform prediction
        yhat = model.predict(samples)
        model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
        accuracy = model.evaluate(samples, yhat)
        print("[INFO] The Loss and accuracy is",accuracy)
        return yhat
    # define a video capture object

args = list(sys.argv)
idfile = args[-1]
ID_face = FaceVerify.extract_face(idfile, FaceVerify.detector)
ID_embedding = FaceVerify.get_embeddings(ID_face, FaceVerify.model)

cap = cv2.VideoCapture(0)
flag = False
while True:
    # Capture frame-by-frame
    __, frame = cap.read()
    print("[INFO] Frame of each image", frame)
    # Use MTCNN to detect faces
    try:
        result = FaceVerify.detector.detect_faces(frame)
        if result != []:
            for person in result:
                x1, y1, width, height = result[0]['box']
                x2, y2 = x1 + width, y1 + height
                # extract the face
                face = frame[y1:y2, x1:x2]
                # resize pixels to the model size
                subject_face = Image.fromarray(face)
                required_size = (224, 224)
                subject_face = subject_face.resize(required_size)
                sample = asarray(subject_face, 'float32')
                sample = preprocess_input(sample, version=2)
                subject_embeddings = FaceVerify.get_embeddings(subject_face)
                score = cosine(ID_embedding, subject_embeddings)
                thresh = 0.5
                if score <= thresh:
                    print('>face is a Match (%.3f <= %.3f)' % (score, thresh))
                    flag = True
                else:
                    print('>face is NOT a Match (%.3f > %.3f)' %(score, thresh))
                    flag = True
    except Exception as e:
        print(e) 
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & (flag or 0xFF == ord('q')):
        break
# When everything's done, release capture
cap.release()
cv2.destroyAllWindows()
