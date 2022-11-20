# example of face detection with mtcnn
from matplotlib import pyplot
from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
import sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import cv2
# extract a single face from a given photograph

class ExtractFace(object):
	
	@staticmethod
	def adjust_gamma(image, gamma=0.8):
		invGamma = 1.0 / gamma
		table = np.array([((i / 255.0) ** invGamma) * 255
			for i in np.arange(0, 256)]).astype("uint8")
		return cv2.LUT(image, table)
		
	@staticmethod
	def histogram_equalization(image_path):
		pixels=pyplot.imread(image_path)
		# rgb_img = ExtractFace.adjust_gamma(pixels, gamma=0.5)
		# convert from RGB color-space to YCrCb
		ycrcb_img = cv2.cvtColor(pixels, cv2.COLOR_BGR2YCrCb)
		# equalize the histogram of the Y channel
		ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])
		# convert back to RGB color-space from YCrCb
		equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)
		# cv2.putText(equalized_img, "equlized hist image", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
		# cv2.imwrite("equalized_image_op",equalized_img)
		return equalized_img

	@staticmethod
	def extract_face( filename,required_size=(224, 224)):
		# load image from file
		colhe=ExtractFace.histogram_equalization(filename)
		gamma_image=ExtractFace.adjust_gamma(colhe,gamma=1.0)
		# create the detector, using default weights
		detector = MTCNN()
		# detect faces in the image
		results = detector.detect_faces(gamma_image)
		# extract the bounding box from the first face
		x1, y1, width, height = results[0]['box']
		x2, y2 = x1 + width, y1 + height
		# extract the face
		face = gamma_image[y1:y2, x1:x2]
		# resize pixels to the model size
		image = Image.fromarray(face)
		image = image.resize(required_size)
		face_array = asarray(image)
		bgr2rgb=cv2.cvtColor(face_array,cv2.COLOR_BGR2RGB)
		cv2.imwrite('face_extract_from_id.jpg',bgr2rgb)
		return face_array
 
# load the photo and extract the face
args=list(sys.argv)
idfile=args[-1]
pixels = ExtractFace().extract_face(idfile)
# plot the extracted face
pyplot.imshow(pixels)
# show the plot
pyplot.show()