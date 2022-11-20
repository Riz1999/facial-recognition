from matplotlib import pyplot
from PIL import Image
from numpy import asarray
from scipy.spatial.distance import cosine
from mtcnn.mtcnn import MTCNN
from tensorflow import keras
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from face_detect import ExtractFace
import sys

# extract faces and calculate face embeddings for a list of photo files
class FaceMatch(object):
	@staticmethod
	def get_embeddings(filenames):
		# extract faces
		faces = [ExtractFace.extract_face(f) for f in filenames]
		# convert into an array of samples
		samples = asarray(faces, 'float32')
		# prepare the face for the model, e.g. center pixels
		samples = preprocess_input(samples, version=2)
		# create a vggface model
		model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
		Image.fromarray# model.summary()
		yhat = model.predict(samples)
		model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
		accuracy = model.evaluate(samples, yhat)
		print("[INFO]--->The accuracy score is",accuracy[1])
		print("[INFO]---> The loss is",accuracy[0])
		return yhat

		# return yhat
	
	# determine if a candidate face is a match for a known face
	@staticmethod
	def is_match(known_embedding, candidate_embedding, thresh=0.5):
		# calculate distance between embeddings
		score = cosine(known_embedding, candidate_embedding)
		if score <= thresh:
			print('>face is a Match (%.3f <= %.3f)' % (score, thresh))
		else:
			print('>face is NOT a Match (%.3f > %.3f)' % (score, thresh))
 
# define filenames
args=list(sys.argv)
filenames = [args[1],args[2]]
# get embedings file filenames
embeddings = FaceMatch.get_embeddings(filenames)
FaceMatch.is_match(embeddings[0], embeddings[1])
