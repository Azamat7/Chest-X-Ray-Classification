from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np


def get_data():
	data_testing = np.load("test_datasets.npy")
	data = []
	for d in data_testing:
		image = np.expand_dims(img_to_array(d), axis=0)
		data.append(image)
	data = np.array(data, dtype="float") / 255.0
	return data


def predict_and_write(data, model):
	f = open("result.txt",'w')
	for i,image in enumerate(data):
		(abnormal, normal) = model.predict(image)[0]
		label = model.predict_classes(image)[0]
		proba = max(normal,abnormal)
		label = "{}. Class: {}, Probability: {:.2f}%\n".format(i+1, label, proba * 100)
		f.write(label)
	f.close()


if __name__ == '__main__':
	data = get_data()
	model = load_model('model.h5')
	predict_and_write(data, model)


	