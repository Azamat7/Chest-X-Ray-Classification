from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from model import CNN_model
import matplotlib.pyplot as plt
import numpy as np


def get_data():
	data_normal = np.load("training_normal.npy")
	data_abnormal = np.load("training_abnormal.npy")

	data = []
	for d in data_normal:
		data.append(img_to_array(d))
	for d in data_abnormal:
		data.append(img_to_array(d))
	data = np.array(data, dtype="float") / 255.0
	labels = np.array([1]*400+[0]*400)

	assert(len(data) == len(labels))
	assert(len(data) == 800)

	return data, labels


def plot_training(H):
	plt.style.use("ggplot")
	plt.figure()
	N = EPOCHS
	plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
	plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
	plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
	plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
	plt.title("Training Loss and Accuracy")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="lower left")
	plt.savefig("plot.jpg")


if __name__ == '__main__':
	EPOCHS = 40
	INIT_LR = 0.0007
	BS = 60

	data, labels = get_data()
	
	# partition the data into training and testing: 75% and 25%
	(x_train, x_test, y_train, y_test) = train_test_split(data,
		labels, test_size=0.25, random_state=42)
	 
	# convert the labels from integers to vectors
	y_train = to_categorical(y_train, num_classes=2)
	y_test = to_categorical(y_test, num_classes=2)

	# construct the image generator for data augmentation
	aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
		height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
		horizontal_flip=True, fill_mode="nearest")

	# compile the model
	model = CNN_model.build(width=200, height=200, depth=1, classes=2)
	opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
	model.compile(loss="binary_crossentropy", optimizer=opt,
		metrics=["accuracy"])
	 
	# train the network
	H = model.fit_generator(aug.flow(x_train, y_train, batch_size=BS),
		validation_data=(x_test, y_test), steps_per_epoch=len(x_train) // BS,
		epochs=EPOCHS, verbose=1)
	 
	# save the model to disk
	model.save('model.h5')

	# plot the training loss and accuracy
	plot_training(H)









