import os
import cv2
from sklearn.utils import shuffle
import numpy as np

def augment_brightness_camera_images(image):
	image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
	image1 = np.array(image1, dtype = np.float64)
	random_bright = .5+np.random.uniform()
	image1[:,:,2] = image1[:,:,2]*random_bright
	image1[:,:,2][image1[:,:,2]>255]  = 255
	image1 = np.array(image1, dtype = np.uint8)
	image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
	return image1

def flip_image(image):
	return cv2.flip(image, 1)


def generator(samples, batch_size):
	num_samples = len(samples)
	while 1:
		shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]
			images = []
			angles = []
			for batch_sample in batch_samples:
				center_angle = float(batch_sample[3])
				correction = 0.25
				left_angle = center_angle + correction
				right_angle = center_angle - correction
				angles_data = [center_angle, left_angle, right_angle]

				for i in range(3):
					name = './data/IMG/'+batch_sample[i].split('/')[-1]
					image = cv2.imread(name)
					image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
					images.append(image)
					angles.append(angles_data[i])
					#Flipped Image
					flipped_image = flip_image(image)
					images.append(flipped_image)
					angles.append(angles_data[i] * -1)
			X_train = np.array(images)
			y_train = np.array(angles)
			yield shuffle(X_train, y_train)
