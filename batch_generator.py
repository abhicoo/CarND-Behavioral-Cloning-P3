import os
import cv2
from sklearn.utils import shuffle
import numpy as np


def generator(samples, batch_size):
	num_samples = len(samples)
	while 1:
		shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]
			images = []
			angles = []
			for batch_sample in batch_samples:
				#Loading Images
				center_name = './data/IMG/'+batch_sample[0].split('/')[-1]
				center_image = cv2.imread(center_name)
				left_name = './data/IMG/'+batch_sample[1].split('/')[-1]
				left_image = cv2.imread(left_name)
				right_name = './data/IMG/'+batch_sample[2].split('/')[-1]
				right_image = cv2.imread(right_name)
				#Loading Steering
				correction = 0.2
				center_angle = float(batch_sample[3])
				left_angle = center_angle + correction
				right_angle = center_angle - correction
				images.extend([center_image, left_image, right_image])
				angles.extend([center_angle, left_angle, right_angle])
			augmented_images = []
			augmented_angles = []
			for image, angle in zip(images, angles):
				augmented_images.append(image)
				augmented_angles.append(angle)
				augmented_images.append(cv2.flip(image, 1))
				augmented_angles.append(angle * -1)
			X_train = np.array(augmented_images)
			y_train = np.array(augmented_angles)
			yield shuffle(X_train, y_train)
