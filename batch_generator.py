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
				center_image = cv2.cvtColor(center_image,cv2.COLOR_BGR2RGB)
				flip_center_image = cv2.flip(center_image, 1)
				left_name = './data/IMG/'+batch_sample[1].split('/')[-1]
				left_image = cv2.imread(left_name)
				left_image = cv2.cvtColor(left_image,cv2.COLOR_BGR2RGB)
				flip_left_image = cv2.flip(left_image, 1)
				right_name = './data/IMG/'+batch_sample[2].split('/')[-1]
				right_image = cv2.imread(right_name)
				right_image = cv2.cvtColor(right_image,cv2.COLOR_BGR2RGB)
				flip_right_image = cv2.flip(right_image, 1)
				#Loading Steering
				correction = 0.25
				center_angle = float(batch_sample[3])
				flip_center_angle = center_angle * -1
				left_angle = center_angle + correction
				flip_left_angle = left_angle * -1
				right_angle = center_angle - correction
				flip_right_angle = right_angle * -1

				images.extend([center_image, left_image, right_image, flip_center_image, flip_left_image, flip_right_image])
				angles.extend([center_angle, left_angle, right_angle, flip_center_angle, flip_left_angle, flip_right_angle])
			X_train = np.array(images)
			y_train = np.array(angles)
			yield shuffle(X_train, y_train)
