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
				name = './data/IMG/'+batch_sample[0].split('/')[-1]
				center_image = cv2.imread(name)
				center_angle = float(batch_sample[3])
				images.append(center_image)
				angles.append(center_angle)
			X_train = np.array(images)
			y_train = np.array(angles)
			yield shuffle(X_train, y_train)
