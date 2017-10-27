from keras.models import Model
from keras.layers import Dense, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers import BatchNormalization, Input, Flatten, Lambda, Cropping2D, Dropout
from sklearn.model_selection import train_test_split
from batch_generator import generator
import csv

batch_size = 128
epochs = 10

samples = []
with open('./data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	next(reader)
	for line in reader:
		samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size = 0.2, random_state=88)
total_training_samples = len(train_samples)
total_validation_samples  = len(validation_samples)
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

#Input Layer for model
input_shape = (160, 320, 3)
inputs = Input(shape = input_shape)
#End of Input Layer

#Image Pre Processing Steps
#Zero Mean And Normalizing image
normalization = Lambda(lambda x: x / 127.5 - 0.5)(inputs)
#Cropping Unwanted region from image
cropping = Cropping2D(cropping = ((70, 25), (0, 0)))(normalization)
#End of Image Pre Processing

#Conv1 layer filter_size of 5 stride 2 and total_filters 24
conv1 = Convolution2D(24, 
							nb_row = 5,
							nb_col = 5,
							border_mode ='valid',
							subsample = (2, 2))(cropping)

#conv1 = BatchNormalization()(conv1)
conv1 = Activation('relu')(conv1)
conv1 = Dropout(0.5)(conv1)
#End of Conv1

#Conv2 layer filter_size of 5 stride 2 and total_filters 36
conv2 = Convolution2D(36,
							nb_row = 5,
							nb_col = 5,
							border_mode = 'valid',
							subsample = (2, 2))(conv1)

#conv2 = BatchNormalization()(conv2)
conv2 = Activation('relu')(conv2)
conv2 = Dropout(0.5)(conv2)
#End of Conv2

#Conv3 layer filter_size of 5 stride 2 and total_filters 48
conv3 = Convolution2D(48,
							nb_row = 5,
							nb_col = 5,
							border_mode = 'valid',
							subsample = (2, 2))(conv2)

#conv3 = BatchNormalization()(conv3)
conv3 = Activation('relu')(conv3)
conv3 = Dropout(0.5)(conv3)
#End of Conv3

#Conv4 layer filter_size of 3 stride 1 and total_filters 64
conv4 = Convolution2D(64,
							nb_row = 3,
							nb_col = 3,
							border_mode = 'valid',
							subsample = (1, 1))(conv3)

#conv4 = BatchNormalization()(conv4)
conv4 = Activation('relu')(conv4)
conv4 = Dropout(0.5)(conv4)
#End of Conv4

#Conv5 layer filter_size of 3 stride 1 and total_filters 64
conv5 = Convolution2D(64,
							nb_row = 3,
							nb_col = 3,
							border_mode = 'valid',
							subsample = (1, 1))(conv4)

#conv5 = BatchNormalization()(conv5)
conv5 = Activation('relu')(conv5)
conv5 = Dropout(0.5)(conv5)
#End of Conv5

#Flatten the last conv output.
#Flattened output will be input to fully connected layer
fc0 = Flatten()(conv5)

#First fully connected layer
fc1 = Dense(100)(fc0)
#fc1 = BatchNormalization()(fc1)
fc1 = Activation('relu')(fc1)
fc1 = Dropout(0.5)(fc1)
#End of FC1

#Second fully connected layer
fc2 = Dense(50)(fc1)
#fc2 = BatchNormalization()(fc2)
fc2 = Activation('relu')(fc2)
fc2 = Dropout(0.5)(fc2)
#End of FC2

#Third fully connected layer
fc3 = Dense(10)(fc2)
#fc3 = BatchNormalization()(fc3)
fc3 = Activation('relu')(fc3)
#End of FC3

#Final output layer
outputs = Dense(1)(fc3)
#End of Ouput layer


model = Model(input = inputs, output = outputs)
model.compile(loss = 'mse', optimizer = 'adam')
model.fit_generator(train_generator,
									samples_per_epoch = total_training_samples,
									validation_data = validation_generator,
									nb_val_samples = total_validation_samples,
									nb_epoch = epochs,
									verbose = 1)
model.save('model.h5')

