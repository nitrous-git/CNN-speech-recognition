###########################################################
# Math import 
import numpy as np 
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt

########################################################
# Keras import
import keras
from keras_preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten
from keras.layers import Activation, Dropout, MaxPooling2D
from keras import backend as K
from keras_preprocessing import image


num_classes = 3 
epochs = 15              # nb_epoch = [nb_sample/batch_size] = [450/32] = 14
batch_size = 32

img_width = 160
img_heigth = 120
nb_train_samples = 450
nb_validation_samples = 450

# Train directory
train_data_dir = r'C:\Users\Of Corrupted Vision\Documents\Source Python\CNN speech recognition\train_specgram'
# Validation directory
validation_data_dir = r'C:\Users\Of Corrupted Vision\Documents\Source Python\CNN speech recognition\validation_specgram'

if K.image_data_format() == 'channels_first':
    input_shape = (3,img_width,img_heigth)
else:
    input_shape = (img_width,img_heigth,3)

############################################
# init the Conv2DNet
model = Sequential()

model.add(Conv2D(64, kernel_size = 3, input_shape=input_shape)) 
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, kernel_size = 3, input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.summary()

####################################
# Starting training and validation generator 

# Randomize data set 
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True
)

test_datagen=ImageDataGenerator(rescale=1./255)

# Generator flow from directory
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width,img_heigth),
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width,img_heigth),
    class_mode='categorical'
)

# Compile and full connection
model.compile(
    optimizer='adam', 
    loss='categorical_crossentropy', 
    metrics=['accuracy'])

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples/batch_size,
    validation_data=validation_generator,
    epochs=epochs,
    validation_steps=nb_validation_samples/batch_size,
    verbose=1
)

###################################
# Convert to specgram & Make Predictions

audio_dir = r'C:\Users\Of Corrupted Vision\Documents\Source Python\CNN speech recognition\train_data\no\no (9).wav'
rate_data, audio_data = wavfile.read(audio_dir)  # Return 2D array [[frqs(hz)],[0,0,1,0,....,n]]

# remove borders
fig, ax = plt.subplots(1)
fig.subplots_adjust(left=0,right=1,bottom=0,top=1)

# plt the specgrams and save files
Pxx, freqs, bins, im = ax.specgram(audio_data, NFFT=1024, Fs=44100, noverlap=900) # NFFT 
ax.axis('off')
ax.axis('tight')
fig.savefig(r'C:\Users\Of Corrupted Vision\Documents\Source Python\CNN speech recognition\test_specgram\sg_no9_test.jpg')
    
# load specgram and model predict
predict = image.load_img(r'C:\Users\Of Corrupted Vision\Documents\Source Python\CNN speech recognition\test_specgram\sg_no9_test.jpg', target_size=input_shape)
predict = image.img_to_array(predict)
predict = np.expand_dims(predict,axis=0) 
result_test = model.predict(predict)

# print binary vector result 
print(result_test)
if result_test[0][1] == 1:
    prediction = 'the words is : no'
else:
    prediction ='other words spoken'
print(prediction)
#plt.show()