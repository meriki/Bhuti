
!pip uninstall tensorflow
pip install tensorflow-gpu

from scipy.io import loadmat
import pandas as pd
import tensorflow as tf
import os
import numpy as np

tf.__version__

from tensorflow.python.client import device_lib
device_lib.list_local_devices()

tf.test.gpu_device_name()

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

from google.colab import drive
drive.mount('/content/drive')

# Data Creation

annots = []
for filename in os.listdir('drive/My Drive/Bhuti_data/'):
  annots.append(loadmat('drive/My Drive/Bhuti_data/'+filename))

emg = []
for files in annots:
  emg.append(pd.DataFrame(files['emg']))

for i in range(len(emg)):
  emg[i]['stimulus'] = annots[i]['stimulus']
  emg[i]['restimulus'] = annots[i]['restimulus']
  emg[i]['repetition'] = annots[i]['repetition']
  emg[i]['rerepetition'] = annots[i]['rerepetition']

new_emg=[]
for i in range(len(emg)):
  new_emg.append(emg[i][::4])

normal_emg = []
for i in range(len(new_emg)):
  normal_emg.append((new_emg[i]-new_emg[i].mean())/new_emg[i].std())
  normal_emg[i]['stimulus'] = new_emg[i]['stimulus']
  normal_emg[i]['restimulus'] = new_emg[i]['restimulus']
  normal_emg[i]['repetition'] = new_emg[i]['repetition']
  normal_emg[i]['rerepetition'] = new_emg[i]['rerepetition']

len(normal_emg[0][0]),len(emg[0][0])

sm_emg = []
for i in range(len(normal_emg)):
  sm_emg.append(normal_emg[i][normal_emg[i]['stimulus'].between(0, 6, inclusive=False)])

for i in range(len(sm_emg)):
  del sm_emg[i]['restimulus']
  del sm_emg[i]['repetition']
  del sm_emg[i]['rerepetition']

len(sm_emg[2][sm_emg[2]['stimulus']==1]),len(sm_emg[2][sm_emg[2]['stimulus']==3])

from scipy import signal

from sklearn.decomposition import PCA

pca = PCA(n_components=25)


# st_train = []
# y_train = []
# st_test = []
# y_test = []

# for i in range(len(sm_emg)):
#   for j in range(len(np.unique(sm_emg[i]['stimulus']))):
#       temp = sm_emg[i][sm_emg[i]['stimulus']==j+1].values
#       # sti = temp[:,-1]
#       temp = temp[:,:-1]
#
#       for k in range(12):
#         frequencies_samples, time_segment_sample, spectrogram_of_vector = signal.spectrogram(x=temp[:,k],nperseg = 256, noverlap = 184,window="hann",scaling="spectrum")
#         #only keep 95 points in frequency
#         #do pca and select first 25 pcs
#         if i<27:
#           st_train.append(spectrogram_of_vector)
#           y_train.append(j+1)
#         else:
#           st_test.append(spectrogram_of_vector)
#           y_test.append(j+1)
#   # break
#
# np.save('drive/My Drive/Bhuti_Colab/final_data',st_emg)
#
# np.save('drive/My Drive/Bhuti_Colab/final_data_labels',y')




x_train = np.load('drive/My Drive/Bhuti_Colab/final_data_train_1.0.npy')
y_train = np.load('drive/My Drive/Bhuti_Colab/final_data_train_labels_1.0.npy')

x_test = np.load('drive/My Drive/Bhuti_Colab/final_data_test.npy')
y_test = np.load('drive/My Drive/Bhuti_Colab/final_data_test_labels.npy')


y_train = tf.one_hot(y_train,5)
y_test = tf.one_hot(y_test,5)

# chunks_x = []
# chunks_y = []
#
# z = 0
# for x,y in zip(x_train,y_train):
#   # print(type(x),y)
#   if z%255546==0 and z!=0:
#     name_x = 'drive/My Drive/Bhuti_Colab/final_data_train_' + str(z//255546)
#     name_y = 'drive/My Drive/Bhuti_Colab/final_data_train_labels_' + str(z//255546)
#     np.save(name_x,chunks_x)
#     np.save(name_y,chunks_y)
#
#     # chunks_x[z/255546] = temp_x
#     # chunks_y[z/255546] = temp_y
#     chunks_x = []
#     chunks_y = []
#     print(z)
#
#   chunks_x.append(x)
#   chunks_y.append(y)
#   z+=1
#   # break

from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import concatenate

inputs = Input(shape=(15,13,12))
x = Conv2D(32, (3, 3), strides=2 ,activation='relu',kernel_initializer=tf.keras.initializers.glorot_uniform(seed=42))(inputs)
x = MaxPooling2D((1, 2), strides=(2, 2))(x)
# x = BatchNormalization(momentum=0.9)(x)
x = Dropout(rate=0.5)(x)

x = Conv2D(64, (3,3), strides=3,activation='relu',kernel_initializer=tf.keras.initializers.glorot_uniform(seed=42))(inputs)
x = MaxPooling2D((2, 2), strides=(2, 2))(x)
# x = BatchNormalization(momentum=0.9)(x)
x = Dropout(rate=0.5)(x)

x = Flatten()(x)
x = Dense(800)(x)
x = Dropout(rate=0.5)(x)

x = Dense(5)(x)
x = Activation("softmax")(x)

model_fft = Model(inputs, x, name="plakshanet_fft")
checkpoint_path = "drive/My Drive/Bhuti_new_data/cp_{epoch:04d}_fft.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path.format(epoch=0))
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1,save_freq='epoch')

model_fft.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.99, amsgrad=True),loss='categorical_crossentropy',metrics=['accuracy'])

history = model_fft.fit(x_train,y_train,epochs=100, batch_size=128, steps_per_epoch=x_train.shape[0]/128 , verbose=1, shuffle=True, callbacks=[cp_callback],use_multiprocessing=True)
model_fft.save('drive/My Drive/Bhuti_new_data/my_model_fft.h5')

history2 = model_fft.fit(x_train,y_train,epochs=200, batch_size=128, steps_per_epoch=x_train.shape[0]/128 , initial_epoch=100, verbose=1, shuffle=True, callbacks=[cp_callback],use_multiprocessing=True)
model_fft.save('drive/My Drive/Bhuti_new_data/my_model_fft_2.h5')

x_test = np.load('drive/My Drive/Bhuti_Colab/Bhuti_fft_data/spectrogram_test.npy')
y_test = np.load('drive/My Drive/Bhuti_Colab/Bhuti_fft_data/spectrogram_test_labels.npy')


model_fft.evaluate(x_test,y_test,verbose=1)
