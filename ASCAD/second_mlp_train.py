import os
import os.path
import sys
import time
import pickle
import h5py
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Input, Conv1D, AveragePooling1D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn import preprocessing
from tensorflow.python.keras import backend

from exploit_pred import *
from clr import OneCycleLR
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

d_in = 'D:/Methodology-for-efficient-CNN-architectures-in-SCA-master/ASCAD/N0=0/ASCAD_dataset/second_dataset/'
second_level_train_set=np.load(d_in + "second_level_train_set.npy")
test_nfolds_sets=np.load(d_in + "test_nfolds_sets.npy")
Y_profiling=np.load(d_in + "Y_profiling.npy")
Bestpred = np.load(d_in+"Best.npy")
predictions1=np.load(d_in+"predictions.npy")
predictions2=np.load("E:/EnsembleSCA-master/ascad/data/ASCAD.npy")

def run_mlp(X_profiling, Y_profiling, X_validation, Y_validation):
    mini_batch = random.randrange(500, 1000, 100)
    learning_rate = random.uniform(0.0001, 0.001)
    activation = ['relu', 'tanh', 'elu', 'selu'][random.randint(0, 3)]
    layers = 2
    neurons = 512

    Y_profiling = to_categorical(Y_profiling, num_classes=256)
    Y_validation = to_categorical(Y_validation, num_classes=256)

    model = mlp_random(256, 256, 'relu', neurons, layers, 0.00001)
    model.fit(
        x=X_profiling,
        y=Y_profiling,
        batch_size=400,
        verbose=1,
        epochs=50,
        shuffle=True,
        validation_data=(X_validation, Y_validation),
        callbacks=[])

    return model


def mlp_random(classes, number_of_samples, activation, neurons, layers, learning_rate):
    model = Sequential()
    model.add(BatchNormalization(input_shape=(number_of_samples,)))
    for l_i in range(layers):
        model.add(Dense(neurons, activation=activation, kernel_initializer='he_uniform', bias_initializer='zeros'))

    model.add(Dense(classes, activation='softmax'))
    model.summary()
    optimizer = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


def run_cnn(X_profiling, Y_profiling, X_validation, Y_validation):
    X_profiling = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1], 1))
    X_validation = X_validation.reshape((X_validation.shape[0], X_validation.shape[1], 1))
    Y_profiling = to_categorical(Y_profiling, num_classes=256)
    Y_validation = to_categorical(Y_validation, num_classes=256)

    mini_batch = random.randrange(500, 1000, 100)
    learning_rate = random.uniform(0.0001, 0.001)
    activation = ['relu', 'tanh', 'elu', 'selu'][random.randint(0, 3)]
    dense_layers = random.randrange(2, 8, 1)
    neurons = random.randrange(500, 800, 100)
    conv_layers = random.randrange(1, 2, 1)
    filters = random.randrange(8, 32, 4)
    kernel_size = random.randrange(10, 20, 2)
    stride = random.randrange(5, 10, 5)

    model = cnn_random(256, 256, activation, 512, conv_layers,
                       filters,kernel_size, stride, 2, learning_rate)
    model.fit(
        x=X_profiling,
        y=Y_profiling,
        batch_size=400,
        verbose=1,
        epochs=15,
        shuffle=True,
        validation_data=(X_validation, Y_validation),
        callbacks=[])

    backend.clear_session()

    return model


def cnn_random(classes, number_of_samples, activation, neurons, conv_layers, filters, kernel_size, stride, layers,
               learning_rate):
    model = Sequential()
    for layer_index in range(conv_layers):
        if layer_index == 0:
            model.add(
                Conv1D(filters=filters, kernel_size=kernel_size, strides=stride, activation='relu', padding='same',
                       input_shape=(number_of_samples, 1)))
        else:
            model.add(
                Conv1D(filters=filters, kernel_size=kernel_size, strides=stride, activation='relu', padding='same'))
        # model.add(MaxPooling1D(pool_size=1))

    model.add(Flatten())
    for layer_index in range(layers):
        model.add(Dense(neurons, activation=activation, kernel_initializer='he_uniform', bias_initializer='zeros'))

    model.add(Dense(classes, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=learning_rate), metrics=['accuracy'])

    return model

#五个模型

model1 = run_mlp(second_level_train_set[:40000], Y_profiling[:40000],second_level_train_set[40000:], Y_profiling[40000:])
predictions=model1.predict(test_nfolds_sets)

# 10个模型

root = "D:/Methodology-for-efficient-CNN-architectures-in-SCA-master/ASCAD/N0=0/"
ASCAD_data_folder = root + "ASCAD_dataset/"
ASCAD_trained_models_folder = root + "ASCAD_trained_models"
history_folder = root + "training_history/"
predictions_folder = root + "model_predictions/"

# Choose the name of the model
nb_epochs = 50
batch_size = 50
input_size = 700
learning_rate = 5e-3
nb_traces_attacks = 300
nb_attacks = 100
real_key = np.load(ASCAD_data_folder + "key.npy")

start = time.time()

# Load the profiling traces
in_file = h5py.File(ASCAD_data_folder+'ASCAD.h5', "r")
plt_attack=in_file['Attack_traces/metadata']['plaintext']

model_name = "ASCAD_desync0"
avg_rank = perform_attacks(nb_traces_attacks, predictions, nb_attacks, plt=plt_attack, key=real_key, byte=2,
                           filename=model_name)

avg_rank1 = perform_attacks(nb_traces_attacks, predictions1, nb_attacks, plt=plt_attack, key=real_key, byte=2,
                            filename=model_name)

avg_rank2 = perform_attacks(nb_traces_attacks, Bestpred, nb_attacks, plt=plt_attack, key=real_key, byte=2,
                            filename=model_name)

# # plot.subplot(1, 2, 1)
# # plot.ylim(-5, 120)
# # plot.xlim(-5,300)
# # plot.rcParams['figure.figsize'] = (20, 10)
# # plot.ylim(-5, 150)
# # plot.grid(True)
# # plot.plot(avg_rank, 'g', label='Ensemble 5 models')
# # plot.plot(avg_rank2, 'r', label='Attack 1 model')
# # plot.plot(avg_rank1, 'b', label='Ensemble 10 models')
# # plot.xlabel('Number of traces')
# # plot.ylabel('Rank')
# # plot.show()
plot.rcParams['figure.figsize'] = (20, 6)
plot.ylim(-5, 150)
plot.grid(True)
# plot.plot(avg_rank, 'b')
plot.plot(avg_rank, 'b', label='Stacking-SCA Ensembles 5 models')
plot.plot(avg_rank2, 'r', label='Stacking-SCA Best Single Model')
plot.plot(avg_rank1, 'g', label='Stacking-SCA Ensembles 10 models')
plot.plot(predictions2[0:300], 'k', label='Bagging Ensembles 10 models')
plot.xlabel('Number of traces',fontsize=24)
plot.ylabel('Guessing Entorpy',fontsize=24)
plot.yticks(fontproperties='Times New Roman', size=24)#设置大小及加粗
plot.xticks(fontproperties='Times New Roman', size=24)
plot.legend(fontsize=20)
plot.show()
print("\n t_GE = ")
print("\n############### Attack on Test Set Done #################\n")
