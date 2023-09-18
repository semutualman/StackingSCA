from exploit_pred import *
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
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
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

d_in='D:/Methodology-for-efficient-CNN-architectures-in-SCA-master/DPA-contest v4/second_data/'
second_level_train_set=np.load(d_in + "second_train.npy")
test_nfolds_sets=np.load(d_in + "second_test.npy")
Y_profiling=np.load(d_in + "Y_profiling.npy")
Bestpred = np.load(d_in+"Bestpred.npy")
predictions1=np.load(d_in+"predict.npy")
predictions2=np.load(d_in+"predict1.npy")
predictions3=np.load(d_in+"predict3.npy")
prediction4=np.load(d_in+"predict4.npy")
predictions5=np.load("E:/EnsembleSCA-master/dpa/data/dpa.npy")

def run_cnn(X_profiling, Y_profiling, X_validation, Y_validation):
    X_profiling = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1], 1))
    X_validation = X_validation.reshape((X_validation.shape[0], X_validation.shape[1], 1))

    mini_batch = random.randrange(500, 1000, 100)
    learning_rate = random.uniform(0.0001, 0.001)
    activation = ['relu', 'tanh', 'elu', 'selu'][random.randint(0, 3)]
    dense_layers = random.randrange(2, 8, 1)
    neurons = random.randrange(500, 800, 100)
    conv_layers = random.randrange(1, 2, 1)
    filters = random.randrange(8, 32, 4)
    kernel_size = random.randrange(10, 20, 2)
    stride = random.randrange(5, 10, 5)

    Y_profiling = to_categorical(Y_profiling, num_classes=256)
    Y_validation = to_categorical(Y_validation, num_classes=256)

    model = cnn_random(256, 256, 'relu', 512, conv_layers,
                                       filters,
                                       kernel_size, stride, 2, learning_rate)
    model.fit(
        x=X_profiling,
        y=Y_profiling,
        batch_size=400,
        verbose=1,
        epochs=50,
        shuffle=True,
        validation_data=(X_validation, Y_validation),
        callbacks=[])

    backend.clear_session()

    return model


def cnn_random(classes, number_of_samples, activation, neurons, conv_layers, filters, kernel_size, stride, layers, learning_rate):
        model = Sequential()
        for layer_index in range(conv_layers):
            if layer_index == 0:
                model.add(Conv1D(filters=filters, kernel_size=kernel_size, strides=stride, activation='relu', padding='same',
                                 input_shape=(number_of_samples, 1)))
            else:
                model.add(Conv1D(filters=filters, kernel_size=kernel_size, strides=stride, activation='relu', padding='same'))
            # model.add(MaxPooling1D(pool_size=1))

        model.add(Flatten())
        for layer_index in range(layers):
            model.add(Dense(neurons, activation=activation, kernel_initializer='he_uniform', bias_initializer='zeros'))

        model.add(Dense(classes, activation='softmax'))
        model.summary()
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=learning_rate), metrics=['accuracy'])

        return model


def run_mlp(X_profiling, Y_profiling, X_validation, Y_validation):
    mini_batch = random.randrange(500, 1000, 100)
    learning_rate = random.uniform(0.0001, 0.001)
    activation = ['relu', 'tanh', 'elu', 'selu'][random.randint(0, 3)]
    layers = random.randrange(2, 8, 1)
    neurons = random.randrange(500, 800, 100)

    Y_profiling = to_categorical(Y_profiling, num_classes=256)
    Y_validation = to_categorical(Y_validation, num_classes=256)

    model = mlp_random(256, 256, 'relu', 512, 2, learning_rate)
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


model1 = run_mlp(second_level_train_set[:4000], Y_profiling[:4000],second_level_train_set[4000:], Y_profiling[4000:])
# predictions = model1.predict(test_nfolds_sets)
model2 = run_cnn(second_level_train_set[:4000], Y_profiling[:4000],second_level_train_set[4000:], Y_profiling[4000:])
# test_nfolds_sets = test_nfolds_sets.reshape((test_nfolds_sets.shape[0], test_nfolds_sets.shape[1], 1))
# predictions1 = model2.predict(test_nfolds_sets)

root = "D:/Methodology-for-efficient-CNN-architectures-in-SCA-master/DPA-contest v4/"
DPAv4_data_folder = root+"DPAv4_dataset/"
DPAv4_trained_models_folder = root+"DPAv4_trained_models"
history_folder = root+"training_history/"
predictions_folder = root+"model_predictions/"

# Choose the name of the model
nb_epochs = 50
batch_size = 50
input_size = 4000
learning_rate = 1e-3
nb_traces_attacks = 50
nb_attacks = 100
real_key = np.load(DPAv4_data_folder + "key.npy")
mask = np.load(DPAv4_data_folder + "mask.npy")
att_offset = np.load(DPAv4_data_folder + "attack_offset_dpav4.npy")

# Load the profiling traces
plt_attack = np.load(DPAv4_data_folder+'attack_plaintext_dpav4.npy')


model_name = "ASCAD_desync0"
avg_rankml = perform_attacks(nb_traces_attacks, predictions1, nb_attacks, plt=plt_attack, key=real_key, mask=mask, offset=att_offset, byte=0, filename=model_name)

avg_rankcn = perform_attacks(nb_traces_attacks, predictions3, nb_attacks, plt=plt_attack, key=real_key, mask=mask, offset=att_offset, byte=0, filename=model_name)

best = perform_attacks(nb_traces_attacks, Bestpred, nb_attacks, plt=plt_attack, key=real_key, mask=mask, offset=att_offset, byte=0, filename=model_name)

# predict1 = perform_attacks(nb_traces_attacks, predictions2, nb_attacks, plt=plt_attack, key=real_key, mask=mask, offset=att_offset, byte=0, filename=model_name)

# predict2 = perform_attacks(nb_traces_attacks, prediction4, nb_attacks, plt=plt_attack, key=real_key, mask=mask, offset=att_offset, byte=0, filename=model_name)

print("\n t_GE = ")
print(np.where(avg_rankml <= 0))
# print(np.where(avg_rankcn <= 0))

# plot.subplot(1, 2, 1)
plot.rcParams['figure.figsize'] = (20, 6)
plot.ylim(-5, 200)
plot.xlim(-1, 50)
plot.grid(True)
plot.plot(avg_rankml,  label='Stacking-SCA Ensembles 5 models')
plot.plot(best,  label='Stacking-SCA Best Single Model')
plot.plot(avg_rankcn, label='Stacking-SCA Ensembles 10 models')
plot.plot(predictions5, 'k', label='Bagging Ensembles 10 models')
plot.xlabel('Number of traces', fontsize=24)
plot.ylabel('Guessing Entorpy', fontsize=24)
plot.yticks(fontproperties='Times New Roman', size=24) # 设置大小及加粗
plot.xticks(fontproperties='Times New Roman', size=24)
plot.legend(fontsize=20)
# plot.subplot(1, 2, 2)
# plot.ylim(-5, 50)
# plot.xlim(-1, 15)
# plot.grid(True)
# plot.plot(predict1, 'g', label='Ensembles 5 models')
# plot.plot(best, 'b', label='Attcak 1 model')
# plot.plot(predict2, 'r', label='Ensembles 10 models')
# plot.xlabel('Number of traces')
# plot.ylabel('Rank')
# plot.legend()
plot.show()


print("\n############### Attack on Test Set Done #################\n")
