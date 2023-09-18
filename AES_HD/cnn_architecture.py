import os.path
import sys
import time
import pickle
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Input, Conv1D, AveragePooling1D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn import preprocessing
from keras import losses
from exploit_pred import *
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
### Scripts based on ASCAD github : https://github.com/ANSSI-FR/ASCAD

def check_file_exists(file_path):
        if os.path.exists(file_path) == False:
                print("Error: provided file path '%s' does not exist!" % file_path)
                sys.exit(-1)
        return

def shuffle_data(profiling_x,label_y):
    l = list(zip(profiling_x,label_y))
    random.shuffle(l)
    shuffled_x,shuffled_y = list(zip(*l))
    shuffled_x = np.array(shuffled_x)
    shuffled_y = np.array(shuffled_y)
    return (shuffled_x, shuffled_y)

### CNN Best model
def cnn_architecture(input_size=1250,learning_rate=0.00001,classes=256):
        
        # Designing input layer
        input_shape = (input_size,1)
        img_input = Input(shape=input_shape)

        # 1st convolutional block
        x = Conv1D(2, 1, kernel_initializer='he_uniform', activation='selu', padding='same', name='block1_conv1')(img_input)
        x = BatchNormalization()(x)
        x = AveragePooling1D(2, strides=2, name='block1_pool')(x)

        x = Flatten(name='flatten')(x)

        # Classification layer
        x = Dense(2, kernel_initializer='he_uniform', activation='selu', name='fc1')(x)

        # Logits layer
        x = Dense(classes, activation='softmax', name='predictions')(x)

        # Create model
        inputs = img_input
        model = Model(inputs, x, name='aes_hd_model')
        optimizer = Adam(lr=learning_rate)
        model.compile(loss='categorical_crossentropy',optimizer=optimizer, metrics=['accuracy'])
        return model

def Stacking(X_profiling, Y_profiling, save_file_name):
    second_level_train_set = np.zeros((50000, 256))
    second_level_test_set = np.zeros((5, 25000, 256))
    test_nfolds_sets = np.zeros((1,25000,256))
    kf = KFold(n_splits=5)
    sum=second_level_train_set.shape[0]
    sum1=second_level_test_set.shape[0]
    for j, (train_index, test_index) in enumerate(kf.split(X_profiling, Y_profiling)):
        x_tra, y_tra = X_profiling[train_index], Y_profiling[train_index]
        x_tst, y_tst = X_profiling[test_index], Y_profiling[test_index]
        meta_train, meta_test,history = train_model(x_tra, y_tra, x_tst, y_tst, X_attack, model, save_file_name, epochs=50,
                                    batch_size=100)
        second_level_train_set[10000 * j:10000 * (j + 1)] = meta_train
        second_level_test_set[j] = meta_test
        print("第", j, "次Stacking")
    test_nfolds_sets[:] = np.mean(second_level_test_set, 0)
    model1=run_mlp(second_level_train_set[:40000], Y_profiling[:40000], second_level_train_set[40000:], Y_profiling[40000:])
    test_nfolds_sets=test_nfolds_sets.reshape([25000, 256])
    predictions=model1.predict(test_nfolds_sets)
    d_in='D:/Methodology-for-efficient-CNN-architectures-in-SCA-master/AES_HD/second_data/'
    np.save(d_in+"second_level_train_set.npy",second_level_train_set)
    np.save(d_in + "test_nfolds_sets.npy", test_nfolds_sets)
    np.save(d_in + "Y_profiling.npy", Y_profiling)

    return predictions

def Stacking1(X_profiling, Y_profiling, save_file_name):
    second_level_train_set = np.zeros((50000, 256))
    second_level_test_set = np.zeros((10, 25000, 256))
    test_nfolds_sets = np.zeros((1,25000,256))
    kf = KFold(n_splits=10)
    sum=second_level_train_set.shape[0]
    sum1=second_level_test_set.shape[0]
    for j, (train_index, test_index) in enumerate(kf.split(X_profiling, Y_profiling)):
        x_tra, y_tra = X_profiling[train_index], Y_profiling[train_index]
        x_tst, y_tst = X_profiling[test_index], Y_profiling[test_index]
        meta_train, meta_test,history = train_model(x_tra, y_tra, x_tst, y_tst, X_attack, model, save_file_name, epochs=50,
                                    batch_size=100)
        second_level_train_set[5000 * j:5000 * (j + 1)] = meta_train
        second_level_test_set[j] = meta_test
        print("第",j,"次")
    test_nfolds_sets[:] = np.mean(second_level_test_set, 0)
    model1=run_mlp(second_level_train_set[:40000], Y_profiling[:40000], second_level_train_set[40000:], Y_profiling[40000:])
    model2 = run_cnn(second_level_train_set[:40000], Y_profiling[:40000], second_level_train_set[40000:],
                     Y_profiling[40000:])

    test_nfolds_sets=test_nfolds_sets.reshape([25000, 256])
    predictions2=model1.predict(test_nfolds_sets)
    d_in='D:/Methodology-for-efficient-CNN-architectures-in-SCA-master/AES_HD/second_data/'
    np.save(d_in+"second_level_train_set1.npy",second_level_train_set)
    np.save(d_in + "test_nfolds_sets1.npy", test_nfolds_sets)

    return predictions2

def run_mlp(X_profiling, Y_profiling, X_validation, Y_validation):
        mini_batch = random.randrange(500, 1000, 100)
        learning_rate = random.uniform(0.0001, 0.001)
        activation = ['relu', 'tanh', 'elu', 'selu'][random.randint(0, 3)]
        layers = 2
        neurons = 512

        Y_profiling = to_categorical(Y_profiling, num_classes=256)
        Y_validation = to_categorical(Y_validation, num_classes=256)

        model = mlp_random(256, 256, activation, neurons, layers, learning_rate)
        model.fit(
            x=X_profiling,
            y=Y_profiling,
            batch_size=400,
            verbose=1,
            epochs=10,
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

    model = cnn_random(256, 256, activation,512 , conv_layers,
                                       filters,
                                       kernel_size, stride, 2, learning_rate)
    model.fit(
        x=X_profiling,
        y=Y_profiling,
        batch_size=400,
        verbose=1,
        epochs=10,
        shuffle=True,
        validation_data=(X_validation, Y_validation),
        callbacks=[])

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

#### Training high
def train_model(X_profiling, Y_profiling, X_test, Y_test,X_attack, model, save_file_name, epochs=150, batch_size=100):
    check_file_exists(os.path.dirname(save_file_name))

    # Save model every epoch
    save_model = ModelCheckpoint(save_file_name)

    Reshaped_X_attack = X_attack.reshape((X_attack.shape[0], X_attack.shape[1], 1))
    Reshaped_X_profiling, Reshaped_X_test  = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1], 1)),X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    callbacks=[save_model] 
          
    history = model.fit(x=Reshaped_X_profiling, y=to_categorical(Y_profiling, num_classes=256), validation_data=(Reshaped_X_test, to_categorical(Y_test, num_classes=256)), batch_size=batch_size, verbose = 1, epochs=epochs, callbacks=callbacks)

    meta_train = model.predict(Reshaped_X_test)
    meta_test = model.predict(Reshaped_X_attack)

    return meta_train, meta_test, history


#################################################
#################################################

#####            Initialization            ######

#################################################
#################################################

# Our folders
root = "D:/Methodology-for-efficient-CNN-architectures-in-SCA-master/AES_HD/"
AESHD_data_folder = root+"AES_HD_dataset/"
AESHD_trained_models_folder = root+"AESHD_trained_models"
history_folder = root+"training_history/"
predictions_folder = root+"model_predictions/"

# Hyperparameters
nb_epochs = 20
batch_size = 256
input_size = 1250
learning_rate = 1e-3
nb_traces_attacks = 1500
nb_attacks = 100
real_key = [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]

start = time.time()

# Load the profiling traces
(X_profiling, Y_profiling), (X_attack, Y_attack), (ciphertext_profiling, ciphertext_attack) = (np.load(AESHD_data_folder + 'profiling_traces_AES_HD.npy'), np.load(AESHD_data_folder + 'profiling_labels_AES_HD.npy')), (np.load(AESHD_data_folder + 'attack_traces_AES_HD.npy'), np.load(AESHD_data_folder + 'attack_labels_AES_HD.npy')), (np.load(AESHD_data_folder + 'profiling_ciphertext_AES_HD.npy'), np.load(AESHD_data_folder + 'attack_ciphertext_AES_HD.npy'))

(X_profiling, Y_profiling) = shuffle_data(X_profiling, Y_profiling)

X_profiling = X_profiling.astype('float32')
X_attack = X_attack.astype('float32')

#Standardization and Normalization (between 0 and 1)
scaler = preprocessing.StandardScaler()
X_profiling = scaler.fit_transform(X_profiling)
X_attack = scaler.transform(X_attack)

scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
X_profiling = scaler.fit_transform(X_profiling)
X_attack = scaler.transform(X_attack)
X_attack = X_attack.reshape((X_attack.shape[0], X_attack.shape[1], 1))


#################################################
#################################################

####                Training               ######

#################################################
#################################################

# Choose your model
model = cnn_architecture(input_size=input_size, learning_rate=learning_rate)
model_name="AES_HD"

print('\n Model name = '+model_name)


print("\n############### Starting Training #################\n")

# Record the metrics

end=time.time()

print('Temps execution = %d'%(end-start))

print("\n############### Training Done #################\n")

# Save the metrics
# with open(history_folder + 'history_' + model_name, 'wb') as file_pi:
#      pickle.dump(history.history, file_pi)

#################################################
#################################################

####               Prediction              ######

#################################################
#################################################

print("\n############### Starting Predictions #################\n")

# predictions = model.predict(X_attack)
# # d_in='D:/Methodology-for-efficient-CNN-architectures-in-SCA-master/AES_HD/second_data/'
# # np.save(d_in+"best.npy",predictions)
# # predictions = Stacking(X_profiling, Y_profiling, AESHD_trained_models_folder + model_name)
predictions1= Stacking(X_profiling, Y_profiling, AESHD_trained_models_folder + model_name)

print("\n############### Predictions Done #################\n")

np.save(predictions_folder + 'predictions_' + model_name +'.npy', predictions1)

#################################################
#################################################

####            Perform attacks            ######

#################################################
#################################################
print("\n############### Starting Attack on Test Set #################\n")

avg_rank = perform_attacks(nb_traces_attacks, predictions1, nb_attacks, ciph=ciphertext_attack, key=real_key, byte=0, filename=model_name)

print("\n t_GE = ")
print(np.where(avg_rank<=0))

print("\n############### Attack on Test Set Done #################\n")
