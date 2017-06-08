# import matplotlib.pyplot as plt
import numpy as np
import time
import csv
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
# np.random.seed(1234)


SEQ = 10


def load_split_data(path_to_dataset, sequence_length, data_dim):
    datasets = np.genfromtxt(path_to_dataset, delimiter=",", skip_header=1)
    print("Data loaded from csv. Formatting...")
    x = [data[:-1] for data in datasets]
    y = [data[-1] for data in datasets]
    feature = []
    for index in range(len(datasets) - sequence_length):
        feature.append(x[index: index + sequence_length])
        target.append(y[index: index + sequence_length])
        
    feature = np.array(feature)
    X_data = np.reshape(feature, (feature.shape[0], feature.shape[1], data_dim))
    return [X_data, y[SEQ:]]

def build_model(sequence_length, data_dim, num_classes):
    model = Sequential()

    model.add(LSTM(
        64,
        input_shape=(sequence_length, data_dim),
        return_sequences=True))
    model.add(Dropout(0.5))

    model.add(LSTM(
        64,
        return_sequences=False))
    model.add(Dropout(0.5))

    model.add(Dense(
        num_classes))
    model.add(Activation("softmax"))

    start = time.time()
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="rmsprop",
                  metrics=['accuracy'])
    print("Compilation Time : ", time.time() - start)
    return model

def build_ann():
    model = Sequential()
    model.add(Dense(1000, activation='relu', input_dim=644))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))
    start = time.time()
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="rmsprop",
                  metrics=['accuracy'])
    print("Compilation Time : ", time.time() - start)
    return model

def run_network(model=None, data=None):
    global_start_time = time.time()
    epochs = 60
    sequence_length = SEQ
    data_dim = 6
    num_classes = 3

    if data is None:
        print('Loading data... ')
        X_train, y_train = load_split_data('sp500_train.csv', sequence_length, data_dim)
        X_test, y_test = load_split_data('sp500_test.csv', sequence_length, data_dim)
    else:
        X_train, y_train, X_test, y_test = data

    print('\nData Loaded. Compiling...\n')

    if model is None:
        model = build_model(sequence_length, data_dim, num_classes)

    try:
        model.fit(
            X_train, y_train,
            batch_size=512, epochs=epochs, shuffle=False, validation_split=0.05)
        predicted = model.predict(X_test)
        predicted = np.reshape(predicted, (predicted.size,))
    except KeyboardInterrupt:
        print('Training duration (s) : ', time.time() - global_start_time)
        return model, y_test, 0

    print('Training duration (s) : ', time.time() - global_start_time)
    score, acc = model.evaluate(X_test, y_test)
    print('\nAccuracy: ', acc*100)#, '\nScore: ', score 
    model.save('test.h5')


if __name__ == '__main__':
    run_network()