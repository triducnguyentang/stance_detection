# import matplotlib.pyplot as plt
import numpy as np
import time
import csv
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
# np.random.seed(1234)


SEQ = 10


def load_dataset(path_to_dataset, sequence_length, data_dim):

    # max_values = ratio * 2049280
    datasets = np.genfromtxt(path_to_dataset, delimiter=",", skip_header=1)
    # print("************************")
    # used smote to balance samples
    x = [data[:-1] for data in datasets]
    y = [data[-1] for data in datasets]

    print("Data loaded from csv. Formatting...")

    feature = []
    target = []
    for index in range(len(datasets) - sequence_length):
        feature.append(x[index: index + sequence_length])
        target.append(y[index: index + sequence_length])
        
    feature = np.array(feature)
    target = np.array(target)

    # result_mean = result.mean()
    # result -= result_mean
    # print "Shift : ", result_mean
    # print "Data  : ", result.shape

    row = int(round(0.8 * feature.shape[0]))
    # train = result[:row, :]
    # np.random.shuffle(train)
    X_train = feature[:row, :]
    y_train = target[:row, :]
    # X_train = feature[:, :-1]
    # y_train = train[:, -1]
    X_test = feature[row:, :]
    y_test = target[row:, :]
    print(".........", X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    # X_train = np.reshape(X_train, (X_train.shape[0], sequence_length-1, data_dim))
    # X_test = np.reshape(X_test, (X_test.shape[0], sequence_length, data_dim))
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], data_dim))
    # y_train = np.reshape(y_train, ()
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], data_dim))
    return [X_train, y_train, X_test, y_test]

def load_split_data(path_to_dataset, sequence_length, data_dim):
    # max_values = ratio * 2049280
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
    model.add(Dense(1000, activation='relu', input_dim=144))
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
    # ratio = 0.5
    sequence_length = SEQ
    data_dim = 6
    num_classes = 3
    # path_to_dataset = 'consumer_staple.csv'

    if data is None:
        print('Loading data... ')
        # X_train, y_train, X_test, y_test = load_dataset(
            # path_to_dataset, sequence_length, data_dim)
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
    # return model, y_test, 
    model.save('test.h5')


if __name__ == '__main__':
    run_network()