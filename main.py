from dataset import Testset, DatasetHandler, Devset

import numpy as np

from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.layers import Dense, Flatten, Dropout, Conv1D, MaxPooling1D, Conv2D, MaxPooling2D
from keras.models import Model, Input, Sequential
from keras.models import  load_model

from sklearn.model_selection import train_test_split

MAXLEN = 100
CHANNELS = 1
EPOCHS = 10
BATCH_SIZE = 32
BUCKETS = 26
FILTERS = 32
KERNEL_SIZE = 3



def model_compile():
    model = Sequential()

    model.add(Conv2D(FILTERS, (3, 3), input_shape=(MAXLEN, BUCKETS, CHANNELS), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(FILTERS, (3, 3), input_shape=(MAXLEN, BUCKETS, CHANNELS), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    """
    model.add(Conv1D(FILTERS, KERNEL_SIZE, input_shape=(MAXLEN, BUCKETS), activation="relu"))
    model.add(MaxPooling1D())
    """

    """
    model.add(Flatten(input_shape=(MAXLEN, BUCKETS)))

    model.add(Conv2D(32, (3,3), input_shape=(MAXLEN, BUCKETS), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    """

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    return model

def train():

    testset = Testset("data/train.json")
    handler = DatasetHandler(testset.dataset)

    # Padding to force the outer array having the all the same length

    X = sequence.pad_sequences(handler.features, maxlen=MAXLEN, value=0.0, dtype='float', padding="post") # , padding="post"
    y = to_categorical(np.asarray(handler.labels))

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, shuffle=True)
    
    X_tr = X_tr.reshape(X_tr.shape[0], MAXLEN, BUCKETS, CHANNELS)
    X_val = X_val.reshape(X_val.shape[0], MAXLEN, BUCKETS, CHANNELS)

    model = model_compile()

    model.fit(X_tr, y_tr,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(X_val, y_val))

    #model.save("ser_model.h5")


def predict():
    devset = Devset("data/dev.json")

    ser_model = load_model("ser_model.h5")

    predictions = {}

    for d in devset.dataset:
        features = devset.dataset[d]['features']
        id = d

        test_X = sequence.pad_sequences([features], maxlen=MAXLEN, value=0.0, dtype='float', padding="post")
        test_X = test_X.reshape(test_X.shape[0], MAXLEN, BUCKETS, CHANNELS)
        pred = ser_model.predict_classes(test_X)
        
        labels = getLabel(pred[0])

        predictions[d] = {}
        predictions[d]['activation'] = labels['activation']
        predictions[d]['valence'] = labels['valence']

    print(predictions["0"])
    devset.write("data/pred.json", predictions)

def getLabel(p):
    if p == 0:
        return {"activation": 0, "valence": 0}
    elif p == 1:
        return {"activation": 1, "valence": 0}
    elif p == 2:
        return {"activation": 0, "valence": 1}
    elif p == 3:
        return {"activation": 1, "valence": 1}



train()

#predict()





