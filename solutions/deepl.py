# Deepl
# training data:
#   two approaches: - give it a pre-defined set of all 'correct' moves in a given environment
#                   - give it the created action data from the genetic alg
#
# further directions:
#    - different data entry for each correct action for a given env #is it healthy that we have multiple outputs for the same input?
#    - multi label classification
# TODO what kind of model do we want?
# TODO weights of different actions?, stepping away from a lion
#  is almost always better than stepping towards,
#  as we limit our freedom of movement

# 1. Small model: small dataset,

import os
import numpy as np

from game_environment.scenario import Environment, TileState, Step
import tensorflow as tf
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, Input
from keras._tf_keras.keras.metrics import categorical_crossentropy
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

PATH_TO_SIMPLE_GENERATED_LEARNING_DATASET = os.path.join("..", "res", "gt_dataset.txt")
PATH_TO_SIMPLE_GENERATED_EVALUATION_DATASET = os.path.join("..", "res", "ge_dataset.txt")

def _get_dataset_from_source(source):
    samples = []
    labels = []
    f = open(source, "+r")
    lines = f.readlines()
    for line in lines:
        env, action = line.split(';')
        tile_states = env.split(',')
        samples.append([
            int(tile_states[0]),
            int(tile_states[1]),
            int(tile_states[2]),
            int(tile_states[3])
        ])
        label = np.zeros(4,)
        label[int(action)] = 1
        labels.append(label)
    f.close()
    return samples, labels


class SmallClassDeepl:
    """Model trained on small data, using traditional classification"""

    def __init__(self, optimizer = Adam, learning_rate = 0.02, loss = 'categorical_crossentropy', metrics = ['accuracy']):
        #model
        self.model = Sequential([ #TODO
            Input(shape=(4,)),
            Dense(units=12, activation='relu'),
            Dense(units=48, activation='relu'),
            Dense(units=4, activation='softmax')
        ])
        self.model.compile(
            optimizer=optimizer(learning_rate=learning_rate),
            loss=loss, metrics=metrics)
        print("compiled model")

    def learn(self, path, batch:int = 100, epochs: int = 50, validation_split = 0.05):
        samples, labels = _get_dataset_from_source(path)
        train_labels = np.array(labels)
        train_samples = np.array(samples)
        train_labels, train_samples = shuffle(train_labels, train_samples)

        print(f"{train_samples[0]}, {train_labels[0]}" )

        # scaler = MinMaxScaler(feature_range=(0,1))
        # scaled_train_samples = scaler.fit_transform(train_samples) #TODOmight need scaling?

        self.model.fit(x=train_samples, y=train_labels,
                       validation_split=validation_split,
                       batch_size=batch, epochs=epochs,
                       verbose=2)

    def evaluate(self, path, batch:int = 10):
        samples, labels = _get_dataset_from_source(path)
        eval_labels = np.array(labels)
        eval_samples = np.array(samples)
        eval_labels, eval_samples = shuffle(eval_labels, eval_samples)
        #scaling? shouldn't fit it

        predictions = self.model.predict(
            x=eval_samples,
            batch_size=batch,
            verbose=0
        )

        rounded_predictions = np.argmax(predictions, axis=-1)

        #temp
        hits = 0
        for i in range(len(rounded_predictions)):
            if rounded_predictions[i] == np.argmax(eval_labels[i]):
                hits += 1
        print(f"chose correctly {hits/len(eval_labels)*100:.2f}% of the time")
        #TODO stats


    def test(self):
        #TODO play ONE game
        pass

    def describe(self):
        print(self.model.summary())


SmallClassDeepl(Adam, 0.001, 'categorical_crossentropy', ['accuracy'])
# 2. Big model: bigger dataset