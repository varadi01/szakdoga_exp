# Deepl
# training data:
#   two approaches: - give it a pre-defined set of all 'correct' moves in a given environment
#                   - give it the created action data from the genetic alg
#
# further directions:
#    - different data entry for each correct action for a given env #is it healthy that we have multiple outputs for the same input?
#    - multi label classification
# DEP what kind of model do we want?
# DEP weights of different actions?, stepping away from a lion
#  is almost always better than stepping towards,
#  as we limit our freedom of movement

# 1. Small model: small dataset,

import os

import keras._tf_keras.keras.models
import numpy as np

import tensorflow as tf
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, Input
from keras._tf_keras.keras.metrics import categorical_crossentropy
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer, LabelBinarizer
from sklearn.utils import shuffle

from game_environment.scenario import SimpleGame
from utils.scenario_utils import Step, ExtendedStep

PATH_TO_SIMPLE_GENERATED_LEARNING_DATASET = os.path.join("..", "res", "gt_dataset.txt")
PATH_TO_SIMPLE_GENERATED_EVALUATION_DATASET = os.path.join("..", "res", "ge_dataset.txt")
PATH_TO_EXTENDED_GENERATE_LEARNING_DATASET = os.path.join("..", "res", "gt_ex_dataset.txt")
PATH_TO_EXTENDED_GENERATE_EVALUATION_DATASET = os.path.join("..", "res", "ge_ex_dataset.txt")

# try normalizing, if results are lackluster

#check!!! train with list of good steps? possible?

def _get_dataset_from_source(source):
    samples = []
    labels = []
    f = open(source, "+r")
    lines = f.readlines()
    lb = LabelBinarizer()
    for line in lines:
        env, action = line.split(';')
        tile_states = env.split(',')
        samples.append([
            int(tile_states[0]),
            int(tile_states[1]),
            int(tile_states[2]),
            int(tile_states[3])
        ])
        label = np.zeros(5, ) #changed!
        label[int(action)] = 1
        labels.append(label)
    #labels = lb.fit_transform(labels) #dont work
    f.close()
    return samples, labels

def _get_multilabel_dataset_from_source(source):
    samples = []
    labels = []
    f = open(source, "+r")
    lines = f.readlines()
    mlb = MultiLabelBinarizer()
    for line in lines:
        env, actions = line.split(';')
        tile_states = env.split(',')
        samples.append([
            int(tile_states[0]),
            int(tile_states[1]),
            int(tile_states[2]),
            int(tile_states[3])
        ])
        label = actions.split(',')
        for i in range(len(label)):
            label[i] = int(label[i])
        labels.append(label)
    labels = mlb.fit_transform(labels)
    f.close()
    return samples, labels

def save_model(model: Sequential, name):
    path = os.path.join(os.getcwd(),'..', 'deepl', 'models_ex_final', name + '.keras') #changed!
    model.save(path)

def load_model(name):
    path = os.path.join(os.getcwd(),'..', 'deepl', 'models_ex_final', name + '.keras') #changed!
    return keras._tf_keras.keras.models.load_model(path)

class Deepl:
    """Model trained on small data, using traditional classification"""

    def __init__(self, optimizer = Adam, learning_rate = 0.02, loss = 'categorical_crossentropy', metrics = ['accuracy']):
        #model
        self.model = Sequential([
            Input(shape=(4,)),
            Dense(units=64, activation='relu'),
            Dense(units=64, activation='relu'),
            Dense(units=4, activation='softmax')
        ])
        self.model.compile(
            optimizer=optimizer(learning_rate=learning_rate),
            loss=loss, metrics=metrics)

    def learn(self, path, batch:int = 100, epochs: int = 50, validation_split = 0.05):
        samples, labels = _get_dataset_from_source(path)
        train_labels = np.array(labels)
        train_samples = np.array(samples)
        train_labels, train_samples = shuffle(train_labels, train_samples)

        print(f"{train_samples[0]}, {train_labels[0]}" )

        # scaler = MinMaxScaler(feature_range=(0,1))
        # scaled_train_samples = scaler.fit_transform(train_samples)

        self.model.fit(x=train_samples, y=train_labels,
                       validation_split=validation_split,
                       batch_size=batch, epochs=epochs,
                       verbose=2)

    def evaluate(self, path, batch:int = 10):
        #might want to evaluate more accurately? consider multiple correct choices
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


    def test(self, game: SimpleGame):
        steps = 0
        while game.is_alive:
            env = game.get_environment()
            prediction = self.model.predict(np.array(env.get_as_list())[None,...], verbose=0)
            # print(prediction)
            step_int = np.argmax(prediction)
            # print(Step(step_int))
            game.make_step(Step(step_int))
            steps += 1
        print(f" taken:{steps} food:{game.steps_left}")


    def describe(self):
        print(self.model.summary())


class BiggerDeepl:

    def __init__(self, optimizer = Adam, learning_rate = 0.02, loss = 'categorical_crossentropy', metrics = ['accuracy']):
        #model
        self.model = Sequential([
            Input(shape=(4,)),
            Dense(units=64, activation='relu'),
            Dense(units=128, activation='relu'),
            Dense(units=128, activation='relu'),
            Dense(units=64, activation='relu'),
            Dense(units=4, activation='softmax')
        ])
        self.model.compile(
            optimizer=optimizer(learning_rate=learning_rate),
            loss=loss, metrics=metrics)

    def learn(self, path, batch:int = 100, epochs: int = 50, validation_split = 0.05):
        samples, labels = _get_dataset_from_source(path)
        train_labels = np.array(labels)
        train_samples = np.array(samples)
        train_labels, train_samples = shuffle(train_labels, train_samples)

        print(f"{train_samples[0]}, {train_labels[0]}" )

        # scaler = MinMaxScaler(feature_range=(0,1))
        # scaled_train_samples = scaler.fit_transform(train_samples)

        self.model.fit(x=train_samples, y=train_labels,
                       validation_split=validation_split,
                       batch_size=batch, epochs=epochs,
                       verbose=2)

    def evaluate(self, path, batch:int = 10):
        #might want to evaluate more accurately? consider multiple correct choices
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


    def test(self, game: SimpleGame):
        steps = 0
        while game.is_alive:
            env = game.get_environment()
            prediction = self.model.predict(np.array(env.get_as_list())[None,...])
            # print(prediction)
            step_int = np.argmax(prediction)
            # print(Step(step_int))
            game.make_step(Step(step_int))
            steps += 1
        print(f" taken:{steps} food:{game.steps_left}")


    def describe(self):
        print(self.model.summary())

class ExtendedDeepl(Deepl):

    def __init__(self, optimizer=Adam, learning_rate=0.02, loss='categorical_crossentropy', metrics=['accuracy']):
        super().__init__(optimizer, learning_rate, loss, metrics) #dunno
        #maybe different model size
        self.model = Sequential([
            Input(shape=(4,)),
            Dense(units=64, activation='relu'),
            Dense(units=64, activation='relu'),
            Dense(units=5, activation='softmax')
        ])
        self.model.compile(
            optimizer=optimizer(learning_rate=learning_rate),
            loss=loss, metrics=metrics)


    def test(self, game: SimpleGame):
        steps = 0
        while game.is_alive:
            env = game.get_environment()
            prediction = self.model.predict(np.array(env.get_as_list())[None,...])
            #print(prediction)
            step_int = np.argmax(prediction)
            print(Step(step_int))
            game.make_step(Step(step_int))
            steps += 1
        print(f" taken:{steps} food:{game.steps_left}")


class MultiLabelDeepl:

    def __init__(self, optimizer=Adam, learning_rate=0.02, loss='binary_crossentropy', metrics=['accuracy']):
        self.model = Sequential([
            Input(shape=(4,)),
            Dense(units=64, activation='relu'),
            Dense(units=64, activation='relu'),
            Dense(units=4, activation='sigmoid')
        ])
        self.model.compile(
            optimizer=optimizer(learning_rate=learning_rate),
            loss=loss, metrics=metrics)

    def learn(self, path, batch: int = 100, epochs: int = 100, validation_split=0.05):
        samples, labels = _get_multilabel_dataset_from_source(path)
        train_labels = np.array(labels)
        train_samples = np.array(samples)
        train_labels, train_samples = shuffle(train_labels, train_samples)

        print(f"{train_samples[0]}, {train_labels[0]}")

        # scaler = MinMaxScaler(feature_range=(0,1))
        # scaled_train_samples = scaler.fit_transform(train_samples)

        self.model.fit(x=train_samples, y=train_labels,
                       validation_split=validation_split,
                       batch_size=batch, epochs=epochs,
                       verbose=2)


    def test(self, game: SimpleGame):
        steps = 0
        while game.is_alive:
            env = game.get_environment()
            prediction = self.model.predict(np.array(env.get_as_list())[None,...])
            print(prediction)
            step_int = np.argmax(prediction)
            # print(Step(step_int))
            game.make_step(Step(step_int))
            steps += 1
        print(f" taken:{steps} food:{game.steps_left}")


class MultiLabelDeeplExtended(MultiLabelDeepl):

    def __init__(self, optimizer=Adam, learning_rate=0.02, loss='binary_crossentropy', metrics=['accuracy']):
        super().__init__(optimizer, learning_rate, loss, metrics)
        self.model = Sequential([
            Input(shape=(4,)),
            Dense(units=64, activation='relu'),
            Dense(units=64, activation='relu'),
            Dense(units=5, activation='sigmoid')
        ])
        self.model.compile(
            optimizer=optimizer(learning_rate=learning_rate),
            loss=loss, metrics=metrics)



#SmallClassDeepl(Adam, 0.001, 'categorical_crossentropy', ['accuracy'])