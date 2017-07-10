from constants import *
import numpy as np
from random import random
from keras.models import Model, Input
from keras.layers import Conv2D, Flatten, Dense, BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing import image
from keras import losses


class DQN(object):

    def __init__(self ):
        pass


    def generate_model(self):
        input_layer = Input(shape=(IMG_HEIGHT, IMG_WIDTH, AGENT_HISTORY_LENGTH))
        layer = BatchNormalization()(input_layer)
        layer = Conv2D(32, (8, 8), strides=(4, 4), activation='relu', data_format="channels_last", padding='same')(layer)
        layer = BatchNormalization()(layer)
        layer = Conv2D(64, (4, 4), strides=(2, 2), activation='relu', data_format="channels_last", padding='same')(layer)
        layer = BatchNormalization()(layer)
        layer = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', data_format="channels_last", padding='same')(layer)
        layer = Flatten()(layer)
        layer = BatchNormalization()(layer)
        layer = Dense(512, activation='relu')(layer)
        layer = BatchNormalization()(layer)
        output_layer = Dense(NUM_OF_ACTIONS)(layer)
        model = Model(input_layer, output_layer)
        model.compile(Adam(LEARNING_RATE), loss=losses.mean_squared_error, metrics=['accuracy'])
        return model


    def copy_model(self, model):
        model.save_weights('weights.h5')
        new_model = self.generate_model()
        new_model.load_weights('weights.h5')
        return new_model


