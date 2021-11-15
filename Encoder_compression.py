import math
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import MeanSquaredLogarithmicError

TRAIN_DATA_PATH = 'Data/movie_data/raw_embeddings.csv'

train_data = pd.read_csv(TRAIN_DATA_PATH, header=None)

x_train = train_data.loc[:,train_data.columns[1:]]



class AutoEncoders(Model):

    def __init__(self, output_units):
        super().__init__()
        self.encoder = Sequential(
            [
                Dense(1024, activation="relu"),
                Dense(512, activation="relu"),
                Dense(256, activation="relu"),
                Dense(64, activation="relu"),
            ]
        )

        self.decoder = Sequential(
            [
                Dense(64, activation="relu"),
                Dense(256, activation="relu"),
                Dense(512, activation="relu"),
                Dense(1024, activation="relu"),
                Dense(output_units, activation="sigmoid")
            ]
        )
    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded


auto_encoder = AutoEncoders(len(x_train.columns))

auto_encoder.compile(
    loss='mse',
    metrics=['mse'],
    optimizer='adam'
)

history = auto_encoder.fit(
    x_train,
    x_train,
    epochs=100,
    batch_size=512,
    validation_data=(x_train, x_train)
)

encoder_layer = auto_encoder.get_layer('sequential')
reduced_df = pd.DataFrame(encoder_layer.predict(x_train))
reduced_df = reduced_df.add_prefix('feature_')
reduced_df['movieID'] = train_data.loc[:,train_data.columns[0]]
reduced_df.to_csv('Data/Movie_data/reduced_embeddings.csv')