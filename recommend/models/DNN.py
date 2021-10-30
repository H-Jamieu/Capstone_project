from models.base_model import BaseModel
from tensorflow.keras.layers import Dense, Dropout
from config import *


class DNN(BaseModel):
    def __init__(self, **kwargs):
        super(DNN, self).__init__(**kwargs)
        self.model_conf = self._conf.model.get("dnn", {})
        self.hidden_units = self.model_conf.get("hidden_units", [32, 8])
        self.activation = self.model_conf.get("activation", "relu")
        self.dropout_rate = self.model_conf.get("dropout_rate", 0.1)

    def build_model(self):

        inputs = self.combine_dense_feature()
        hidden_layer = inputs
        for unit in self.hidden_units:
            hidden_layer = Dense(units=unit, activation=self.activation)(hidden_layer)
            hidden_layer = Dropout(rate=self.dropout_rate)(hidden_layer)

        if self._conf.task == TASK_BINARY:
            outputs = Dense(units=1, activation="sigmoid")(hidden_layer)
        elif self._conf.task == TASK_REGRESSION:
            outputs = Dense(units=1, activation=None)(hidden_layer)
        else:
            NotImplemented()
        return outputs
