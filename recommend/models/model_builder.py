from .DNN import DNN
from config import *
import tensorflow as tf

class ModelBuilder:
    model_dict = {
        "dnn": DNN,
    }

    def __init__(self, conf, feature_builder):
        self.conf = conf.flags
        self.feature_builder = feature_builder
        self.model_type = self.conf.model_type

    def build(self):

        if self.conf.run_mode == MODE_TRAIN:
            if self.model_type not in self.model_dict:
                raise Exception("model type error!!")
            model = self.model_dict[self.model_type](conf=self.conf,
                                                     feature_builder=self.feature_builder)
            model.build()
            model.compile()
        else:
            model = tf.keras.models.load_model(self.conf.model_path)
        return model
