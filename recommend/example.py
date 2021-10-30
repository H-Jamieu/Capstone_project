import os
import tensorflow as tf

from config import *
from features.dataset_builder import DatasetBuilder
from features.feature_builder import FeatureBuilder
from models.model_builder import ModelBuilder


def example_DNN():
    config = Config()
    feature_builder = FeatureBuilder(conf=config).build()

    if config.flags.run_mode == MODE_TRAIN:
        train_dataset = DatasetBuilder(conf=config, data_inputs=config.flags.train_data_inputs).build()
        valid_dataset = DatasetBuilder(conf=config, data_inputs=config.flags.valid_data_inputs).build()
        model = ModelBuilder(conf=config, feature_builder=feature_builder).build()
        model.fit(train_dataset, epochs=10, validation_data=valid_dataset)
        model.save(config.flags.model_path)
    elif config.flags.run_mode == MODE_PREDICT:
        model = ModelBuilder(conf=config, feature_builder=feature_builder).build()
        test_dataset = DatasetBuilder(conf=config, data_inputs=config.flags.test_data_inputs).build()
        preds = model.predict(test_dataset)
        print(preds)

if __name__ == "__main__":
    example_DNN()