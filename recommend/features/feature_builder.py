import tensorflow as tf
from config import *


class FeatureBuilder:

    def __init__(self, conf):
        self.conf = conf.flags
        self.feat_columns = []
        self.feat_inputs = {}
        self.embedding_size = 8

    def build(self):
        # 特征处理
        for feat, usage in self.conf.schema.items():
            if usage in [FEATURE_UNUSED, FEATURE_LABEL]:
                continue

            # 类别型特征处理
            if usage == FEATURE_CATE:
                # 处理标签编码特征, 对于超出[0, num_bucket)的特征值使用default_value代替
                # feat_column = tf.feature_column.categorical_column_with_identity(feat, num_buckets=1000, default_value=None)
                # onehot编码
                # feat_column = tf.feature_column.indicator_column(categorical_column=column)
                feat_column = tf.feature_column.categorical_column_with_hash_bucket(feat, 1000)
                # embedding
                feat_column = tf.feature_column.embedding_column(categorical_column=feat_column,
                                                                 dimension=self.embedding_size, combiner="mean")
                self.feat_inputs[feat] = tf.keras.layers.Input(shape=(self.embedding_size,), name=feat, dtype=tf.string)
            # 数值型特征处理
            elif usage == FEATURE_NUME:
                feat_column = tf.feature_column.numeric_column(feat)
                self.feat_inputs[feat] = tf.keras.layers.Input(shape=(1,), name=feat, dtype=tf.float64)
            else:
                raise Exception(f"Error feature usage: {usage}")

            self.feat_columns.append(feat_column)
        return self


