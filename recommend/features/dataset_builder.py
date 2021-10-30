import tensorflow as tf
import tensorflow.strings as tfs
from config import *

def parse_csv(conf):

    DEFAULT_VALUE = "0"

    # 标签处理函数
    def _output_label(feats):
        label = tfs.to_number(
            tfs.split(feats.pop(conf.label), '|') if conf.task == TASK_MULTILABEL else feats.pop(conf.label))
        binary_label = tf.where(label > 0, 1, 0)
        output_label = label if conf.task == TASK_REGRESSION else binary_label
        # 样本权重设置
        sample_weighted = tf.ones_like(label)
        if conf.sample_weighted:
            sample_weighted = tfs.to_number(feats.pop(conf.sample_weighted))

        return label, binary_label, output_label, sample_weighted

    # 数据处理函数
    def _parse(line):
        # 设置特征默认填充值
        default_list = [[DEFAULT_VALUE] for _ in conf.schema]
        columns = tf.io.decode_csv(line, record_defaults=default_list, field_delim=',', use_quote_delim=False)

        feats = dict(zip(list(conf.schema.keys()), columns))

        # 特征处理
        for key, usage in conf.schema.items():
            # 标签或不使用特征不做处理
            if usage in [FEATURE_UNUSED, FEATURE_LABEL]:
                continue

            if usage == FEATURE_NUME:
                feats[key] = tf.strings.to_number(feats[key])
            elif usage == FEATURE_CATE:
                pass

        if conf.run_mode == MODE_PREDICT:
            return feats

        # 标签处理
        label, binary_label, output_label, sample_weighted = _output_label(feats)

        if conf.sample_weighted:
            return feats, output_label, sample_weighted
        else:
            return feats, output_label

    return _parse

class DatasetBuilder:

    def __init__(self, conf, data_inputs):
        self.conf = conf.flags
        self.data_inputs = data_inputs
        if self.conf.run_mode == MODE_PREDICT:
            self.conf.schema.pop(self.conf.label)

    def build(self):
        dataset = tf.data.TextLineDataset(self.data_inputs) \
            .map(parse_csv(self.conf), num_parallel_calls=1) \
            .batch(self.conf.batch_size)
        return dataset
