import argparse
import json
import yaml

TASK_BINARY = "binary"
TASK_REGRESSION = "regression"
TASK_MULTILABEL = "multitask"

FEATURE_UNUSED = "unused"
FEATURE_LABEL = "label"
FEATURE_CATE = "category"
FEATURE_NUME = "continuous"

MODE_TRAIN = "train"
MODE_PREDICT = "predict"

class Config(object):
    def __init__(self):
        self._get_args()
        self._read_yaml()

    def _get_args(self):
        parser = argparse.ArgumentParser(description="")
        parser.add_argument("--task", type=str, help="任务类型",
                            choices=[TASK_BINARY, TASK_REGRESSION, TASK_MULTILABEL],
                            default=TASK_BINARY)
        parser.add_argument("--run_mode", type=str, help="运行模式",
                            choices=[MODE_TRAIN, MODE_PREDICT],
                            default=MODE_TRAIN)

        ## 数据集配置
        parser.add_argument("--train_data_inputs", type=str, help="训练数据集路径", nargs="+", default="data/ml-25m/example_train_ratings.csv")
        parser.add_argument("--valid_data_inputs", type=str, help="验证数据集路径", nargs="+", default="data/ml-25m/example_train_ratings.csv")
        parser.add_argument("--test_data_inputs", type=str, help="测试数据集路径", nargs="+", default="data/ml-25m/example_test_ratings.csv")
        parser.add_argument("--batch_size", type=int, help="数据集每个批次数据量", default=128)

        ## 特征配置
        parser.add_argument("--schema", type=str, help="特征配置文件路径", default="conf/schema.yaml")
        parser.add_argument("--feature", type=str, help="特征处理文件路径", default="conf/feature.yaml")
        parser.add_argument("--label", type=str, help="标签特征", default="rating")

        ## 模型配置
        parser.add_argument("--model", type=str, help="模型配置文件路径", default="conf/model.yaml")
        parser.add_argument("--sample_weighted", type=str, help="模型样本权重")
        parser.add_argument("--model_type", type=str, help="模型类型", default="dnn")
        parser.add_argument("--model_conf", type=str, help="模型参数配置", default="{}")
        parser.add_argument("--model_path", type=str, help="模型保存路径", default='output/model.h5')

        self.flags = parser.parse_args()

    def _read_yaml(self):
        with open(self.flags.schema) as f:
            self.flags.schema = yaml.load(f, Loader=yaml.FullLoader)
        with open(self.flags.feature) as f:
            self.flags.feature = yaml.load(f, Loader=yaml.FullLoader)
        with open(self.flags.model) as f:
            self.flags.model = yaml.load(f, Loader=yaml.FullLoader)


