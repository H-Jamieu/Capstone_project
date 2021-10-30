import tensorflow as tf


class Core(object):

    def __init__(self):
        self._model = None

    def load(self, model_path, **kwargs):
        self._model = tf.keras.models.load_model(model_path, **kwargs)

    def fit(self, *args, **kwargs):
        return self._model.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self._model.predict(*args, **kwargs)

    def summary(self, *args, **kwargs):
        return self._model.summary(*args, **kwargs)

    def save(self, *args, **kwargs):
        return self._model.save(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        return self._model.evaluate(*args, **kwargs)

    def build(self):
        raise NotImplementedError


class BaseModel(Core):
    metrics = {
        'regression':
            [tf.keras.metrics.MeanAbsoluteError(name='mae'),
             tf.keras.metrics.MeanAbsolutePercentageError(name='mape')],
        'binary': [
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.TruePositives(name='tp'),
            tf.keras.metrics.FalsePositives(name='fp'),
            tf.keras.metrics.TrueNegatives(name='tn'),
            tf.keras.metrics.FalseNegatives(name='fn'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ],
        'multiclass': [
            tf.keras.metrics.SparseCategoricalAccuracy(name='cate_accuracy'),
            tf.keras.metrics.SparseCategoricalCrossentropy(name='cate_crossentropy', from_logits=True)
        ],
    }

    loss_function = {
        'regression': 'mse',
        'binary': 'binary_crossentropy',
        'multiclass': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    }

    optimizer = tf.keras.optimizers.Adam()

    def __init__(self, **kwargs):
        super(BaseModel, self).__init__()
        conf = kwargs.get("conf", None)
        if conf is None:
            raise ValueError('conf must be set.')
        feature_builder = kwargs.get("feature_builder", None)
        if feature_builder is None:
            raise ValueError('feature_builder must be set.')
        self._conf = conf
        self._feature_builder = feature_builder

        # set model parameters according to model.yaml and task
        self._optimizer = tf.keras.optimizers.Adam()
        self._loss = self.loss_function[conf.task]
        self._metric = self.metrics[conf.task]
        self._loss_weight = None

    def compile(self):
        self._model.compile(optimizer=self._optimizer,
                            loss=self._loss,
                            loss_weights=self._loss_weight,
                            metrics=self._metric)

    def combine_dense_feature(self):
        feature_layer = tf.keras.layers.DenseFeatures(self._feature_builder.feat_columns)
        feature_inputs = feature_layer(self._feature_builder.feat_inputs)
        return feature_inputs

    def build(self):
        inputs = self._feature_builder.feat_inputs
        outputs = self.build_model()
        self._model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    def build_model(self):
        NotImplemented()



