import tensorflow_hub as hub
from keras import backend as K
from keras.layers import Layer
from . import helpers


class BertLayer(Layer):
    """
    BertLayer which support next output_representation param:
    
    pooled_output: the first CLS token after adding projection layer () with shape [batch_size, 768]. 
    sequence_output: all tokens output with shape [batch_size, max_length, 768].
    mean_pooling: mean pooling of all tokens output [batch_size, max_length, 768].

    You can simple fine-tune last n layers in BERT with n_fine_tune_layers parameter.
    For view trainable parameters call model.trainable_weights after creating model.
    """

    def __init__(self, n_fine_tune_layers=10, tf_hub=None, output_representation='pooled_output',
                 trainable=False, **kwargs):
        """
        Initialize BERT layer
        :param n_fine_tune_layers: Count of tuneable BERT layers
        :type n_fine_tune_layers: int
        :param tf_hub: tensorflow Hub module specification
        :type tf_hub: hub.ModuleSpec|NoneType
        :param output_representation: one of "pooled_output"/"mean_pooling"/"sequence_output" (
            TODO: replace with Enum
        )
        :type output_representation: str
        :param trainable: Is layer trainable or not?
        :type trainable: bool
        """
        assert output_representation in ["pooled_output", "mean_pooling", "sequence_output"]
        super(BertLayer, self).__init__(**kwargs)
        self.n_fine_tune_layers = n_fine_tune_layers
        self.is_trainable = trainable
        self.output_size = 768
        self.tf_hub = tf_hub
        self.output_representation = output_representation
        self.supports_masking = True
        self.bert = hub.Module(
            self.tf_hub,
            trainable=self.is_trainable,
            name="{}_module".format(self.name)
        )

    def build(self, input_shape):
        variables = list(self.bert.variable_map.values())
        if self.is_trainable:
            # 1 first remove unused layers
            trainable_vars = [var for var in variables if "/cls/" not in var.name]
            if self.output_representation == "sequence_output" or self.output_representation == "mean_pooling":
                # 1 first remove unused pooled layers
                trainable_vars = [var for var in trainable_vars if "/pooler/" not in var.name]
            # Select how many layers to fine tune
            trainable_vars = trainable_vars[-self.n_fine_tune_layers:]
            # Add to trainable weights
            for var in trainable_vars:
                self._trainable_weights.append(var)
            # Add non-trainable weights
            for var in self.bert.variables:
                if var not in self._trainable_weights:
                    self._non_trainable_weights.append(var)
        else:
            for var in variables:
                self._non_trainable_weights.append(var)
        super(BertLayer, self).build(input_shape)

    def call(self, inputs, **_):
        """
        Apply BERT to given inputs
        :param inputs: Tensors with input ids / input mask / segment ids
        :return: pooled (or sequence) output layer
        """
        inputs = [K.cast(x, dtype="int32") for x in inputs]
        input_ids, input_mask, segment_ids = inputs
        bert_inputs = dict(
            input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids
        )
        result = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)
        if self.output_representation == "pooled_output":
            return result["pooled_output"]
        elif self.output_representation == "mean_pooling":
            return helpers.masked_reduce_mean(result["sequence_output"],
                                              input_mask)
        elif self.output_representation == "sequence_output":
            return result["sequence_output"]

    def compute_mask(self, inputs, mask=None):
        if self.output_representation == 'sequence_output':
            inputs = [K.cast(x, dtype="bool") for x in inputs]
            mask = inputs[1]
            
            return mask
        else:
            return None

    def compute_output_shape(self, input_shape):
        if self.output_representation == "sequence_output":
            return input_shape[0][0], input_shape[0][1], self.output_size
        else:
            return input_shape[0][0], self.output_size
