import os
import tensorflow as tf
import tensorflow_hub as hub
if os.environ.get('TF_KERAS'):
    import tensorflow.keras.backend as K
    from tensorflow.keras.layers import Layer
else:
    import keras.backend as K
    from keras.layers import Layer
from . import helpers


class ElmoLayer(Layer):
    """
    ElmoLayer which support next output_representation param (like https://tfhub.dev/google/elmo/2):
    
    word_emb: the character-based word representations with shape [batch_size, max_length, 512].
    lstm_outputs1: the first LSTM hidden state with shape [batch_size, max_length, 1024].
    lstm_outputs2: the second LSTM hidden state with shape [batch_size, max_length, 1024].
    elmo: the weighted sum of the 3 layers, where the weights are trainable. \
        This tensor has shape [batch_size, max_length, 1024]
    default: a fixed mean-pooling of all contextualized word representations (elmo) with shape [batch_size, 1024].


    if trainable = True, 4 (3 weights for sum of all layer and 1 scale) elmo aggregation params are trainable
    """
    
    def __init__(self, trainable=False, tf_hub=None,
                 output_representation='default', pad_word='--PAD--',
                 tokens_input='tokens', sequence_len_input='sequence_len', elmo_output='elmo',
                 **kwargs):
        """
        Initialize ELMO layer
        :param trainable: is trainable
        :type trainable: bool
        :param tf_hub: tensorflow hub module specification
        :type tf_hub: hub.ModuleSpec
        :param output_representation: string descripts output (TODO: find possible values, replace with Enum)
        :type output_representation: str
        :param pad_word: ELMO's padding word
        :type pad_word: str
        :param tokens_input: ELMO tokens input name
        :type tokens_input: str
        :param sequence_len_input: ELMO sequence lenght input name
        :type sequence_len_input: int|NoneType
        :param elmo_output: ELMO output name
        :type elmo_output: str
        """
        super(ElmoLayer, self).__init__(**kwargs)
        self.dimensions = 512 if output_representation == 'word_emb' else 1024
        self.output_representation = output_representation 
        self.is_trainable = trainable
        self.supports_masking = True
        self.tf_hub = tf_hub
        self.pad_word = pad_word
        self.tokens_input = tokens_input
        self.sequence_len_input = sequence_len_input
        self.elmo_output = elmo_output
        self.elmo = hub.Module(self.tf_hub, trainable=self.is_trainable,
                               name="{}_module".format(self.name))

    def get_config(self):
        config = super(ElmoLayer, self).get_config()
        config['tf_hub'] = self.tf_hub
        config['output_representation'] = self.output_representation
        config['pad_word'] = self.pad_word
        config['tokens_input'] = self.tokens_input
        config['sequence_len_input'] = self.sequence_len_input
        config['elmo_output'] = self.elmo_output
        return config

    def build(self, input_shape):
        variables = list(self.elmo.variable_map.values())
        if self.is_trainable:
            trainable_vars = [var for var in variables if "/aggregation/" in var.name]
            for var in trainable_vars:
                self._trainable_weights.append(var)
            # Add non-trainable weights
            for var in variables:
                if var not in self._trainable_weights:
                    self._non_trainable_weights.append(var)
        else:
            for var in variables:
                self._non_trainable_weights.append(var)
        super(ElmoLayer, self).build(input_shape)

    def call(self, x, mask=None):
        """
        Apply ELMO to sequences
        :param x: sequences tensor
        :param mask: mask tensow
        :return: ELMO embeddings tensor
        """
        inputs = {self.tokens_input: K.cast(x, tf.string)}
        if self.sequence_len_input:
            inputs[self.sequence_len_input] = self.compute_token_mask(x)
        result = self.elmo(inputs,
                           as_dict=True,
                           signature='tokens')
        if self.output_representation == 'default':
            input_mask = K.cast(K.not_equal(x, self.pad_word), K.floatx())
            return helpers.masked_reduce_mean(result[self.elmo_output],
                                              input_mask)
        else:
            return result[self.output_representation]

    def compute_token_mask(self, inputs, mask=None):
        return K.sum(K.cast(K.not_equal(inputs, self.pad_word), 'int32'), axis=-1)
    
    def compute_mask(self, inputs, mask=None):
        if self.output_representation == 'default':
            return None
        else:
            return K.not_equal(inputs, self.pad_word)

    def compute_output_shape(self, input_shape):
        if self.output_representation == 'default':
            return input_shape[0], self.dimensions
        
        else:
            return input_shape[0], input_shape[1], self.dimensions
