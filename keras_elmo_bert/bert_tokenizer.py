import tensorflow_hub as hub
import tensorflow as tf
from bert.tokenization import FullTokenizer
from tqdm import tqdm
from bert import tokenization
import numpy as np


class InputExample(object):
    """
    A single training/test example for simple sequence classification.
    """

    def __init__(self, guid, text_a, text_b=None, label=None):
        """
        Constructs a InputExample.
        :param guid: Unique id for the example.
        :type guid: str|NoneType
        :param text_a: The untokenized text of the first sequence. \
            For single sequence tasks, only this sequence must be specified.
        :type text_a: str
        :param text_b: (Optional) string. The untokenized text of the second sequence. \
            Only must be specified for sequence pair tasks.
        :type text_b: str|NoneType
        :param label: (Optional) string. The label of the example. \
            This should be specified for train and dev examples, but not for test examples.
        :type label: str|NoneType
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class PaddingInputExample(object):
    """
    Fake example so the num input examples is a multiple of the batch size.
    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.
    We use this class instead of `None` because treating `None` as padding
    batches could cause silent errors.
    """
    pass


class InputFeatures(object):
    """
    A single set of features of data.
    """

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 is_real_example=True):
        """
        Initialize set
        :param input_ids: Input token ids
        :type input_ids: np.ndarray|list[int]
        :param input_mask: Input mask
        :type input_mask: np.ndarray|list[int]
        :param segment_ids: Segment ids for lm task
        :type segment_ids: np.ndarray|list[int]
        :param label_id: label id
        :type label_id: int
        :param is_real_example:
        """
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example


def create_tokenizer_from_hub_module(tf_hub):
    """
    Get the vocab file and casing info from the Hub module.
    :param tf_hub: Tensorflow Hub module specification
    :type tf_hub: hub.ModuleSpec
    :return: tokenizer instance
    :rtype: FullTokenizer
    """
    with tf.Session() as sess:
        bert_module = hub.Module(tf_hub)
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        vocab_file, do_lower_case = sess.run([
            tokenization_info["vocab_file"],
            tokenization_info["do_lower_case"],
        ])
    return FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)


def _truncate_sequences(max_length, tokens_a, tokens_b=None):
    """
    Inplace truncates a sequence pair in place to the maximum length.
    :param max_length: max length for both sequences
    :type max_length: int
    :param tokens_a: sequence A
    :type tokens_a: list
    :param tokens_b: sequence B or None
    :type tokens_b: list|NoneType
    """
    if tokens_b is None:
        tokens_b = []
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_single_example(verbose_log, example, max_seq_length, tokenizer):
    """
    Converts a single `InputExample` into a single `InputFeatures`.
    :param verbose_log: Add sample to TF log?
    :type verbose_log: bool
    :param example: instance of InputExample
    :type example: InputExample
    :param max_seq_length: max BERT input's sequence length
    :type max_seq_length: int
    :param tokenizer: FullTokenizer instance (for example, from ```create_tokenizer_from_hub_module```)
    :type tokenizer: FullTokenizer
    :return: InputFeatures instance
    :rtype: InputFeatures
    """

    if isinstance(example, PaddingInputExample):
        return InputFeatures(input_ids=[0] * max_seq_length,
                             input_mask=[0] * max_seq_length,
                             segment_ids=[0] * max_seq_length,
                             label_id=0,
                             is_real_example=False)

    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_sequences(max_length=max_seq_length - 3,
                            tokens_a=tokens_a,
                            tokens_b=tokens_b)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        _truncate_sequences(max_length=max_seq_length - 2,
                            tokens_a=tokens_a)

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
    segment_ids = [0] * len(tokens)
    if tokens_b:
        tokens += tokens_b + ["[SEP]"]
        segment_ids += [1] * (len(tokens_b) + 1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    # Input mask will be ones for real tokens and zeros for mask
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    label_id = example.label
    if verbose_log:
        tf.logging.info("*** Example ***")
        tf.logging.info("Tokens True: %s" % " ".join(tokens))
        tf.logging.info("guid: %s" % example.guid)
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

    feature = InputFeatures(input_ids=input_ids,
                            input_mask=input_mask,
                            segment_ids=segment_ids,
                            label_id=label_id,
                            is_real_example=True)
    return feature


def convert_examples_to_features(examples, max_seq_length, tokenizer, verbose=False):
    """
    Convert a set of `InputExample`s to a list of `InputFeatures`.
    :param examples: input examples
    :type examples: list[InputExample]
    :param max_seq_length: max BERT's sequence length
    :type max_seq_length: int
    :param tokenizer: FullTokenizer instance (for example, from ```create_tokenizer_from_hub_module```)
    :type tokenizer: FullTokenizer
    :param verbose: Verbose building process?
    :type verbose: bool
    :return: Arrays with input ids/masks/segment ids/label ids
    :rtype: [np.ndarray,np.ndarray,np.ndarray,np.ndarray]
    """
    input_ids, input_masks, segment_ids, labels = [], [], [], []

    if verbose:
        iterable = tqdm(examples, desc="Converting examples to features")
    else:
        iterable = examples

    for i, example in enumerate(iterable):
        if (i % 10000 == 0) and verbose:
            tf.logging.info("Writing example %d of %d" % (i, len(examples)))
        features = convert_single_example(verbose_log=(i == 0) and verbose,
                                          example=example,
                                          max_seq_length=max_seq_length,
                                          tokenizer=tokenizer)
        input_ids.append(features.input_ids)
        input_masks.append(features.input_mask)
        segment_ids.append(features.segment_ids)
        labels.append(features.label_id)
    return np.array(input_ids),\
           np.array(input_masks), \
           np.array(segment_ids), \
           np.array(labels)


def convert_text_to_examples(texts, labels):
    """
    Create InputExamples from texts & labels (for 1-text superwised case)
    :param texts: list of texts
    :type texts: list[str]
    :param labels: list of labels
    :type labels: list[str]
    :return: list of InputExample instances
    :rtype: list[InputExample]
    """
    assert len(texts) == len(labels)
    input_examples = []
    for text, label in zip(texts, labels):
        assert isinstance(text, str)  # TODO: How can other options be possible?
        input_examples.append(InputExample(guid=None, text_a=text[0], text_b=text[1], label=label))
    return input_examples


class BertTokenizer(object):
    """
    Bert tokenizer for raw text
    """
    def __init__(self, tf_hub=None, max_seq_length=256):
        """
        Initialize tokenizer
        :param tf_hub: tokenizer tensorflow module specification
        :type tf_hub: hub.ModuleSpec
        :param max_seq_length: max BERT's sequence length
        :type max_seq_length: int
        """
        self.tokenizer = create_tokenizer_from_hub_module(tf_hub)
        self.max_seq_length = max_seq_length
    
    def predict(self, texts, labels=None, verbose=False):
        """
        Convert texts & labels to BERT features
        :param texts: texts
        :type texts: list[str]
        :param labels: labels
        :type labels: list[str]
        :param verbose: Verbose building process?
        :type verbose: bool
        :return: Arrays with input ids/masks/segment ids/label ids
        :rtype: [np.ndarray,np.ndarray,np.ndarray,np.ndarray]
        """
        # Convert data to InputExample format
        if labels is None:
            labels = np.zeros([len(texts)])
        examples = convert_text_to_examples(texts, labels)
        return convert_examples_to_features(examples=examples,
                                            max_seq_length=self.max_seq_length,
                                            tokenizer=self.tokenizer,
                                            verbose=verbose)
