import os
import numpy as np
if os.environ.get('TQDM_NOTEBOOK'):
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm


class ElmoTokenizer(object):
    """
    Bert tokenizer for raw text
    """
    def __init__(self, max_seq_length=256, pad_word='--PAD--'):
        self.pad_word = pad_word
        self.max_seq_length = max_seq_length

    def _pad_sequence(self, sequence):
        padding_length = max(self.max_seq_length - len(sequence), 0)
        padding = np.array([self.pad_word] * padding_length)
        padded = np.concatenate([sequence, padding])
        return padded[:self.max_seq_length]

    def _split_text(self, text):
        return np.array(text.split())[:self.max_seq_length]

    def predict(self, texts, verbose=False):
        """
        Parse texts for ELMO
        :param texts: texts
        :type texts: list[str]
        :param verbose: Verbose building process?
        :type verbose: bool
        """
        if verbose:
            iterable = tqdm(texts, desc="Converting examples to tokens")
        else:
            iterable = texts
        tokens = [self._pad_sequence(self._split_text(text))
                  for text in iterable]
        return np.vstack(tokens)
