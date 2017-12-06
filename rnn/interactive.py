from collections import defaultdict
from pathlib import Path
from typing import Union

import pandas as pd
import tensorflow as tf
import numpy as np

import rnn.data.utils as du
from rnn.data.input import create_prediction_input_arrays

CHECKPOINT_PREFIX_LENGTH = len('model.ckpt-')
CHECKPOINT_SUFFIX_LENGTH = len('.meta')


def is_metafile(path: Path) -> bool:
    return str(path).endswith('.meta')


def get_checkpoint_number(path: Path) -> int:
    return int(path.name[CHECKPOINT_PREFIX_LENGTH:-CHECKPOINT_SUFFIX_LENGTH])


def create_session(path: Union[str, Path]) -> tf.Session:
    path = Path(path)

    if path.is_dir():
        metafile = max(filter(is_metafile, path.iterdir()), key=get_checkpoint_number)
    else:
        metafile = path

    sess = tf.Session()
    saver = tf.train.import_meta_graph(str(metafile))
    saver.restore(sess, str(metafile.with_suffix('')))
    return sess


class Prediction:
    def __init__(self, tokens: list, class_id: int, token_features: np.ndarray):
        self.tokens = tokens
        self.length = len(tokens)
        self.class_label = du.CLASSES[class_id]

        self.token_features = pd.DataFrame(token_features[:self.length, :].T)
        self.token_features.columns = self.tokens


class Evaluator:
    def __init__(self, path: Union[str, Path]):
        self.data_dir = du.get_data_dir()
        self.sess = create_session(path)
        # self.embedding_model = du.load_embedding_model(self.data_dir)

        vocabulary = du.load_list(self.data_dir / du.VOCAB_SUBPATH)
        self.original_vocab_size = len(vocabulary)
        self.word_encoding = defaultdict()
        self.word_encoding.update({k: v for v, k in enumerate(vocabulary)})

        self.embedding_matrix = du.load_embeddings_matrix(self.data_dir)

    def eval(self, text: str) -> Prediction:
        df = pd.DataFrame({
            "id": ["-"],
            "text": [text]
        })

        arrays = create_prediction_input_arrays(df, self.word_encoding)
        feed_dict = {
            'features/id:0': arrays['id'],
            'features/text:0': arrays['text'],
            'features/text_length:0': arrays['text_length'],
            'embeddings:0': self.embedding_matrix
        }
        rev_words_encoding = {v: k for k, v in self.word_encoding.items()}
        tokens = [rev_words_encoding[i] for i in arrays['text'][0]]

        raw_predictions = self.sess.run(['output/prediction:0', 'encoder/token/dense_output:0'], feed_dict)
        return Prediction(tokens, int(raw_predictions[0][0]), raw_predictions[1][0])
