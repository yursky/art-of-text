import json
import logging
import os
import re
from collections import defaultdict
from contextlib import suppress
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


MIN_AFTER_DEQUEUE = 1024
PARAMS_FILENAME = 'params.json'
MODELS_DIR_VARIABLE_NAME = 'TENSORFLOW_MODELS_DIR'
DATA_DIR_VARIABLE_NAME = 'RESEARCH_DATA_DIR'
PROBLEM_SUBDIR = ''
PREPROCESSED_DATA_SUBDIR = Path('preproc')
TRAIN_DATA_SUBDIR = PREPROCESSED_DATA_SUBDIR / 'train'
TEST_DATA_SUBDIR = PREPROCESSED_DATA_SUBDIR / 'test'
TEST_CSV_SUBPATH = PREPROCESSED_DATA_SUBDIR / 'test.csv'
RAW_DATA_SUBDIR = 'raw'
EMBEDDINGS_SUBPATH = 'wiki.simple'
DIRECT_EMBEDDINGS_SUBPATH = PREPROCESSED_DATA_SUBDIR / 'wiki.simple.npy'
VOCAB_SUBPATH = PREPROCESSED_DATA_SUBDIR / 'vocab.txt'
UNKNOWN_WORD_CODE = -1
EOS_TAG = '</s>'
SEQUENCE_LENGTH_LIMIT = 1280

CLASSES = ['EAP', 'HPL', 'MWS']


def load_embedding_model(data_dir: Path):
    embeddings_path = data_dir / EMBEDDINGS_SUBPATH
    import gensim
    return gensim.models.wrappers.FastText.load_fasttext_format(str(embeddings_path))


def create_embedding_vectors(words: list, embedding_model):
    def get_vector(word: str):
        vec = None
        with suppress(KeyError):
            vec = embedding_model[word]
        if vec is None:
            with suppress(KeyError):
                vec = embedding_model[word.lower()]
        if vec is None:
            with suppress(KeyError):
                vec = embedding_model['_']
        return vec.astype(np.float32)

    return [get_vector(w) for w in words]


def store_vocab(words, data_dir: Path):
    vocab_path = data_dir / VOCAB_SUBPATH
    direct_embeddings_path = data_dir / DIRECT_EMBEDDINGS_SUBPATH

    store_list(words, vocab_path)
    embedding_model = load_embedding_model(data_dir)

    def get_vector(word: str):
        vec = None
        with suppress(KeyError):
            vec = embedding_model[word]
        if vec is None:
            with suppress(KeyError):
                vec = embedding_model[word.lower()]
        if vec is None:
            with suppress(KeyError):
                vec = embedding_model['_']
        return vec.astype(np.float32)

    vectors = [get_vector(w) for w in words]
    np.save(direct_embeddings_path, vectors)


def encoding_as_list(encoding: dict):
    keys = [None for _ in encoding]
    for k, i in encoding.items():
        keys[i] = k

    return keys


class WordEncoder(defaultdict):
    def __init__(self, data_dir: Path):
        super().__init__(lambda: len(self))
        self.data_dir = data_dir
        self[EOS_TAG] = 0

    def store_direct_embeddings(self):
        words = encoding_as_list(self)
        store_vocab(words, self.data_dir)


def load_embeddings_matrix(source_dir: Path) -> np.ndarray:
    return np.load(str(source_dir / DIRECT_EMBEDDINGS_SUBPATH))


def get_data_dir() -> Path:
    import tempfile

    data_dir = os.environ.get(DATA_DIR_VARIABLE_NAME)
    if data_dir is not None:
        log.info("Using '{}' as the data directory.".format(data_dir))
        data_dir = Path(data_dir) / PROBLEM_SUBDIR
    else:
        data_dir = Path(tempfile.gettempdir()) / 'data'
        log.warning("Data directory variable '{}' not defined. Using '{}' in its place.".format(
            DATA_DIR_VARIABLE_NAME, data_dir))
        data_dir = data_dir / PROBLEM_SUBDIR

    return data_dir


def get_model_dir(name: str = None) -> Path:
    import tempfile

    models_dir = os.environ.get(MODELS_DIR_VARIABLE_NAME)
    if models_dir is None:
        tmp_dir = Path(tempfile.gettempdir())
        default_models_subdir = 'tf_models'
        models_dir = tmp_dir / default_models_subdir
    else:
        models_dir = Path(models_dir)

    if name is not None:
        model_dir = models_dir / name
    else:
        unnamed_models_dir = models_dir / 'unnamed'
        os.makedirs(str(unnamed_models_dir), exist_ok=True)
        model_dir = Path(tempfile.mkdtemp(dir=str(unnamed_models_dir), prefix=''))

    return model_dir


def store_params(params: dict, model_dir: Path):
    os.makedirs(str(model_dir), exist_ok=True)
    with (model_dir / PARAMS_FILENAME).open('w') as params_file:
        json.dump(params, params_file)


def load_params(model_dir: Path) -> dict:
    os.makedirs(str(model_dir), exist_ok=True)
    try:
        with (model_dir / PARAMS_FILENAME).open() as params_file:
            return json.load(params_file)
    except FileNotFoundError:
        return {}


def store_list(values: list, path: Path):
    with path.open('w') as output_file:
        for value in values:
            output_file.write(str(value) + '\n')


def load_list(path: Path) -> list:
    try:
        with path.open() as input_file:
            return [value.strip() for value in input_file.readlines()]
    except FileNotFoundError:
        return []


def encode_text(text: str, encoding: dict, default=UNKNOWN_WORD_CODE):
    clean_text = re.sub(r'([^\w-]+)', ' \g<0> ', text).strip()
    tokens = clean_text.split()
    tokens = ['#' if t.isnumeric() else t for t in tokens]
    if isinstance(encoding, defaultdict):
        return [encoding[token] for token in tokens][:SEQUENCE_LENGTH_LIMIT-1] + [0]
    else:
        return [encoding.get(token, default) for token in tokens][:SEQUENCE_LENGTH_LIMIT-1] + [0]
