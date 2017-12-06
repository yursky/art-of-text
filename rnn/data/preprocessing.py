from pathlib import Path
import logging

import numpy as np
import pandas as pd
import tensorflow as tf

import rnn.data.utils as utils
from rnn.data.utils import encode_text

log = logging.getLogger(__name__)

INDEX_FILENAME = 'index.csv'
NUM_EXAMPLES_PER_RECORDS_FILE = 2000


def prepare_data(filenames: list, data_dir: Path, train_fraction=1.0):
    input_dir = data_dir / utils.RAW_DATA_SUBDIR
    train_dir = data_dir / utils.TRAIN_DATA_SUBDIR
    test_dir = data_dir / utils.TEST_DATA_SUBDIR

    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    word_encoder = utils.WordEncoder(data_dir)
    test_dfs = []

    for filename in filenames:
        log.info("Preprocessing file '{}'".format(filename))
        name, _ = filename.split('.')
        with (input_dir / filename).open() as f:
            df = pd.read_csv(f)
        df['id'] = df['id'].apply(lambda x: name + str(x))

        if train_fraction == 1.0:
            train_df = df
        else:
            np.random.seed(0)
            train_df = df.sample(frac=train_fraction)

        store_tfrecords_from_df(name, word_encoder, train_df, train_dir)
        if train_fraction < 1.0:
            test_df = df.drop(train_df.index)
            store_tfrecords_from_df(name, word_encoder, test_df, test_dir)
            test_dfs.append(test_df)

    log.info("Storing valid tags for the test set")
    whole_test_df = pd.concat(test_dfs)
    assert(isinstance(whole_test_df, pd.DataFrame))
    whole_test_df.to_csv(data_dir / utils.TEST_CSV_SUBPATH, columns=['id', 'author'], index=False)
    log.info("Storing direct embeddings")
    word_encoder.store_direct_embeddings()


def encoded_string_features_dict(name: str, value: str, word_encoding: dict) -> dict:
    tokens = encode_text(value, word_encoding)
    return {
        name: tf.train.Feature(int64_list=tf.train.Int64List(value=tokens)),
        name + '_length': tf.train.Feature(int64_list=tf.train.Int64List(value=[len(tokens)]))
    }


def string_feature(value: str) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))


def int_feature(value: int) -> tf.train.Feature:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def store_tfrecords_from_df(name: str,
                            word_encoding: dict,
                            df: pd.DataFrame,
                            output_dir: Path):
    """
    Store a data frame as TFRecords. It splits the data frame across multiple files to allow better shuffling.

    :param name: base name for the output filename (without extension)
    :param word_encoding: words encoding to be used
    :param df: data frame to store
    :param output_dir: directory used to store the output files
    """
    df.to_csv(output_dir / INDEX_FILENAME, columns=['id'], header=False, index=False, mode='a')
    record_writer = None
    for i, (_, row) in enumerate(df.iterrows()):
        if i % NUM_EXAMPLES_PER_RECORDS_FILE == 0:
            file_number = int(i / NUM_EXAMPLES_PER_RECORDS_FILE)
            record_writer = tf.python_io.TFRecordWriter(
                str(output_dir / '{}{:04d}.tfrecords'.format(name, file_number)))
        features = {
            'id': string_feature(row['id']),
            'author': int_feature(utils.CLASSES.index(row['author']))
        }
        features.update(encoded_string_features_dict('text', row['text'], word_encoding))
        example = tf.train.Example(features=tf.train.Features(feature=features))
        record_writer.write(example.SerializeToString())
