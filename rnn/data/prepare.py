import argparse
from pathlib import Path

import rnn.data.preprocessing as preprocessing
import rnn.data.utils as du
from rnn.utils import cprint
from rnn.logging import setup_logger

DATA_FILENAMES = ['train.csv']


class CLI:
    def __init__(self):
        parser = argparse.ArgumentParser(description='Prepare data for the model.')
        parser.add_argument('--data_dir', '-d', metavar='DATA_DIR', default=None, type=str, help='data directory')

        args = parser.parse_args()

        self.data_dir = args.data_dir


def main():
    setup_logger()
    cli = CLI()
    if cli.data_dir is None:
        data_dir = du.get_data_dir()
    else:
        data_dir = Path(cli.data_dir)
    data_filenames = DATA_FILENAMES
    cprint("Using '{}' as data directory.".format(str(data_dir)))
    cprint("Preprocessing files {}.".format(data_filenames))
    preprocessing.prepare_data(data_filenames, data_dir, train_fraction=0.85)


if __name__ == '__main__':
    main()
