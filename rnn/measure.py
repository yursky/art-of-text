import argparse
from pathlib import Path
import csv

import rnn.data.utils as du
from rnn.metrics import confusion_matrix_from_iterables, Example, f1
from rnn.utils import cprint
from rnn.logging import setup_logger


def compare_files(target_path: Path, prediction_path: Path):
    with target_path.open() as target_file, prediction_path.open() as prediction_file:
        target_reader = csv.reader(target_file)
        prediction_reader = csv.reader(prediction_file)

        # skip the headers
        next(target_reader)
        next(prediction_reader)

        cm = confusion_matrix_from_iterables((Example(row[0], row[-1]) for row in target_reader),
                                             (Example(row[0], row[-1]) for row in prediction_reader))

    precision_score = cm.precision()
    recall_score = cm.recall()
    f1_score = f1(precision_score, recall_score)

    return {
        "f1": f1_score,
        "precision": precision_score,
        "recall": recall_score
    }


class CLI:
    def __init__(self):
        parser = argparse.ArgumentParser(description='Prepare data for the model.')
        parser.add_argument('--data_dir', '-d', metavar='DATA_DIR', default=None, type=str, help='data directory')
        parser.add_argument('--target_path', '-t', metavar='DATA_DIR', default=None, type=str,
                            help='path to a CSV file with the valid tags')
        parser.add_argument('--predictions_path', '-p', metavar='DATA_DIR', type=str, required=True,
                            help='path to a CSV file with the predicted tags')

        args = parser.parse_args()

        self.data_dir = args.data_dir
        self.target_path = args.target_path
        self.predictions_path = args.predictions_path


def main():
    setup_logger()
    cli = CLI()

    if cli.target_path is None:
        if cli.data_dir is None:
            data_dir = du.get_data_dir()
        else:
            data_dir = Path(cli.data_dir)
        target_path = data_dir / du.TEST_TAGS_CSV_SUBPATH
    else:
        target_path = Path(cli.target_path)
    predictions_path = Path(cli.predictions_path)

    cprint("Using '{}' as the targets.".format(str(target_path)))
    cprint("Using '{}' as the predictions.".format(str(predictions_path)))

    cprint(compare_files(target_path, predictions_path))


if __name__ == '__main__':
    main()
