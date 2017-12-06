import argparse
import ast
from contextlib import suppress

import rnn.data.utils as io


def create_parser(allowed_actions, allowed_params) -> argparse.ArgumentParser:

    def parse_override(expr):
        try:
            key, value = expr.split('=')
            if key not in allowed_params:
                raise argparse.ArgumentTypeError("no such parameter '{}'".format(key))
            with suppress(SyntaxError):
                value = ast.literal_eval(value)
            return key, value
        except ValueError:
            raise argparse.ArgumentTypeError("failed to parse override '{}'".format(expr))

    def add_common_arguments(parser: argparse.ArgumentParser):
        parser.add_argument('--name', '-n', metavar='NAME', default=None, type=str, help='model name')
        params_help = 'parameters overrides. Available parameters: ' + ', '.join(allowed_params)
        parser.add_argument('--params', '-p', metavar='PARAM=VALUE',
                            default=[], type=parse_override, nargs="*",
                            help=params_help)

    main_parser = argparse.ArgumentParser(description='Train and evaluate TensorFlow model.')
    subparsers = main_parser.add_subparsers(title='action', dest="action")
    subparsers.required = True
    action_parsers = [subparsers.add_parser(action) for action in allowed_actions]
    for action_parser in action_parsers:
        add_common_arguments(action_parser)
    add_common_arguments(main_parser)

    return main_parser


class CLI:
    def __init__(self, allowed_actions, allowed_params):
        parser = create_parser(allowed_actions, allowed_params)
        args = parser.parse_args()

        self.action = args.action
        self.model_dir = io.get_model_dir(args.name)
        self.overrides = dict(args.params)
