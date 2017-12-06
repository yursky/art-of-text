import sys
from copy import deepcopy


if sys.stdout.isatty():
    RED_FONT_ESC = "\033[1;31m"
    GREEN_FONT_ESC = "\033[1;32m"
    DEFAULT_FONT_ESC = "\033[0;0m"
else:
    RED_FONT_ESC = ""
    GREEN_FONT_ESC = ""
    DEFAULT_FONT_ESC = ""


class DictWrapper:
    @classmethod
    def from_dict(cls, source):
        ret = cls()
        ret.__dict__.update(source)
        return ret

    def as_dict(self) -> dict:
        return deepcopy(self.__dict__)


def cprint(*args, **kwargs):
    line = (RED_FONT_ESC, '-=>' + GREEN_FONT_ESC) + args + (DEFAULT_FONT_ESC,)
    print(*line, **kwargs)
