import logging


def setup_logger():
    root_formatter = logging.Formatter('%(asctime)s : %(name)s : %(levelname)s : %(message)s')
    std_handler = logging.StreamHandler()
    std_handler.setLevel(logging.INFO)
    std_handler.setFormatter(root_formatter)

    log = logging.getLogger()
    log.setLevel(logging.INFO)
    log.addHandler(std_handler)
