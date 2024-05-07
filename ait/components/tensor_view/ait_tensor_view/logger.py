import logging


def get_logger():
    _logger = logging.getLogger("tensor-view")
    _logger.propagate = False
    _logger.setLevel(logging.INFO)
    if not _logger.handlers:
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(asctime)s][%(name)s] %(message)s')
        stream_handler.setFormatter(formatter)
        _logger.addHandler(stream_handler)
    return _logger


logger = get_logger()
