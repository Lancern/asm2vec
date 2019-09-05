import logging


def asm2vec_logger() -> logging.Logger:
    return logging.getLogger('asm2vec')


def config_asm2vec_logging(**kwargs):
    level = kwargs.get('level', logging.INFO)
    handlers = kwargs.get('handlers', [logging.StreamHandler()])
    filters = kwargs.get('filters', [])

    asm2vec_logger().setLevel(level)
    for hd in handlers:
        asm2vec_logger().addHandler(hd)
    for ft in filters:
        asm2vec_logger().addFilter(ft)
