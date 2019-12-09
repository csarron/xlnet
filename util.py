import logging

import tensorflow
import absl.flags

logger = logging.getLogger('eet')

logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(levelname)s:%(asctime)s.%(msecs)03d:%(pathname)s:"
                        "%(lineno)d: %(message)s", "%Y-%m-%d_%H:%M:%S")
handler = logging.StreamHandler()
handler.setFormatter(fmt)
logger.addHandler(handler)
logger.propagate = False

tf = tensorflow
flags = absl.flags


def abbreviate(x):
    abbreviations = ["", "K", "M", "B", "T"]
    thing = "1"
    a = 0
    while len(thing) <= len(str(x)) - 3:
        thing += "000"
        a += 1
    b = int(thing)
    thing = round(x / b, 2)
    return str(thing) + " " + abbreviations[a]
