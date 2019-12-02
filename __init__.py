import tensorflow

import logging

logger = logging.getLogger('xlnet')

logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(levelname)s:%(asctime)s.%(msecs)03d:"
                        "%(pathname)s:%(lineno)d: %(message)s",
                        "%Y-%m-%d_%H:%M:%S")
handler = logging.StreamHandler()
handler.setFormatter(fmt)
logger.addHandler(handler)
logger.propagate = False

tf = tensorflow
