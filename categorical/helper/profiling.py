import logging
from logging.config import dictConfig
from functools import wraps
from time import process_time, perf_counter

# Global object
loglevel_this = logging.WARN
dictConfig(
    dict(
        version=1,
        formatters={
            "f": {"format": "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"}
        },
        handlers={
            "h": {"class": "logging.StreamHandler", "formatter": "f", "level": loglevel_this}
        },
        root={"handlers": ["h"], "level": loglevel_this},
    )
)
logger = logging.getLogger("Profiling")

def timing(method):
    @wraps(method)
    def wrap(*args, **kw):
        wall_ts = perf_counter()
        pu_ts = process_time()
        return_value = method(*args, **kw)
        wall_te = perf_counter()
        pu_te = process_time()
        logger.debug(
            f"Method:{method.__name__}"
            f" took {wall_te-wall_ts:2.4f}sec ({pu_te-pu_ts:2.4f}sec)"
        )
        return return_value

    return wrap
