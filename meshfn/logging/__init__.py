import functools

from meshfn.logging.logger import DistributedLogger, make_logger


@functools.lru_cache(maxsize=128)
def get_logger(
    parallel_context=None,
    name="main",
    mode="rich",
    abbrev_name=None,
    keywords=("INIT", "FROM"),
):
    if parallel_context is not None:
        return DistributedLogger(name, parallel_context, mode, abbrev_name, keywords)

    return make_logger(name, mode, abbrev_name, keywords)


logger = get_logger()
