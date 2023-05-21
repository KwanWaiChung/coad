import logging
import logging.config
import datetime
import os


level_map = {
    "critical": logging.CRITICAL,
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}


def getlogger(
    name: str = None,
    logger_level: str = None,
    console_level: str = None,
    file_level: str = None,
    log_path: str = None,
):
    """Configure the logger and return it.

    Args:
        name (str, optional): Name of the logger, usually __name__.
            Defaults to None. None refers to root logger, usually useful
            when setting the default config at the top level script.
        logger_level (str, optional): level of logger. Defaults to None.
            None is treated as `debug`.
        console_level (str, optional): level of console. Defaults to None.
            None is treated as `debug`.
        file_level (str, optional): level of file. Defaults to None.
            None is treated `debug`.
        log_path (str, optional): The path of the log.
            None is treated as 'log-{datetime}.txt'.

    Note that console_level should only be used when configuring the
    root logger.
    """
    logger = logging.getLogger(name)
    if name:
        logger.setLevel(level_map[logger_level or "debug"])
    else:  # root logger default lv should be high to avoid external lib log
        logger.setLevel(level_map[logger_level or "warning"])

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # set up the logfile handler
    if file_level:
        logTime = datetime.datetime.now()
        log_path = log_path or "log.txt"
        fn1, fn2 = os.path.splitext(log_path)
        log_filename = f"{fn1}-{logTime.strftime('%Y%m%d-%H%M%S')}{fn2}"
        os.makedirs(os.path.dirname(log_filename), exist_ok=True)
        fh = logging.FileHandler(log_filename)
        fh.setLevel(level_map[file_level or "debug"])
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    # set up the console/stream handler
    if name and console_level:
        raise ValueError(
            "`console_level` should only be set when configuring root logger."
        )
    if console_level:
        sh = logging.StreamHandler()
        sh.setLevel(level_map[console_level or "debug"])
        sh.setFormatter(formatter)
        logger.addHandler(sh)
    return logger
