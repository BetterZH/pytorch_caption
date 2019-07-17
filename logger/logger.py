import os
import logging

g_logger = None

def init_logger(opt):

    global g_logger

    g_logger = logging.getLogger()
    g_logger.setLevel(logging.INFO)

    if not os.path.exists(opt.eval_result_path):
        os.makedirs(opt.eval_result_path)

    logfile = os.path.join(opt.eval_result_path, opt.id + '.log')
    fh = logging.FileHandler(logfile, mode='a')
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    g_logger.addHandler(fh)
    g_logger.addHandler(ch)

def info(msg):

    global g_logger
    if g_logger is None:
        raise Exception("g_logger is not init")

    g_logger.info(msg)
