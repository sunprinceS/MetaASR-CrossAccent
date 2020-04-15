import tqdmlogger
from tqdmlogger.ansistyle import stylize, fg, bg, attr, RESET

PREFIX_MAP = {
    'info': ('Info','yellow'),
    'train': ('Training', 'cyan'),
    'test': ('Testing', 'green'),
    'debug': ('DEBUG', 'magenta'),
}

def flush():
    tqdmlogger.flush()

def log(*args, prefix=None, update=False):
    if prefix is None:
        tqdmlogger.log(*args)
    elif prefix in PREFIX_MAP:
        tqdmlogger.seclog(PREFIX_MAP[prefix], *args, update=update)
    else:
        tqdmlogger.seclog(prefix, *args, update=update)


def log_info(info, prefix):
    msg = '\n'.join(f'{k.ljust(15)}: {v:.6f}' for k,v in info.items())
    log(msg, prefix=prefix, update=True)

def warning(msg):
    tqdmlogger.logger.info(stylize(f'Warning: {msg}', fg('red'), attr('bold')))

def notice(msg):
    tqdmlogger.logger.info(stylize(msg, bg('blue')))

def show(msg,prefix):
    log(msg, prefix=prefix)
