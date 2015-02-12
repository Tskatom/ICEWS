#!/usr/bin/env python

import sys
import re
import os
import os.path
import logging

def get_log_file(logfile=None):
    if logfile:
        lgf = logfile
    elif os.environ.get('UPSTART_JOB'):
        lgf = os.environ['UPSTART_JOB']
    else:
        lgf = sys.argv[0]

    (d, p) = os.path.split(lgf)
    f = re.sub(r'(\.py)?$', '.log', p)

    path = os.path.join(os.getcwd(), 'logs')
    if not os.path.exists(path):
        path = os.path.join(os.path.realpath(d), 'logs')
        if not os.path.exists(path):
            path = os.path.realpath(d)

    return os.path.join(path, f)

def init(args=None,l=logging.INFO, logfile=None):
    '''
    Initialize a log in the logs directory using the name of the program.
    It first looks for logs in the current directory (typically ${HOME} in production).
    Then in the same directory as the program.
    The fallback is a file with the same name as the program and .log appended.
    '''
    lf = get_log_file(logfile)

    if args and vars(args).get('verbose', False):
        l = logging.DEBUG

    logging.basicConfig(filename=lf,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(process)d - %(thread)d - %(funcName)s - %(message)s',
                        level=l)

def getLogger(log_name):
    return logging.getLogger(log_name)
