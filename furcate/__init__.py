# Copyright (c) 2020 Matt Struble. All Rights Reserved.
#
# Use is subject to license terms.
#
# Author: Matt Struble
# Date: Nov. 18 2020
from importlib import import_module

from .fork import Fork  # noqa. F401
from .gpu_helper import get_gpu_stats  # noqa. F401

# Import different versions of furcate with specific frameworks overriding base furcate
modules = ["furcate.furcate-tf.fork"]

for module in modules:
    try:
        lib = import_module(module)
    except ImportError:
        pass
    else:
        globals()["Fork"] = lib.Fork
