# Copyright (c) 2020 Matt Struble. All Rights Reserved.
#
# Use is subject to license terms.
#
# Author: Matt Struble
# Date: Nov. 18 2020
from importlib import import_module

modules = ["furcate.fork", "furcate.furcate-tf.fork"]

for module in modules:
    print(module)
    try:
        lib = import_module(module)
    except ImportError:
        import sys

        print(sys.exc_info())
        pass
    else:
        globals()["Fork"] = lib.Fork
