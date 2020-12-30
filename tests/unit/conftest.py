# Copyright (c) 2020 Matt Struble. All Rights Reserved.
#
# Use is subject to license terms.
#
# Author: Matt Struble
# Date: Dec. 30 2020

import json
import os
import tempfile

import pytest

from furcate.config_reader import ConfigReader


@pytest.fixture(scope="class")
def basic_config_reader():
    config = {"data_name": "test", "data_dir": "foo", "batch_size": 32, "epochs": 10}

    fd, tmp_path = tempfile.mkstemp(prefix="furcate_tests")

    try:
        with os.fdopen(fd, "w") as tmp:
            json.dump(config, tmp)

        config_reader = ConfigReader(tmp_path)

        yield config, config_reader

    finally:
        os.remove(tmp_path)
