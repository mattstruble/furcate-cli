# Copyright (c) 2020 Matt Struble. All Rights Reserved.
#
# Use is subject to license terms.
#
# Author: Matt Struble
# Date: Dec. 30 2020
import pytest

from furcate.config_reader import ConfigReader
from tests.util import close_tmpfile, make_tmpfile


@pytest.fixture(scope="class")
def basic_config_reader():
    config = {"data_name": "test", "data_dir": "foo", "batch_size": 32, "epochs": 10}

    path = make_tmpfile(config)
    yield config, ConfigReader(path)
    close_tmpfile(path)
