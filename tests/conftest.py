# Copyright (c) 2020 Matt Struble. All Rights Reserved.
#
# Use is subject to license terms.
#
# Author: Matt Struble
# Date: Dec. 29 2020

import json
import os
import tempfile

import pytest

from furcate.config_reader import ConfigReader


@pytest.fixture(scope="class")
def basic_config():
    config = {"data_name": "test", "data_dir": "foo", "batch_size": 32, "epochs": 10}

    path = make_tmpfile(config)
    yield config, path
    close_tmpfile(path)


@pytest.fixture(scope="class")
def broken_config(basic_config):
    config = {"data_dir": "foo", "batch_size": 32, "epochs": 10}

    path = make_tmpfile(config)
    yield config, path
    close_tmpfile(path)


@pytest.fixture(scope="class")
def multiple_runs_config():
    config = {
        "data_name": "test",
        "data_dir": "foo",
        "batch_size": [16, 32, 64],
        "epochs": [10, 20, 30],
    }

    path = make_tmpfile(config)
    yield config, path
    close_tmpfile(path)


@pytest.fixture(scope="class")
def unique_keys_config(basic_config):
    config = {
        "data_name": "test",
        "data_dir": "foo",
        "batch_size": 32,
        "epochs": 10,
        "unique": "foo",
        "test": "bar",
    }

    path = make_tmpfile(config)
    yield config, path
    close_tmpfile(path)


@pytest.fixture(scope="class")
def combination_config():
    config = {
        "data_name": "test",
        "data_dir": "foo",
        "batch_size": [16, 32, 64],
        "epochs": [10, 20, 30],
        "unique": True,
        "test": ["bar", "foo", "baz"],
    }

    path = make_tmpfile(config)
    yield config, path
    close_tmpfile(path)


@pytest.fixture(scope="class")
def excluded_config():
    config = {
        "data_name": "test",
        "data_dir": "foo",
        "batch_size": [16, 32, 64],
        "epochs": [10, 20, 30],
        "unique": True,
        "test": ["bar", "foo", "baz"],
        "meta": {
            "exclude_configs": [
                {"batch_size": 32},
                {"test": "foo", "epochs": 10, "batch_size": 16},
            ]
        },
    }

    path = make_tmpfile(config)
    yield config, path
    close_tmpfile(path)


def make_tmpfile(config):
    fd, tmp_path = tempfile.mkstemp(prefix="furcate_tests")

    with os.fdopen(fd, "w") as tmp:
        json.dump(config, tmp)

    return tmp_path


def close_tmpfile(tmp_path):
    os.remove(tmp_path)


@pytest.fixture(
    params=[
        "basic_config",
        "multiple_runs_config",
        "unique_keys_config",
        "combination_config",
    ],
    scope="class",
)
def config_init(request):
    config, path = request.getfixturevalue(request.param)
    request.cls.config_reader = ConfigReader(path)
    request.cls.config = config
    request.cls.config_path = path
    yield


@pytest.mark.usefixtures("config_init")
class ConfigLoader:
    pass
