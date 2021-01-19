# Copyright (c) 2020 Matt Struble. All Rights Reserved.
#
# Use is subject to license terms.
#
# Author: Matt Struble
# Date: Dec. 30 2020

import pytest

from furcate.config_reader import ConfigReader


@pytest.fixture()
def default_values():
    data = {
        "log_dir": "logs",
        "learning_rate": 0.001,
        "verbose": 2,
        "cache": False,
        "seed": 42,
        "prefetch": 1,
        "train_prefix": "test.train",
        "test_prefix": "test.test",
        "valid_prefix": "test.valid",
    }

    meta_data = {"allow_cpu": False, "exclude_configs": [], "mem_trace": False}

    return data, meta_data


def test_load_defaults(basic_config_reader, default_values):
    config, config_reader = basic_config_reader
    config_reader.data = config
    config_reader.meta_data = {}

    config_reader._load_defaults()

    data = config_reader.data
    meta_data = config_reader.meta_data

    expected_data, expected_meta_data = default_values

    for key in expected_data:
        assert expected_data[key] == data[key]

    for key in expected_meta_data:
        assert expected_meta_data[key] == meta_data[key]


def test_load_config(basic_config_reader):
    config, config_reader = basic_config_reader
    fname = config_reader.filename

    data, meta_data = config_reader._load_config(fname)

    for key in config:
        assert key in data
        assert config[key] == data[key]

    assert len(meta_data) == 0


@pytest.mark.parametrize("missing_key", ConfigReader._REQUIRED_KEYS)
def test_invalid_data(basic_config_reader, missing_key):
    config, config_reader = basic_config_reader

    config.pop(missing_key, None)

    try:
        config_reader._validate_data(config, "test")
        print("Expected ValueError")
        assert False
    except ValueError as ve:
        assert missing_key in str(ve)


@pytest.mark.parametrize(
    "foo_value,bar_value",
    [
        ([1, 2], 3),
        (1, 3),
        (1, [2, 3]),
        ([1, 2, 3], [4, 5, 6]),
        ("foo", [2]),
        (["one", "two"], True),
        (0.001, [0.1, 0.2, 0.3, 0.4]),
        ({"test": "val"}, 3),
        (None, None),
    ],
)
def test_gen_config_permutations(basic_config_reader, foo_value, bar_value):
    config, config_reader = basic_config_reader

    enumerated_data = [
        {"key": "foo", "value": foo_value},
        {"key": "bar", "value": bar_value},
    ]

    expected = []
    if isinstance(foo_value, list) and isinstance(bar_value, list):
        for foo in foo_value:
            for bar in bar_value:
                expected.append({"foo": foo, "bar": bar})
    elif isinstance(foo_value, list):
        for foo in foo_value:
            expected.append({"foo": foo, "bar": bar_value})
    elif isinstance(bar_value, list):
        for bar in bar_value:
            expected.append({"foo": foo_value, "bar": bar})
    else:
        expected.append({"foo": foo_value, "bar": bar_value})

    actual = config_reader._gen_config_permutations(0, {}, enumerated_data)

    assert len(expected) == len(actual)

    for exp in expected:
        found = False
        for act in actual:
            found = False
            for key in exp:
                if exp[key] == act[key]:
                    found = True
                else:
                    found = False
                    break

            if found:
                break

        assert found is True


def test_gen_run_configs(basic_config_reader):
    config, config_reader = basic_config_reader

    data = {"foo": [1, 2], "bar": ["test", "bazz"]}
    expected = [
        {"foo": 1, "bar": "test"},
        {"foo": 1, "bar": "bazz"},
        {"foo": 2, "bar": "test"},
        {"foo": 2, "bar": "bazz"},
    ]

    actual = config_reader._gen_run_configs(data)

    assert len(expected) == len(actual)

    for exp in expected:
        found = False
        for act in actual:
            found = False
            for key in exp:
                if exp[key] == act[key]:
                    found = True
                else:
                    found = False
                    break

            if found:
                break

        assert found is True
