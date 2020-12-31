# Copyright (c) 2020 Matt Struble. All Rights Reserved.
#
# Use is subject to license terms.
#
# Author: Matt Struble
# Date: Dec. 30 2020

import os
import threading

import pandas as pd
import pytest

from furcate.runner import config_to_csv, seconds_to_string


@pytest.mark.parametrize(
    "seconds,expected",
    [(34, "34s"), (124, "2m 4s"), (7224, "2h 0m 24s"), (178560, "2d 1h 36m 0s")],
)
def test_seconds_to_string(seconds, expected):
    assert expected == seconds_to_string(seconds)


def test_config_to_csv(log_basic_config_reader):
    config, config_reader = log_basic_config_reader
    log_dir = os.path.dirname(config_reader.data["log_dir"])

    assert 1 == len(os.listdir(log_dir))

    config_to_csv(config_reader)

    assert 2 == len(os.listdir(log_dir))
    assert "run_data.csv" in os.listdir(log_dir)

    df = pd.read_csv(os.path.join(log_dir, "run_data.csv"))

    assert 1 == len(df)

    for key in config_reader.data:
        assert key in df.head()

    records = df.to_dict("records")

    assert 1 == len(records)

    for record in records:
        for key in config_reader.data:
            assert config_reader.data[key] == record[key]


def test_multiple_config_to_csv(log_basic_config_reader):
    config, config_reader = log_basic_config_reader
    log_dir = os.path.dirname(config_reader.data["log_dir"])

    for _ in range(5):
        config_to_csv(config_reader)

    df = pd.read_csv(os.path.join(log_dir, "run_data.csv"))
    assert 5 == len(df)

    records = df.to_dict("records")

    assert 5 == len(records)

    for record in records:
        for key in config_reader.data:
            assert config_reader.data[key] == record[key]


def test_concurrent_config_to_csv(log_basic_config_reader):
    config, config_reader = log_basic_config_reader
    log_dir = os.path.dirname(config_reader.data["log_dir"])

    num_threads = 10
    num_calls = 100

    def thread_function(num, cr):
        for _ in range(num):
            config_to_csv(cr)

    threads = []
    for _ in range(num_threads):
        x = threading.Thread(target=thread_function, args=(num_calls, config_reader))
        x.start()
        threads.append(x)

    for t in threads:
        t.join()

    assert 2 == len(os.listdir(log_dir))
    assert "run_data.csv" in os.listdir(log_dir)

    df = pd.read_csv(os.path.join(log_dir, "run_data.csv"))
    assert num_calls * num_threads == len(df)

    for record in df.to_dict("records"):
        for key in config_reader.data:
            assert config_reader.data[key] == record[key]
