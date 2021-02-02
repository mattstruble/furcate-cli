# Copyright (c) 2020 Matt Struble. All Rights Reserved.
#
# Use is subject to license terms.
#
# Author: Matt Struble
# Date: Dec. 30 2020

import copy
import datetime
import json
import os
import threading
from copy import deepcopy

import pandas as pd
import pytest

from furcate.runner import (
    TrainingThread,
    config_to_csv,
    get_num_completed_runs,
    get_run_data_csv_path,
    seconds_to_string,
)

from .conftest import ThreadHelper


@pytest.mark.parametrize(
    "seconds,expected",
    [(34, "34s"), (124, "2m 4s"), (7224, "2h 0m 24s"), (178560, "2d 1h 36m 0s")],
)
def test_seconds_to_string(seconds, expected):
    """
    Assert that the passed in seconds is converted to the proper string representation.
    :param seconds: Seconds (int)
    :param expected: String in format of "Xh Xm Xs"
    """
    assert expected == seconds_to_string(seconds)


def test_get_run_data_csv_path(log_basic_config_reader):
    """
    Assert that the run_data.csv path generation works from the parent folder as well as subdirs.
    """
    config, config_reader = log_basic_config_reader

    expected_path = os.path.join(config_reader.data["log_dir"], "run_data.csv")
    actual_path = get_run_data_csv_path(config_reader)

    assert expected_path == actual_path

    config_reader.data["log_dir"] = os.path.join(
        config_reader.data["log_dir"], "test0_dd10_123"
    )

    actual_path = get_run_data_csv_path(config_reader, has_subdir=True)

    assert expected_path == actual_path


def test_get_num_completed_runs(log_basic_config_reader):
    """
    Assert that the number of completed runs is properly read from the file.
    """
    config, config_reader = log_basic_config_reader

    run_data_path = get_run_data_csv_path(config_reader)

    assert os.path.exists(run_data_path) is False
    assert 0 == get_num_completed_runs(config_reader)

    expected_runs = 5
    with open(run_data_path, "w") as f:
        f.write("header,csv,title,names\n")
        for _ in range(expected_runs):
            f.write("test1,test2,test4,5\n")

    assert expected_runs == get_num_completed_runs(config_reader)


def test_config_to_csv(log_basic_config_reader):
    """
    Assert that the configurations are properly saved as csv files and can be re-read in as DataFrames.
    """
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


def test_finished_run_config_to_csv(finished_run_config_reader):
    """
    Asserts that a finished run config is translated into run_{} records in the final csv.
    """
    config, config_reader = finished_run_config_reader
    log_dir = os.path.dirname(config_reader.data["log_dir"])

    run_data = deepcopy(config_reader.meta_data["data"])

    assert len(config_reader.meta_data["data"]) > 0

    config_to_csv(config_reader)

    for key in run_data:
        assert "run_" + key in config_reader.data

    df = pd.read_csv(os.path.join(log_dir, "run_data.csv"))

    for key in config_reader.data:
        assert key in df.head()

    records = df.to_dict("records")

    for record in records:
        for key in config_reader.data:
            assert config_reader.data[key] == record[key]


def test_multiple_config_to_csv(log_basic_config_reader):
    """
    Assert that multiple reads and writes are supported by config_to_csv.
    """
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
    """
    Assert that concurrent read and writes are supported by config_to_csv.
    """
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


@pytest.mark.usefixtures("log_config_class_init")
class TestTrainingThread(ThreadHelper):
    def _setup(self, thread_id=0, script_name="foo", log_keys=[]):
        self.training_thread = TrainingThread(
            thread_id, self.config, script_name, log_keys
        )
        self.training_thread.config["gpu_id"] = None

    def _wait_for_init(self):
        self.wait_for_init(self.training_thread)

    def _wait_for_shutdown(self):
        self.wait_for_shutdown(self.training_thread)

    def _teardown(self):
        self._wait_for_shutdown()
        assert self.training_thread.is_alive() is False

    def test_init(self):
        """
        Assert that the TrainingThread is initialized with the properly values.
        """
        thread_id = 0
        script_name = "foo"
        log_keys = ["foo", "bar"]
        self._setup(thread_id=thread_id, script_name=script_name, log_keys=log_keys)

        assert self.training_thread.thread_id == thread_id
        assert self.training_thread.config == self.config
        assert self.training_thread.script_name == script_name
        assert self.training_thread.log_keys == log_keys

        assert self.training_thread.dir_name == os.path.basename(
            self.config["data_dir"]
        )
        assert self.training_thread.name == self.config["data_name"] + str(thread_id)
        assert self.training_thread.run_time == datetime.timedelta(0)
        assert self.training_thread._running is False

    def test_gen_log_dir_creates_dirs(self):
        """
        Assert that generating the log_dir path also creates the directory.
        """
        self._setup()
        thread_name = self.training_thread.name
        thread_dir_name = self.training_thread.dir_name

        expected_path = os.path.join(
            self.config["log_dir"], "{}_{}".format(thread_name, thread_dir_name)
        )

        assert not os.path.exists(expected_path)
        self.training_thread._gen_log_dir()
        assert os.path.exists(expected_path)

    def test_gen_log_dir_no_keys(self):
        """
        Assert that if not presented with any unique keys a log dir is still generated without error.
        """
        self._setup()
        thread_name = self.training_thread.name
        thread_dir_name = self.training_thread.dir_name

        expected_path = os.path.join(
            self.config["log_dir"], "{}_{}".format(thread_name, thread_dir_name)
        )

        self.training_thread._gen_log_dir()
        actual_path = self.training_thread.config["log_dir"]

        assert expected_path == actual_path

    def test_gen_log_dir_key_shortening(self):
        """
        Assert that when presented with unique keys they are shortened properly with the correct data values formatted.
        """
        log_keys = ["data_name"]
        self._setup(log_keys=log_keys)

        thread_name = self.training_thread.name
        thread_dir_name = self.training_thread.dir_name

        folder_name = "{}_{}_{}{}".format(
            thread_name, thread_dir_name, "dn", self.config["data_name"]
        )

        expected_path = os.path.join(self.config["log_dir"], folder_name)
        self.training_thread._gen_log_dir()
        actual_path = self.training_thread.config["log_dir"]
        assert expected_path == actual_path

    def test_gen_log_dir_ignores_data_dir(self):
        """
        Asserts that if data_dir is present as a key it isn't included in the key shortening logic.
        """
        log_keys = ["data_dir"]
        self._setup(log_keys=log_keys)

        thread_name = self.training_thread.name
        thread_dir_name = self.training_thread.dir_name

        folder_name = "{}_{}".format(thread_name, thread_dir_name)

        expected_path = os.path.join(self.config["log_dir"], folder_name)
        self.training_thread._gen_log_dir()
        actual_path = self.training_thread.config["log_dir"]
        assert expected_path == actual_path

    def test_gen_log_dir_all_keys(self):
        """
        Assert that all configuration keys can be converted and create a proper folder structure.
        """
        log_keys = list(self.config.keys())
        log_keys.remove("log_dir")

        self._setup(log_keys=log_keys)

        thread_name = self.training_thread.name
        thread_dir_name = self.training_thread.dir_name

        folder_name = "{}_{}".format(thread_name, thread_dir_name)

        for key in log_keys:
            if key != "data_dir":
                short = "".join([s[0] for s in key.split("_")])
                value = str(self.config[key]).replace(".", "-")
                folder_name += "_{}{}".format(short, value)

        expected_path = os.path.join(self.config["log_dir"], folder_name)
        self.training_thread._gen_log_dir()
        actual_path = self.training_thread.config["log_dir"]
        assert expected_path == actual_path

    @pytest.mark.parametrize("config_path", ("foo", "bar", 5))
    def test_generate_run_command(self, config_path):
        """
        Asserts that the run command can be created using a varity of different configuration paths.
        :param config_path: Path to be inserted into the run command.
        """
        self._setup()
        expected_command = 'python3 {} --config "{}" --name "{}" --id "{}"'.format(
            self.training_thread.script_name,
            config_path,
            self.training_thread.name,
            self.training_thread.thread_id,
        )

        actual_command = self.training_thread._generate_run_command(config_path)
        assert expected_command == actual_command

    @pytest.mark.parametrize("gpu_id", (-1, 2, 30000000, "gpu", None))
    def test_generate_run_command_with_gpu(self, gpu_id):
        """
        Asserts that the gpu_id can be inserted and removed properly from the run command without error.
        :param gpu_id: ID to be inserted into the run command.
        """
        self._setup()
        self.training_thread.config["gpu_id"] = gpu_id
        config_path = "foo"

        expected_command = 'python3 {} --config "{}" --name "{}" --id "{}"'.format(
            self.training_thread.script_name,
            config_path,
            self.training_thread.name,
            self.training_thread.thread_id,
        )

        if gpu_id is not None:
            expected_command += ' --gpu "{}"'.format(gpu_id)

        actual_command = self.training_thread._generate_run_command(config_path)
        assert expected_command == actual_command

    def test_run(self):
        """
        Assert that run creates the log_dir, populates it with .log and .err files, and updates run_time.
        :return:
        """
        self._setup()

        thread_name = self.training_thread.name
        thread_dir_name = self.training_thread.dir_name

        log_dir = os.path.join(
            self.config["log_dir"], thread_name + "_" + thread_dir_name
        )

        assert not os.path.exists(log_dir)

        self.training_thread.start()
        self._wait_for_init()
        self._wait_for_shutdown()

        assert os.path.exists(log_dir)

        log_file = os.path.join(log_dir, self.training_thread.name + ".log")
        err_file = os.path.join(log_dir, self.training_thread.name + ".err")

        assert os.path.exists(log_file)
        assert os.path.exists(err_file)
        assert self.training_thread.run_time > datetime.timedelta(0)

    def test_run_with_run_data(self):
        """
        Asserts that when a run_data.json is created by a successful run that it is then stored as a CSV in the root dir.
        """

        def mock_command(config_path):
            return "python --version"

        self._setup()
        self.training_thread._generate_run_command = mock_command

        thread_name = self.training_thread.name
        thread_dir_name = self.training_thread.dir_name

        log_dir = os.path.join(
            self.config["log_dir"], thread_name + "_" + thread_dir_name
        )

        os.makedirs(log_dir, exist_ok=True)

        run_data = copy.deepcopy(self.config)
        run_data["log_dir"] = log_dir
        with open(os.path.join(log_dir, "run_data.json"), "w") as f:
            json.dump(run_data, f)

        self.training_thread.start()
        self._wait_for_init()
        self._wait_for_shutdown()

        run_data_csv_path = get_run_data_csv_path(self.config_reader)
        assert os.path.exists(run_data_csv_path)
