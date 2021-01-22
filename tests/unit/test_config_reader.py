# Copyright (c) 2020 Matt Struble. All Rights Reserved.
#
# Use is subject to license terms.
#
# Author: Matt Struble
# Date: Dec. 30 2020

import csv
import os
import time
from pathlib import Path

import pytest

from furcate.config_reader import ConfigReader, ConfigWatcher

from .conftest import ThreadHelper


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
    """
    Asserts that the config reader properly loads in default values for missing config fields.
    """
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
    """
    Asserts that the load_config method properly reads in the data from the file.
    """
    config, config_reader = basic_config_reader
    fname = config_reader.filename

    data, meta_data = config_reader._load_config(fname)

    for key in config:
        assert key in data
        assert config[key] == data[key]

    assert len(meta_data) == 0


@pytest.mark.parametrize("missing_key", ConfigReader._REQUIRED_KEYS)
def test_invalid_data(basic_config_reader, missing_key):
    """
    Asserts that ConfigReader throws a value error if any of the required keys are missing.
    """
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
    """
    Asserts that the config permutation generator generates all permutations of the supplied data based on its format.
    """
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
    """
    Asserts that the generated permutations are broken up into independent run configurations.
    """
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


@pytest.fixture(params=["log_basic_config_reader"])
def config_watcher_basic_init(request):
    config, config_reader = request.getfixturevalue(request.param)
    request.cls.config_reader = config_reader
    request.cls.config = config
    yield


@pytest.mark.usefixtures("config_watcher_basic_init")
class TestConfigWatcher(ThreadHelper):
    def _setup(self, refresh_rate=1):
        self.config_watcher = ConfigWatcher(self.config_reader, refresh_rate)
        self.config_watcher.start()

    def _wait_for_init(self):
        self.wait_for_init(self.config_watcher)

    def _wait_for_thread_update(self):
        self.wait_for_thread_update(
            self.config_watcher,
            expected_delay=self.config_watcher.refresh_rate,
            attr="flagged",
        )

    def _wait_for_shutdown(self):
        self.wait_for_shutdown(self.config_watcher)

    def _teardown(self):
        """
        Stops the MemoryTrace thread and asserts that it shutsdown properly.
        """
        self.config_watcher.stop()
        assert self.config_watcher._running is False
        self._wait_for_shutdown()
        assert self.config_watcher.is_alive() is False

        if os.path.exists(
            os.path.join(self.config_reader.data["log_dir"], "run_data.csv")
        ):
            os.remove(os.path.join(self.config_reader.data["log_dir"], "run_data.csv"))

    def test_get_config_reader(self):
        """
        Asserts get_config_reader returns the proper config_reader object.
        """
        self._setup()

        assert self.config_reader == self.config_watcher.get_config_reader()

        self._teardown()

    def test_reset_flagged(self):
        """
        Asserts reset_flagged properly resets the flagged state of the thread.
        """
        self._setup()

        assert self.config_watcher.flagged is False
        self.config_watcher.flagged = True
        self.config_watcher.reset_flagged()
        assert self.config_watcher.flagged is False

        self._teardown()

    @pytest.mark.parametrize("refresh_rate", (5, 10))
    def test_config_update(self, refresh_rate):
        """
        Asserts that the ConfigWatcher thread detects file changes and updates in time with the configured delay.
        :param refresh_rate:
        :return:
        """
        self._setup(refresh_rate)
        self._wait_for_init()

        prev_mtime = self.config_watcher._mtime

        # Wait for one update cycle to pass to better time the touch response
        self._wait_for_thread_update()

        Path(self.config_reader.filename).touch()
        start_time = time.time()
        self._wait_for_thread_update()
        end_time = time.time()

        # Assert the time taken for the thread to update is close to the refresh rate
        assert abs((end_time - start_time) - refresh_rate) < 1
        assert prev_mtime != self.config_watcher._mtime
        assert self.config_watcher.flagged is True

        self._teardown()

    def test_remove_completed_runs(self):
        """
        Assert remove_completed_runs reads in the run_data.csv from the log_dir and removes duplicate runs from run_configs.
        :return:
        """
        self._setup()
        self._wait_for_init()

        self._gen_run_data()

        run_configs, _ = self.config_watcher.get_config_reader().gen_run_configs()

        assert len(run_configs) == 1

        self.config_watcher._remove_completed_runs()

        run_configs, _ = self.config_watcher.get_config_reader().gen_run_configs()

        assert len(run_configs) == 0

        self._teardown()

    def test_remove_completed_runs_empty_dir(self):
        """
        Asserts that if run_data.csv doesn't exist that no errors are thrown, nor is the run_configs changed.
        """
        self._setup()
        self._wait_for_init()

        run_configs, _ = self.config_watcher.get_config_reader().gen_run_configs()

        assert len(run_configs) == 1

        self.config_watcher._remove_completed_runs()

        run_configs, _ = self.config_watcher.get_config_reader().gen_run_configs()

        assert len(run_configs) == 1

        self._teardown()

    def test_remove_completed_runs_empty_run_configs(self):
        """
        Asserts that if the existing run_configs are empty that no errors are thrown, nor is the run_configs changed.
        """
        self._setup()
        self._wait_for_init()

        self.config_watcher.config_reader._generated = True
        _, _ = self.config_watcher.config_reader.gen_run_configs()
        self.config_watcher.config_reader.run_configs = []
        run_configs, _ = self.config_watcher.config_reader.gen_run_configs()

        assert len(run_configs) == 0

        self.config_watcher._remove_completed_runs()

        self._teardown()

    def test_remove_completed_runs_on_init(self):
        """
        Asserts that on initializing ConfigWatcher will remove completed runs.
        """
        self._gen_run_data()
        self._setup()
        self._wait_for_init()

        run_configs, _ = self.config_watcher.get_config_reader().gen_run_configs()

        assert len(run_configs) == 0

        self._teardown()

    @pytest.mark.xfail
    def test_remove_completed_runs_on_touch(self):
        """
        Asserts that after detecting a file change and reloading the file previous runs will be excluded from the generated config.

        * Currently marked xfail due to odd race conditions in CI environment.
        """
        self._setup()
        self._wait_for_init()

        run_configs, _ = self.config_watcher.get_config_reader().gen_run_configs()
        assert len(run_configs) == 1

        self._gen_run_data()
        self.config_watcher.reset_flagged()
        Path(self.config_reader.filename).touch()
        self._wait_for_thread_update()
        self.wait_for_delay(10)

        run_configs, _ = self.config_watcher.get_config_reader().gen_run_configs()
        assert len(run_configs) == 0

        self._teardown()

    def _gen_run_data(self):
        """
        Creates a run_data.csv in the correct location for the ConfigWatcher to detect when removing completed runs.
        """
        run_configs, _ = self.config_reader.gen_run_configs()
        log_dir = self.config_reader.data["log_dir"]

        assert len(run_configs) == 1

        with open(os.path.join(log_dir, "run_data.csv"), "w") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.config_reader.data.keys())
            writer.writeheader()
            writer.writerow(self.config_reader.data)
