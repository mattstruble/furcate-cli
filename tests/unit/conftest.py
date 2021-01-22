# Copyright (c) 2020 Matt Struble. All Rights Reserved.
#
# Use is subject to license terms.
#
# Author: Matt Struble
# Date: Dec. 30 2020
import os
import shutil
import threading
import time

import pytest

from furcate.config_reader import ConfigReader
from tests.util import close_tmpfile, make_tmpdir, make_tmpfile


@pytest.fixture()
def basic_config_reader():
    config = {"data_name": "test", "data_dir": "foo", "batch_size": 32, "epochs": 10}

    path = make_tmpfile(config)
    config_reader = ConfigReader(path)

    yield config, config_reader

    close_tmpfile(path)


@pytest.fixture()
def log_basic_config_reader():
    base_dir = make_tmpdir()
    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(log_dir)

    config = {
        "data_name": "test",
        "log_dir": log_dir,
        "data_dir": "foo",
        "batch_size": 32,
        "epochs": 10,
    }

    path = make_tmpfile(config)
    config_reader = ConfigReader(path)

    yield config, config_reader

    close_tmpfile(path)
    close_tmpfile(base_dir)


@pytest.fixture()
def finished_run_config_reader():
    config = {
        "data_name": "test",
        "log_dir": "test/logs",
        "data_dir": "foo",
        "batch_size": 32,
        "epochs": 10,
        "meta": {"data": {"results": 10, "run_time": 1000}},
    }

    path = make_tmpfile(config)
    config_reader = ConfigReader(path)
    os.makedirs(config_reader.data["log_dir"])

    yield config, config_reader

    close_tmpfile(path)
    shutil.rmtree("test")


@pytest.fixture(scope="class")
def threading_event(request):
    request.cls.event = threading.Event()


@pytest.mark.usefixtures("threading_event")
class ThreadHelper:
    def wait_for_init(
        self, thread, attr="_running", wait_condition=False, timeout_seconds=5
    ):
        """
        Waits for MemoryTrace to start and run by checking if the start_stats have been set.
        :param timeout_seconds: Timeout to wait for thread to start
        """
        total_time = 0
        prev_time = time.time()
        while (
            attr and getattr(thread, attr) is wait_condition
        ) and total_time < timeout_seconds:
            self.event.wait(1)
            total_time += time.time() - prev_time
            prev_time = time.time()

    def wait_for_shutdown(self, thread, timeout_seconds=5):
        """
        Waits for MemoryTrace is_alive() to return false.
        :param timeout_seconds: Timeout to wait for thread to stop.
        """
        total_time = 0
        prev_time = time.time()
        while thread.is_alive() and total_time < timeout_seconds:
            self.event.wait(1)
            total_time += time.time() - prev_time
            prev_time = time.time()

    def wait_for_thread_update(self, thread, expected_delay, attr=None):
        timeout = expected_delay * 2
        wait_time = expected_delay / 3
        total_time = 0
        prev_value = getattr(thread, attr)
        prev_time = time.time()

        while prev_value == getattr(thread, attr) and total_time < timeout:
            self.event.wait(wait_time)
            total_time += time.time() - prev_time
            prev_time = time.time()

    def wait_for_delay(self, delay):
        """
        Waits for MemoryTrace to change stats based upon the configured delay. Timesout after delay*2 seconds.
        :param prev_stats: Previous stats to compare to current thread stats
        """
        total_time = 0
        prev_time = time.time()
        while total_time < delay:
            self.event.wait(delay / 3)
            total_time += time.time() - prev_time
            prev_time = time.time()
