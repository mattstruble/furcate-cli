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


@pytest.fixture(params=["log_basic_config_reader"])
def log_config_class_init(request):
    config, config_reader = request.getfixturevalue(request.param)
    request.cls.config_reader = config_reader
    request.cls.config = config
    yield


@pytest.fixture(scope="class")
def threading_event(request):
    request.cls.event = threading.Event()


@pytest.mark.usefixtures("threading_event")
class ThreadHelper:
    def wait_for_init(
        self, thread, attr="_running", wait_condition=False, timeout_seconds=5
    ):
        """
        Waits for the thread to initialize by waiting for the attr to not equal the wait condition.
        :param thread: Thread to wait for update.
        :param attr: Attribute to watch for change.
        :param wait_condition: Condition to wait on so long as attribute equals.
        :param timeout_seconds: Max seconds to wait before timeout.
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
        Waits for Thread is_alive() to return false.
        :param thread: Thread to wait for shutdown.
        :param timeout_seconds: Timeout to wait for thread to stop.
        """
        total_time = 0
        prev_time = time.time()
        while thread.is_alive() and total_time < timeout_seconds:
            self.event.wait(1)
            total_time += time.time() - prev_time
            prev_time = time.time()

    def wait_for_thread_update(self, thread, expected_delay, attr=None):
        """
        Waits for the provided thread's attribute to change. Timesout after 2x expected_delay.
        :param thread: Thread to wait on.
        :param expected_delay: (int) Expected time in seconds for the thread to update.
        :param attr: (string) Attribute to watch for change.
        """
        timeout = expected_delay * 2
        wait_time = 1
        total_time = 0
        prev_value = getattr(thread, attr)
        prev_time = time.time()

        while prev_value == getattr(thread, attr) and total_time < timeout:
            self.event.wait(wait_time)
            total_time += time.time() - prev_time
            prev_time = time.time()

    def wait_for_delay(self, delay):
        """
        Waits for the configured delay. Timesout after delay*2 seconds.
        :param delay: Amount of time to delay the thread by
        """
        total_time = 0
        prev_time = time.time()
        while total_time < delay:
            self.event.wait(delay - total_time)
            new_time = time.time()
            total_time += new_time - prev_time
            prev_time = new_time
