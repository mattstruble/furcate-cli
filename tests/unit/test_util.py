import os
import threading
import time
from subprocess import PIPE, CalledProcessError, Popen

import pytest

from furcate.util import MemoryTrace, get_gpu_stats

from .conftest import ThreadHelper


def get_nvidia_gpus():
    try:
        stdout, _ = Popen(
            ["ls -la /dev | grep nvidia[0-9]+"],
            stdout=PIPE,
        ).communicate()
    except (CalledProcessError, FileNotFoundError):
        return []

    devices = stdout.decode("UTF-8").split(os.linesep)

    ids = [int(s) for s in devices if s.isdigit()]

    return ids


def test_get_gpu_stats():
    """
    Assert that the gpu stats returns the expected number of GPUS. Best we can do with limited build env is to test if
    matches the return of a /dev grep.
    """
    expected = get_nvidia_gpus()
    actual = get_gpu_stats()

    assert len(expected) == len(actual)

    for gpu in actual:
        assert gpu.id in expected


def test_get_gpu_stats_no_dups():
    """
    Assert that the gpu stats doesn't return duplicates.
    """
    gpus = get_gpu_stats()
    ids = set([gpu.id for gpu in gpus])

    assert len(gpus) == len(ids)


def test_init_memory_trace_from_config_reader(basic_config_reader):
    _, config_reader = basic_config_reader
    mem_trace_config = config_reader.meta_data["mem_trace"]
    mem_trace = MemoryTrace(**mem_trace_config)

    assert mem_trace_config["enabled"] == mem_trace.enabled
    assert mem_trace_config["delay"] == mem_trace.delay
    assert mem_trace_config["top"] == mem_trace.top
    assert mem_trace_config["trace"] == mem_trace.trace


def test_init_memory_trace_from_dict():
    mem_trace_config = {"enabled": True, "delay": 20, "top": 5, "trace": 10}

    mem_trace = MemoryTrace(**mem_trace_config)

    assert mem_trace_config["enabled"] == mem_trace.enabled
    assert mem_trace_config["delay"] == mem_trace.delay
    assert mem_trace_config["top"] == mem_trace.top
    assert mem_trace_config["trace"] == mem_trace.trace


class TestMemoryTrace(ThreadHelper):
    def _setup(self, enabled, delay):
        """
        Create the memory trace class object and instantiate the threading event for waiting.
        :param enabled: Whether the memory trace object will be enabled (boolean)
        :param delay: MemoryTrace delay (seconds)
        """
        self.mem_trace = MemoryTrace(enabled, delay)
        self.event = threading.Event()

    def _wait_for_init(self, timeout_seconds=5):
        self.wait_for_init(self.mem_trace, timeout_seconds=timeout_seconds)

    def _wait_for_thread_update(self):
        self.wait_for_thread_update(
            self.mem_trace, expected_delay=self.mem_trace.delay, attr="_prev_stats"
        )

    def _wait_for_shutdown(self):
        self.wait_for_shutdown(self.mem_trace)

    def test_disabled_init(self):
        """
        Assert that when disabled MemoryTrace doesn't run, nor allow snapshots to be performed.
        """
        self._setup(False, 1)
        self._wait_for_init(2)

        # Assert nothing is initialized or changed when disabled
        assert self.mem_trace.enabled is False
        assert self.mem_trace._running is False
        assert self.mem_trace._start_stats is None
        assert self.mem_trace._prev_stats is None
        assert self.mem_trace.is_alive() is False

        # Assert snapshotting doesn't do anything when disabled
        self.mem_trace.snapshot()
        assert self.mem_trace._start_stats is None
        assert self.mem_trace._prev_stats is None

        self._teardown()

    @pytest.mark.parametrize("delay", (5, 10))
    def test_init(self, delay):
        """
        Asserts that an enabled MemoryTrace properly initializes and waits for the delayed time before taking a snapshot.
        :param delay: Delay in seconds for the MemoryTrace thread to wait between snapshots.
        """
        self._setup(True, delay)

        assert self.mem_trace.enabled is True
        self._wait_for_init()

        assert self.mem_trace._running is True

        assert self.mem_trace._start_stats is not None
        assert self.mem_trace._prev_stats == self.mem_trace._start_stats
        assert self.mem_trace.is_alive() is True

        self._wait_for_thread_update()

        assert self.mem_trace._prev_stats != self.mem_trace._start_stats

        self._teardown()

    @pytest.mark.parametrize("delay", (3, 6))
    def test_multiple_snapshots(self, delay):
        """
        Asserts that MemoryTrace performs multiple snapshots with different delays.
        :param delay: Delay time between MemoryTrace snapshots.
        :return:
        """
        self._setup(True, delay)
        self._wait_for_init()

        prev_stats = self.mem_trace._start_stats

        for _ in range(3):
            start_time = time.time()
            self._wait_for_thread_update()
            time_taken = time.time() - start_time

            assert (
                time_taken >= self.mem_trace.delay
                or abs(time_taken - self.mem_trace.delay) < 1
            )
            assert prev_stats != self.mem_trace._prev_stats
            prev_stats = self.mem_trace._prev_stats

        self._teardown()

    def test_snapshot(self):
        """
        Asserts that external threads can initiate a snapshot.
        """
        self._setup(True, 10)

        assert self.mem_trace._prev_stats == self.mem_trace._start_stats

        self._wait_for_init()

        self.mem_trace.snapshot("test")

        assert self.mem_trace._prev_stats != self.mem_trace._start_stats
        prev_stats = self.mem_trace._prev_stats

        self.mem_trace.snapshot("test2")
        assert self.mem_trace._prev_stats != prev_stats

        self._teardown()

    def _teardown(self):
        """
        Stops the MemoryTrace thread and asserts that it shutsdown properly.
        """
        self.mem_trace.stop()
        assert self.mem_trace._running is False
        self._wait_for_shutdown()
        assert self.mem_trace.is_alive() is False
