import os
import threading
import time
from subprocess import PIPE, CalledProcessError, Popen

import pytest

from furcate.util import MemoryTrace, get_gpu_stats


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


class TestMemoryTrace:
    def _setup(self, enabled, delay):
        """
        Create the memory trace class object and instantiate the threading event for waiting.
        :param enabled: Whether the memory trace object will be enabled (boolean)
        :param delay: MemoryTrace delay (seconds)
        """
        self.mem_trace = MemoryTrace(enabled, delay)
        self.event = threading.Event()

    def _wait_for_init(self, timeout_seconds):
        """
        Waits for MemoryTrace to start and run by checking if the start_stats have been set.
        :param timeout_seconds: Timeout to wait for thread to start
        """
        total_time = 0
        prev_time = time.time()
        while self.mem_trace._start_stats is None and total_time < timeout_seconds:
            self.event.wait(1)
            total_time += time.time() - prev_time
            prev_time = time.time()

    def _wait_for_shutdown(self, timeout_seconds):
        """
        Waits for MemoryTrace is_alive() to return false.
        :param timeout_seconds: Timeout to wait for thread to stop.
        """
        total_time = 0
        prev_time = time.time()
        while self.mem_trace.is_alive() and total_time < timeout_seconds:
            self.event.wait(1)
            total_time += time.time() - prev_time
            prev_time = time.time()

    def _wait_for_delay(self, prev_stats):
        """
        Waits for MemoryTrace to change stats based upon the configured delay. Timesout after delay*2 seconds.
        :param prev_stats: Previous stats to compare to current thread stats
        """
        timeout = self.mem_trace.delay * 2
        wait_time = self.mem_trace.delay / 3
        total_time = 0
        prev_time = time.time()
        while self.mem_trace._prev_stats == prev_stats and total_time < timeout:
            self.event.wait(wait_time)
            total_time += time.time() - prev_time
            prev_time = time.time()

    def test_disabled_init(self):
        """
        Assert that when disabled MemoryTrace doesn't run, nor allow snapshots to be performed.
        """
        self._setup(False, 100)

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

    @pytest.mark.parametrize("delay", (10, 20))
    def test_init(self, delay):
        """
        Asserts that an enabled MemoryTrace properly initializes and waits for the delayed time before taking a snapshot.
        :param delay: Delay in seconds for the MemoryTrace thread to wait between snapshots.
        """
        self._setup(True, delay)

        assert self.mem_trace.enabled is True
        assert self.mem_trace._running is True

        self._wait_for_init(5)

        assert self.mem_trace._start_stats is not None
        assert self.mem_trace._prev_stats == self.mem_trace._start_stats
        assert self.mem_trace.is_alive() is True

        self._wait_for_delay(self.mem_trace._start_stats)

        assert self.mem_trace._prev_stats != self.mem_trace._start_stats

        self._teardown()

    @pytest.mark.parametrize("delay", (3, 6, 9))
    def test_multiple_snapshots(self, delay):
        """
        Asserts that MemoryTrace performs multiple snapshots with different delays.
        :param delay: Delay time between MemoryTrace snapshots.
        :return:
        """
        self._setup(True, delay)
        self._wait_for_init(5)

        prev_stats = self.mem_trace._start_stats

        for _ in range(3):
            start_time = time.time()
            self._wait_for_delay(prev_stats)
            time_taken = time.time() - start_time

            assert time_taken >= self.mem_trace.delay
            assert prev_stats != self.mem_trace._prev_stats
            prev_stats = self.mem_trace._prev_stats

        self._teardown()

    def test_snapshot(self):
        """
        Asserts that external threads can initiate a snapshot.
        """
        self._setup(True, 100)

        assert self.mem_trace._prev_stats == self.mem_trace._start_stats

        self._wait_for_init(5)

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
        self._wait_for_shutdown(5)
        assert self.mem_trace.is_alive() is False
