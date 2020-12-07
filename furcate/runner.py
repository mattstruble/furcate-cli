# Copyright (c) 2020 Matt Struble. All Rights Reserved.
#
# Use is subject to license terms.
#
# Author: Matt Struble
# Date: Nov. 18 2020
import gc
import json
import logging
import os
import re
import subprocess
import tempfile
import threading
import time
import tracemalloc
from datetime import datetime

import pandas as pd

from .config_reader import ConfigReader
from .gpu_helper import get_gpu_stats

logger = logging.getLogger(__name__)


def seconds_to_string(seconds):
    day = int(seconds // (24 * 3600))
    time_mod = seconds % (24 * 3600)
    hour = int(time_mod // 3600)
    time_mod %= 3600
    minute = int(time_mod // 60)
    seconds = int(time_mod % 60)

    if day > 0:
        res = "{}d {}h {}m {}s".format(day, hour, minute, seconds)
    elif hour > 0:
        res = "{}h {}m {}s".format(hour, minute, seconds)
    elif minute > 0:
        res = "{}m {}s".format(minute, seconds)
    else:
        res = "{}s".format(seconds)

    return res


csv_lock = threading.Lock()


def config_to_csv(config):
    log_dir = os.path.dirname(config.data["log_dir"])
    fname = os.path.join(log_dir, "run_data.csv")

    run_data = config.meta_data.pop("data", {})

    # Package metadata up to the data layer for writing to csv
    for key, value in run_data.items():
        config.data["run_" + key] = value

    config.data["meta"] = str(config.data["meta"])

    with csv_lock:
        if os.path.exists(fname):
            csv_df = pd.read_csv(fname)
            csv_df = csv_df.append(config.data, ignore_index=True)
        else:
            csv_df = pd.DataFrame.from_dict(config.data)

        csv_df.to_csv(fname, header=True, mode="w", encoding="utf-8", index=False)


class MemoryTrace(threading.Thread):
    # https://tech.buzzfeed.com/finding-and-fixing-memory-leaks-in-python-413ce4266e7d

    def __init__(self, enabled, delay=300, top=10, trace=1):
        """
        Log memory usage on a delay.

        :param delay: in seconds (int)
        :param top: number of top allocations to list (int)
        :param trace: number of top allocations to trace (int)
        :return: None
        """
        super().__init__()

        self.enabled = enabled
        self.delay = delay
        self.top = top
        self.trace = trace

        self.setDaemon(True)
        self._event = threading.Event()

        self._running = False
        self._start_stats = None
        self._prev_stats = None
        self._snapshot_lock = threading.Lock()

        self.start()

    def run(self):
        if self.enabled:
            self._running = True

            logger.debug("Starting MemoryTrace")
            tracemalloc.start(25)
            self._start_stats = tracemalloc.take_snapshot()
            self._prev_stats = self._start_stats

            while self._running:
                self._event.wait(self.delay)
                self.snapshot()

    def snapshot(self, title="Snapshot"):
        if self.enabled:
            with self._snapshot_lock:
                current = tracemalloc.take_snapshot()

                if title:
                    logger.debug("------ %s ------", title)

                # compare current snapshot to starting snapshot
                stats = current.compare_to(self._start_stats, "filename")

                # compare current snapshot to previous snapshot
                prev_stats = current.compare_to(self._prev_stats, "lineno")

                logger.debug("GPU Stats")
                for gpu in get_gpu_stats():
                    logger.debug(
                        "gpu_stats id={}, name={}, mem_used={}, mem_total={}, mem_util={} %, volatile_gpu={}, temp={} C".format(
                            gpu.id,
                            gpu.name,
                            gpu.memory_used,
                            gpu.memory_total,
                            int(gpu.memory_util * 100),
                            gpu.util,
                            gpu.temperature,
                        )
                    )

                logger.debug("Top Diffs since Start")
                for i, stat in enumerate(stats[: self.top], 1):
                    logger.debug("top_diffs i=%d, stat=%s", i, str(stat))

                logger.debug("Top Incremental")
                for i, stat in enumerate(prev_stats[: self.top], 1):
                    logger.debug("top_incremental i=%d, stat=%s", i, str(stat))

                logger.debug("Top Current")
                for i, stat in enumerate(current.statistics("filename")[: self.top], 1):
                    logger.debug("top_current i=%d, stat=%s", i, str(stat))

                traces = current.statistics("traceback")
                for stat in traces[: self.trace]:
                    logger.debug(
                        "traceback memory_blocks=%d, size_kB=%d",
                        stat.count,
                        stat.size / 1024,
                    )
                    for line in stat.traceback.format():
                        logger.debug(line)

                self._prev_stats = current

    def stop(self):
        logger.debug("Stopping MemoryTrace")
        self._running = False
        self._event.set()


class ConfigUpdater(threading.Thread):
    def __init__(self, config, refresh_rate=60):
        threading.Thread.__init__(self)
        self.setDaemon(True)

        self.config_path = config.filename
        self.config = config
        self.refresh_rate = refresh_rate
        self.flagged = False

        self._remove_completed_runs()

        self._mtime = os.path.getmtime(self.config_path)
        self._running = True
        self._config_lock = threading.Lock()
        self._event = threading.Event()

    def run(self):
        while self._running:
            self._event.wait(self.refresh_rate)

            if self._mtime != os.path.getmtime(self.config_path):
                logger.info(
                    "Detected change in %s, reloading configurations.", self.config_path
                )
                self._mtime = os.path.getmtime(self.config_path)
                self.flagged = True

                with self._config_lock:
                    self.config = ConfigReader(self.config_path)
                    self._remove_completed_runs()

    def _remove_completed_runs(self):
        run_configs, _ = self.config.gen_run_configs()
        log_dir = run_configs[0]["log_dir"]

        if os.path.exists(os.path.join(log_dir, "run_data.csv")):
            df = pd.read_csv(os.path.join(log_dir, "run_data.csv"))
            logger.info(
                "Detected previous runs, removing %d configuration(s).", len(df)
            )
            for _, row in df.iterrows():
                run_dict = row.to_dict()
                self.config.remove_completed_runs(run_dict)

    def reset_flagged(self):
        self.flagged = False

    def get_config(self):
        with self._config_lock:
            config = self.config

        return config

    def stop(self):
        logger.debug("Stopping ConfigUpdater")
        self._running = False
        self._event.set()


class TrainingThread(threading.Thread):
    def __init__(self, id, config, script_name, log_keys):
        threading.Thread.__init__(self)
        self.setDaemon(True)

        self.threadID = id
        self.config = config
        self.script_name = script_name
        self.log_keys = log_keys

        self.dir_name = os.path.basename(self.config["data_dir"])
        self.name = self.config["data_name"] + str(id)

    def _gen_log_dir(self):
        folder = "{}_{}".format(self.name, self.dir_name)

        for key in self.config.keys():
            if key in self.log_keys and key not in ["data_dir"]:
                short_key = "".join([s[0] for s in key.split("_")])
                value = str(self.config[key]).replace(".", "-")
                folder += "_{}{}".format(short_key, value)

        self.config["log_dir"] = os.path.join(self.config["log_dir"], folder)

        if not os.path.exists(self.config["log_dir"]):
            os.makedirs(self.config["log_dir"])

    def _generate_run_command(self, config_path):
        command = 'python3 {} --config "{}" --name "{}" --id "{}"'.format(
            self.script_name, config_path, self.name, self.threadID
        )

        if self.config["gpu"] is not None:
            command += ' --gpu "{}"'.format(self.config["gpu"])

        return command

    def run(self):

        fd, temppath = tempfile.mkstemp()
        start_time = datetime.now()

        try:
            self._gen_log_dir()

            with os.fdopen(fd, "w") as tmp:
                json.dump(self.config, tmp)

            command = self._generate_run_command(temppath)

            logger.debug("Starting: %s", command)

            with open(
                os.path.join(self.config["log_dir"], self.name + ".log"),
                "w",
                encoding="utf-8",
            ) as log, open(
                os.path.join(self.config["log_dir"], self.name + ".err"),
                "w",
                encoding="utf-8",
            ) as err:
                subprocess.call(command, shell=True, stdout=log, stderr=err)

            if os.path.exists(os.path.join(self.config["log_dir"], "run_data.json")):
                run_config = ConfigReader(
                    os.path.join(self.config["log_dir"], "run_data.json")
                )
                config_to_csv(run_config)

        finally:
            os.remove(temppath)

        self.run_time = datetime.now() - start_time


class Runner(object):
    def __init__(self, config):
        self.config_updater = ConfigUpdater(config)
        self.config = self.config_updater.get_config()
        self.meta = config.meta_data

        self.run_configs, self.log_keys = self.config.gen_run_configs()

    def run(self, script_name, gpu_indices):
        mem_trace = MemoryTrace(self.meta["mem_trace"])

        if len(gpu_indices) < 1 and self.meta["allow_cpu"] is False:
            raise ValueError(
                "CPU processing is not enabled and could not find GPU devices to run on. If you want to enable CPU processing please update the config: { 'meta': { 'allow_cpu': true } }"
            )

        max_threads = self._get_max_threads(gpu_indices)

        logger.info(
            "Generated %d unique combinations from configuration keys: %s",
            len(self.run_configs),
            str(self.log_keys),
        )

        main_thread = threading.current_thread()
        thread_id = 0
        gpu_mapping = {}

        if max_threads == 1:
            gpu_indices = [-1]

        run_times = []
        avg_seconds = 0
        sleep_seconds = 60

        self.config_updater.start()

        while len(self.run_configs) > 0 or len(gpu_mapping) > 0:
            while (
                len(gpu_mapping) == max_threads
                and all(t.isAlive() for t, _ in gpu_mapping.items())
            ) or (
                len(gpu_mapping) > 0
                and len(self.run_configs) == 0
                and all(t.isAlive() for t, _ in gpu_mapping.items())
            ):
                if 0 < avg_seconds < sleep_seconds:
                    sleep_seconds = max(1, min(sleep_seconds, int(avg_seconds)))

                time.sleep(sleep_seconds)

                if self.config_updater.flagged:
                    self.config = self.config_updater.get_config()
                    logger.debug("Config Updater Flagged")
                    for t, (gpu, config) in gpu_mapping.items():
                        logger.debug("Removing run: [%s]", str(config))
                        self.config.remove_completed_runs(config)
                        self.run_configs, _ = self.config.gen_run_configs()

                    self.config_updater.reset_flagged()

            to_del = []
            for t, (gpu, config) in gpu_mapping.items():
                if not t.isAlive():
                    gpu_indices.append(gpu)
                    to_del.append(t)

                    run_times.append(t.run_time.total_seconds())
                    avg_seconds = (sum(run_times) / len(run_times)) / max_threads
                    thread_time = seconds_to_string(run_times[-1])
                    remaining_time = seconds_to_string(
                        avg_seconds * (len(self.run_configs) + len(gpu_mapping) - 1)
                        + (sleep_seconds * len(self.run_configs))
                    )
                    logger.info(
                        "Thread %d finished - %s - %d configs remaining - est. total time remaining: %s",
                        t.threadID,
                        thread_time,
                        len(self.run_configs),
                        remaining_time,
                    )

            if len(to_del) > 0:
                mem_trace.snapshot("Prior to deleting threads [{}]".format(str(to_del)))
                for t in to_del:
                    del gpu_mapping[t]

                to_del.clear()
                gc.collect()
                mem_trace.snapshot("After deleting threads")

            gpu = gpu_indices.pop()

            if len(self.run_configs) > 0:
                config = self.run_configs.pop()
                config["gpu"] = gpu

                training = TrainingThread(thread_id, config, script_name, self.log_keys)
                training.start()

                gpu_mapping[training] = (gpu, config)
                thread_id += 1

                mem_trace.snapshot(
                    "Started thread {}: [{}]".format(thread_id, str(config))
                )

        self.config_updater.stop()
        mem_trace.stop()
        for t in threading.enumerate():
            if t is not main_thread:
                t.join()

    def _get_max_threads(self, gpus):
        if self.meta and "max_threads" in self.meta:
            max_threads = min(1, self.meta["max_threads"])

            if max_threads > len(gpus) > 1:
                logger.warning(
                    "Configured max_threads [{}] is higher than total number of GPUs [{}]. Defaulting to number of GPUs".format(
                        max_threads, len(gpus)
                    )
                )
                max_threads = min(len(gpus), max_threads)
        else:
            max_threads = max(1, len(gpus))
            logger.warning(
                "Couldn't find max_threads in config, defaulting to number of GPUs [{}].".format(
                    max_threads
                )
            )

        if len(gpus) > max_threads:
            logger.warning(
                "Potentially not utilizing all the GPUs. Check the config to ensure the meta tag 'max_threads' is set properly: { 'meta': { 'max_threads': X } }"
            )

        return max_threads
