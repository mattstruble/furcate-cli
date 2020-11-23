# Copyright (c) 2020 Matt Struble. All Rights Reserved.
#
# Use is subject to license terms.
#
# Author: Matt Struble
# Date: Nov. 18 2020
import os
import tempfile
import subprocess
import threading
import time
from datetime import datetime
import json
import logging
import pandas as pd

from .gpu_helper import get_gpus
from .config_reader import ConfigReader

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
    log_dir = os.path.dirname(config.data['log_dir'])
    fname = os.path.join(log_dir, 'run_data.csv')

    run_data = config.meta_data.pop('data', {})

    # Package metadata up to the data layer for writing to csv
    for key, value in run_data.items():
        config.data['run_'+key] = value

    config.data['meta'] = str(config.data['meta'])

    with csv_lock:
        pd.DataFrame(config.data, index=[0]).to_csv(fname, header=not os.path.exists(fname), mode='a', encoding='utf-8', index=False)


class TrainingThread (threading.Thread):

    def __init__(self, id, config, script_name, log_keys):
        threading.Thread.__init__(self)
        self.threadID = id
        self.config = config
        self.script_name = script_name
        self.log_keys = log_keys

        self.dir_name = os.path.basename(self.config['data_dir'])
        self.name = self.dir_name + str(id)

    def _gen_log_dir(self):
        folder = "{}_{}".format(self.name, self.dir_name)

        for key in self.config.keys():
            if key in self.log_keys:
                short_key = ''.join([s[0] for s in key.split('_')])
                value = str(self.config[key]).replace('.','-')
                folder += "_{}{}".format(short_key, value)

        self.config['log_dir'] = os.path.join(self.config['log_dir'], folder)

        if not os.path.exists(self.config['log_dir']):
            os.makedirs(self.config['log_dir'])

    def _generate_run_command(self, config_path):
        command = 'python3 {} --config "{}" --name "{}" --id "{}"'.format(
            self.script_name, config_path, self.name, self.threadID)

        if self.config['gpu']:
            command += ' --gpu "{}"'.format(self.config['gpu'])

        return command

    def run(self):

        fd, temppath = tempfile.mkstemp()
        start_time = datetime.now()

        try:
            self._gen_log_dir()

            with os.fdopen(fd, 'w') as tmp:
                json.dump(self.config, tmp)

            command = self._generate_run_command(temppath)

            logger.debug('Starting:', os.getcwd(), command)

            with open(os.path.join(self.config['log_dir'], self.name + '.log'), 'w', encoding='utf-8') as log, \
                open(os.path.join(self.config['log_dir'], self.name + '.err'), 'w', encoding='utf-8') as err:
                subprocess.call(command, shell=True, stdout=log, stderr=err)

            if os.path.exists(os.path.join(self.config['log_dir'], 'run_data.json')):
                run_config = ConfigReader(os.path.join(self.config['log_dir'], 'run_data.json'))
                config_to_csv(run_config)

        finally:
            os.remove(temppath)

        self.run_time = datetime.now() - start_time


class Runner(object):

    def __init__(self, config):
        self.config = config
        self.meta = config.meta_data

        self.run_configs, self.log_keys = self.config.gen_run_configs()

    def run(self, script_name):
        gpus = get_gpus(self.meta['framework'])

        if len(gpus) < 1 and self.meta['allow_cpu'] is False:
            raise ValueError(
                "CPU processing is not enabled and could not find GPU devices to run on. If you want to enable CPU processing please update the config: { 'meta': { 'allow_cpu': true } }")

        max_threads = self._get_max_threads(gpus)

        main_thread = threading.current_thread()
        thread_id = 0
        gpu_mapping = {}

        if max_threads > 1:
            gpu_idxs = list(range(len(gpus)))
        else:
            gpu_idxs = [None]

        run_times = []
        avg_seconds = 0
        sleep_seconds = 60
        while len(self.run_configs) > 0 or len(gpu_mapping) > 0:
            while threading.activeCount() -1 == max_threads or (len(gpu_mapping) > 0 and threading.activeCount() -1 == len(gpu_mapping)):
                if 0 < avg_seconds < sleep_seconds:
                    sleep_seconds = max(1, min(sleep_seconds, int(avg_seconds)))

                time.sleep(sleep_seconds)

            to_del = []
            for t, gpu in gpu_mapping.items():
                if not t.isAlive():
                    gpu_idxs.append(gpu)
                    to_del.append(t)

                    run_times.append(t.run_time.total_seconds())
                    avg_seconds = (sum(run_times) / len(run_times)) / max_threads
                    thread_time = seconds_to_string(run_times[-1])
                    remaining_time = seconds_to_string(avg_seconds*(len(self.run_configs)+len(gpu_mapping)-1)+(sleep_seconds*len(self.run_configs)))
                    logger.info("Thread %d finished - %s - est. total time remaining: %s",
                                t.threadID, thread_time , remaining_time)

            for t in to_del:
                del gpu_mapping[t]

            gpu = gpu_idxs.pop()

            if len(self.run_configs) > 0:
                config = self.run_configs.pop()
                config['gpu'] = gpu

                training = TrainingThread(thread_id, config, script_name, self.log_keys)
                training.start()

                gpu_mapping[training] = gpu
                thread_id += 1

        for t in threading.enumerate():
            if t is not main_thread:
                t.join()


    def _get_max_threads(self, gpus):
        if self.meta and 'max_threads' in self.meta:
            max_threads = min(1, self.meta['max_threads'])

            if max_threads > len(gpus) > 1:
                logger.warning(
                    "Configured max_threads [{}] is higher than total number of GPUs [{}]. Defaulting to number of GPUs".format(
                        max_threads, len(gpus)))
                max_threads = min(len(gpus), max_threads)
        else:
            max_threads = max(1, len(gpus))
            logger.warning("Couldn't find max_threads in config, defaulting to number of GPUs [{}].".format(max_threads))

        if len(gpus) > max_threads:
            logger.warning(
                "Potentially not utilizing all the GPUs. Check the config to ensure the meta tag 'max_threads' is set properly: { 'meta': { 'max_threads': X } }")

        return max_threads

