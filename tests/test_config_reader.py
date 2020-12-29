# Copyright (c) 2020 Matt Struble. All Rights Reserved.
#
# Use is subject to license terms.
#
# Author: Matt Struble
# Date: Dec. 29 2020

import random

import pytest

from furcate.config_reader import ConfigReader

from .conftest import ConfigLoader, OnlyMultipleRunsLoader


def assert_config(expected, actual):
    for key, value in expected.items():
        assert key in actual
        assert actual[key] == value


def get_expected_permutations(config):
    expected_permutations = 1

    for key, value in config.items():
        if isinstance(value, list):
            expected_permutations *= len(value)

    return expected_permutations


@pytest.mark.parametrize(
    "test_config",
    [
        "basic_config",
        "multiple_runs_config",
        "unique_keys_config",
        "combination_config",
    ],
)
def test_read_config(test_config, request):
    config, path = request.getfixturevalue(test_config)
    cr = ConfigReader(path)

    assert_config(config, cr.data)


def test_open_broken_config(broken_config):
    config, path = broken_config

    try:
        _ = ConfigReader(path)
        print("ConfigReader should throw a ValueError")
        assert False
    except ValueError as ve:
        assert "data_name" in str(ve)


def test_excluded_configs(excluded_config):
    config, path = excluded_config

    meta_config = config.pop("meta", None)
    assert meta_config

    cr = ConfigReader(path)

    assert_config(config, cr.data)
    assert_config(meta_config, cr.meta_data)

    expected_permutations = get_expected_permutations(config)
    excluded_configs = meta_config["exclude_configs"]
    run_configs, _ = cr.gen_run_configs()

    assert expected_permutations > len(run_configs)

    for excluded in excluded_configs:
        for run in run_configs:
            all_found = True
            for key, val in excluded.items():
                assert key in run
                all_found = all_found & (run[key] == val)

            # None of the run_configs should match all of the keys in the excluded configs.
            assert all_found is False


class TestConfig(ConfigLoader):
    def test_run_configs(self):
        expected_permutations = get_expected_permutations(self.config)
        run_configs, _ = self.config_reader.gen_run_configs()

        assert expected_permutations == len(run_configs)

    def test_unique_keys(self):
        _, unique_keys = self.config_reader.gen_run_configs()

        expected_unique = []
        for key, value in self.config.items():
            if isinstance(value, list):
                expected_unique.append(key)

        assert len(expected_unique) == len(unique_keys)
        for expected in expected_unique:
            assert expected in unique_keys

    def test_no_duplicates(self):
        run_configs, _ = self.config_reader.gen_run_configs()

        for rc1 in run_configs:
            found = 0
            for rc2 in run_configs:
                matched = True
                for key in rc1:
                    matched = matched & (rc2[key] == rc1[key])
                    if not matched:
                        break

                found += int(matched)

            assert 1 == found


class TestMultipleRuns(OnlyMultipleRunsLoader):
    def assert_config_in_data(self, config, is_found=True):
        run_configs, _ = self.config_reader.gen_run_configs()

        found = False
        for rc in run_configs:
            matched = False
            for key, value in config.items():
                if rc[key] == value:
                    matched = True
                else:
                    matched = False
                    break

            if matched:
                found = True
                break

        assert is_found == found

    def assert_config_not_in_data(self, config):
        self.assert_config_in_data(config, is_found=False)

    def test_remove_completed_runs(self):
        run_configs, _ = self.config_reader.gen_run_configs()

        # Randomly select one to remove
        to_remove = random.choice(run_configs)
        num_runs = len(run_configs)

        # Assert it exists
        self.assert_config_in_data(to_remove)

        # Remove and assert that the number of run configs decreased by one
        self.config_reader.remove_completed_runs(to_remove)
        run_configs, _ = self.config_reader.gen_run_configs()

        self.assert_config_not_in_data(to_remove)
        assert num_runs - 1 == len(run_configs)

        # Duplicate removal attempts don't remove anything extra nor throw any errors
        self.config_reader.remove_completed_runs(to_remove)
        run_configs, _ = self.config_reader.gen_run_configs()

        self.assert_config_not_in_data(to_remove)
        assert num_runs - 1 == len(run_configs)
