import json
import os
import shutil
import tempfile


def make_tmpfile(config):
    fd, tmp_path = tempfile.mkstemp(prefix="furcate_tests")

    with os.fdopen(fd, "w") as tmp:
        json.dump(config, tmp)

    return tmp_path


def close_tmpfile(tmp_path):
    if os.path.isdir(tmp_path):
        shutil.rmtree(tmp_path)
    else:
        os.remove(tmp_path)


def make_tmpdir():
    tmp_path = tempfile.mkdtemp(prefix="furcate_test_dir")

    return tmp_path
