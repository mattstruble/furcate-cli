import json
import os
import tempfile


def make_tmpfile(config):
    fd, tmp_path = tempfile.mkstemp(prefix="furcate_tests")

    with os.fdopen(fd, "w") as tmp:
        json.dump(config, tmp)

    return tmp_path


def close_tmpfile(tmp_path):
    os.remove(tmp_path)
