init:
	pip install -r requirements.txt

clean:
    find . -name '*.pyc' -exec rm --force {} +
    find . -name '*.pyo' -exec rm --force {} +
    name '*~' -exec rm --force {}

    rm --force --recursive build/
    rm --force --recursive dist/
    rm --force --recursive *.egg-info

build:
    python setup.py install

build-tf:
    python setup.py install tf

test:
	python -m unittest discover tests

