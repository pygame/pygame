#fake Makefile for pygame, to support the common
# ./configure;make;make install

PYTHON = python

build: Setup setup.py
	$(PYTHON) setup.py build

install: Setup setup.py
	$(PYTHON) setup.py install

Setup:
	$(PYTHON) configure.py

check tests:
	$(PYTHON) run_tests.py

test: build test src
	$(PYTHON) run_tests.py

docs:
	$(PYTHON) makeref.py

clean:
	rm -rf build dist
	rm -f lib/*~ src/*~ test/*~
	rm -f lib/*.pyc src/*.pyc test/*.pyc test/test_utils/*.pyc
	rm -rf __pycache__ lib/__pycache__ test/__pycache__/ test/test_utils/__pycache__/
	# Sphinx generated files: makeref.py. See .gitignore
	rm -rf docs/generated/
