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

docs:	install
	cd docs/utils
	$(PYTHON) makedocs.py

clean:
	rm -rf build dist
	rm -f lib/*~ src/*~ test/*~

