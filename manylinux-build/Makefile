REPO_ROOT = $(abspath ..)

wheels-x64:
	docker run --rm -v ${REPO_ROOT}:/io quay.io/pypa/manylinux1_x86_64 /io/manylinux-build/build-wheels.sh

wheels-x86:
	docker run --rm -v ${REPO_ROOT}:/io quay.io/pypa/manylinux1_x86_64 /io/manylinux-build/build-wheels.sh

all: wheels-x64 wheels-x86

