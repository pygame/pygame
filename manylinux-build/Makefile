REPO_ROOT = $(abspath ..)

wheels-x64:
	docker run --rm -v ${REPO_ROOT}:/io pygame/manylinux1_base_x86_64 /io/manylinux-build/build-wheels.sh

wheels-x86:
	docker run --rm -v ${REPO_ROOT}:/io pygame/manylinux1_base_i686 /io/manylinux-build/build-wheels.sh

wheels: wheels-x64 wheels-x86

base-image-x64:
	docker build -t pygame/manylinux1_base_x86_64 -f docker_base/Dockerfile-x86_64 docker_base

base-image-x86:
	docker build -t pygame/manylinux1_base_i686 -f docker_base/Dockerfile-i686 docker_base

base-images: base-image-x64 base-image-x86

push-x64:
	docker push pygame/manylinux1_base_x86_64

push-x86:
	docker push pygame/manylinux1_base_i686

push: push-x64 push-x86

pull-x64:
	docker pull pygame/manylinux1_base_x86_64

pull-x86:
	docker pull pygame/manylinux1_base_i686

pull: pull-x64 pull-x86
