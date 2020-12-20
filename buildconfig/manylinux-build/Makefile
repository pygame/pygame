REPO_ROOT = $(abspath ../..)

wheels-manylinux-x64:
	docker run --rm -v ${REPO_ROOT}:/io pygame/manylinux1_base_x86_64 /io/buildconfig/manylinux-build/build-wheels.sh

wheels-manylinux-x86:
	docker run --rm -v ${REPO_ROOT}:/io pygame/manylinux1_base_i686 /io/buildconfig/manylinux-build/build-wheels.sh

wheels-x64:
	docker run --rm -v ${REPO_ROOT}:/io pygame/manylinux2010_base_x86_64 /io/buildconfig/manylinux-build/build-wheels.sh buildpypy

wheels-x86:
	docker run --rm -v ${REPO_ROOT}:/io pygame/manylinux2010_base_i686 /io/buildconfig/manylinux-build/build-wheels.sh buildpypy


wheels: wheels-x64 wheels-x86
wheels-manylinux: wheels-manylinux-x64 wheels-manylinux-x86



base-image-manylinux-x64:
	docker build --build-arg BASE_IMAGE=manylinux1_x86_64 --build-arg BASE_IMAGE2=manylinux1_x86_64 -t pygame/manylinux1_base_x86_64 -f docker_base/Dockerfile-x86_64 docker_base

base-image-manylinux-x86:
	docker build --build-arg BASE_IMAGE=manylinux1_i686 --build-arg BASE_IMAGE2=manylinux1_i686 -t pygame/manylinux1_base_i686 -f docker_base/Dockerfile-i686 docker_base

base-images-manylinux: base-image-manylinux-x64 base-image-manylinux-x86



base-image-x64:
	docker build -t pygame/manylinux2010_base_x86_64 -f docker_base/Dockerfile-x86_64 docker_base --build-arg BASE_IMAGE=manylinux2010_x86_64 --build-arg BASE_IMAGE2=manylinux2010_x86_64

base-image-x86:
	docker build -t pygame/manylinux2010_base_i686 -f docker_base/Dockerfile-i686 docker_base --build-arg BASE_IMAGE=manylinux2010_i686 --build-arg BASE_IMAGE2=manylinux2010_i686

base-images: base-image-x64 base-image-x86




push-manylinux-x64:
	docker push pygame/manylinux1_base_x86_64

push-manylinux-x86:
	docker push pygame/manylinux1_base_i686

push-manylinux: push-manylinux-x64 push-manylinux-x86


push-x64:
	docker push pygame/manylinux2010_base_x86_64

push-x86:
	docker push pygame/manylinux2010_base_i686

push: push-x64 push-x86




pull-manylinux-x64:
	docker pull pygame/manylinux1_base_x86_64

pull-manylinux-x86:
	docker pull pygame/manylinux1_base_i686

pull-manylinux: pull-manylinux-x64 pull-manylinux-x86

pull-x64:
	docker pull pygame/manylinux2010_base_x86_64

pull-x86:
	docker pull pygame/manylinux2010_base_i686

pull: pull-x64 pull-x86

