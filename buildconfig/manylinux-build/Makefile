REPO_ROOT = $(abspath ../..)

wheels-manylinux1-x64:
	docker run --rm -v ${REPO_ROOT}:/io pygame/manylinux1_base_x86_64 /io/buildconfig/manylinux-build/build-wheels.sh

wheels-manylinux1-x86:
	docker run --rm -v ${REPO_ROOT}:/io pygame/manylinux1_base_i686 /io/buildconfig/manylinux-build/build-wheels.sh

wheels-manylinux2010-x64:
	docker run --rm -v ${REPO_ROOT}:/io pygame/manylinux2010_base_x86_64 /io/buildconfig/manylinux-build/build-wheels.sh buildpypy

wheels-manylinux2010-x86:
	docker run --rm -v ${REPO_ROOT}:/io pygame/manylinux2010_base_i686 /io/buildconfig/manylinux-build/build-wheels.sh buildpypy

wheels-manylinux2014-x64:
	docker run --rm -v ${REPO_ROOT}:/io pygame/manylinux2014_base_x86_64 /io/buildconfig/manylinux-build/build-wheels.sh buildpypy

wheels-manylinux2014-x86:
	docker run --rm -v ${REPO_ROOT}:/io pygame/manylinux2014_base_i686 /io/buildconfig/manylinux-build/build-wheels.sh buildpypy

wheels-manylinux2014-aarch64:
	docker run --rm -v ${REPO_ROOT}:/io pygame/manylinux2014_base_aarch64 /io/buildconfig/manylinux-build/build-wheels.sh

wheels: wheels-manylinux1-x64 wheels-manylinux1-x86 wheels-manylinux2010-x64 wheels-manylinux2010-x86 wheels-manylinux2014-x64 wheels-manylinux2014-x86 wheels-manylinux2014-aarch64



base-image-manylinux1-x64:
	docker build --build-arg BASE_IMAGE=manylinux1_x86_64 --build-arg BASE_IMAGE2=manylinux1_x86_64 -t pygame/manylinux1_base_x86_64 -f docker_base/Dockerfile-x86_64 docker_base

base-image-manylinux1-x86:
	docker build --build-arg BASE_IMAGE=manylinux1_i686 --build-arg BASE_IMAGE2=manylinux1_i686 -t pygame/manylinux1_base_i686 -f docker_base/Dockerfile-i686 docker_base

base-image-manylinux2010-x64:
	docker build -t pygame/manylinux2010_base_x86_64 -f docker_base/Dockerfile-x86_64 docker_base --build-arg BASE_IMAGE=manylinux2010_x86_64 --build-arg BASE_IMAGE2=manylinux2010_x86_64 --progress=plain

base-image-manylinux2010-x86:
	docker build -t pygame/manylinux2010_base_i686 -f docker_base/Dockerfile-i686 docker_base --build-arg BASE_IMAGE=manylinux2010_i686 --build-arg BASE_IMAGE2=manylinux2010_i686

base-image-manylinux2014-x64:
	docker build -t pygame/manylinux2014_base_x86_64 -f docker_base/Dockerfile-x86_64 docker_base --build-arg BASE_IMAGE=manylinux2014_x86_64 --build-arg BASE_IMAGE2=manylinux2014_x86_64 --progress=plain

base-image-manylinux2014-x86:
	docker build -t pygame/manylinux2014_base_i686 -f docker_base/Dockerfile-i686 docker_base --build-arg BASE_IMAGE=manylinux2014_i686 --build-arg BASE_IMAGE2=manylinux2014_i686

base-image-manylinux2014-aarch64:
	docker build -t pygame/manylinux2014_base_aarch64 -f docker_base/Dockerfile-aarch64 docker_base --build-arg BASE_IMAGE=manylinux2014_aarch64 --build-arg BASE_IMAGE2=manylinux2014_aarch64

# base-images: base-image-manylinux1-x64 base-image-manylinux1-x86 base-image-manylinux2010-x64 base-image-manylinux2010-x86 base-image-manylinux2014-x64 base-image-manylinux2014-x86 base-image-aarch64
base-images: base-image-manylinux2010-x64 base-image-manylinux2010-x86 base-image-manylinux2014-x64 base-image-manylinux2014-x86 base-image-manylinux2014-aarch64


# push-manylinux1-x64:
# 	docker push pygame/manylinux1_base_x86_64

# push-manylinux1-x86:
# 	docker push pygame/manylinux1_base_i686

push-manylinux2010-x64:
	docker push pygame/manylinux2010_base_x86_64

push-manylinux2010-x86:
	docker push pygame/manylinux2010_base_i686

push-manylinux2014-x64:
	docker push pygame/manylinux2014_base_x86_64

push-manylinux2014-x86:
	docker push pygame/manylinux2014_base_i686

push-manylinux2014-aarch64:
	docker push pygame/manylinux2014_base_aarch64
	
# push: push-manylinux1-x64 push-manylinux1-x86 push-manylinux2010-x64 push-manylinux2010-x86 push-manylinux2014-x64 push-manylinux2014-x86
push: push-manylinux2010-x64 push-manylinux2010-x86 push-manylinux2014-x64 push-manylinux2014-x86 push-manylinux2014-aarch64



pull-manylinux1-x64:
	docker pull pygame/manylinux1_base_x86_64

pull-manylinux1-x86:
	docker pull pygame/manylinux1_base_i686

pull-manylinux2010-x64:
	docker pull pygame/manylinux1_base_x86_64

pull-manylinux2010-x86:
	docker pull pygame/manylinux2010_base_i686

pull-manylinux2014-x64:
	docker pull pygame/manylinux1_base_x86_64

pull-manylinux2014-x86:
	docker pull pygame/manylinux2010_base_i686

pull-manylinux2014-aarch64:
	docker pull pygame/manylinux2014_base_aarch64

pull: pull-manylinux1-x64 pull-manylinux1-x86 pull-manylinux2010-x64 pull-manylinux2010-x86 pull-manylinux2014-x64 pull-manylinux2014-x86 pull-manylinux2014-aarch64
# pull: pull-manylinux2010-x64 pull-manylinux2010-x86 pull-manylinux2014-x64 pull-manylinux2014-x86 pull-aarch64
