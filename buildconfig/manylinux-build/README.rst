Manylinux wheels
================

This is for building linux binary wheels. So "pip install pygame" works on linux.

The *manylinux1* tag (see `PEP 513 <https://www.python.org/dev/peps/pep-0513/>`__)
refers to a specific set of core library minimum versions, which most recent
desktop Linux distros have.
To ensure that our libraries are ABI-compatible with these core libraries, we
build on an old Linux distribution in a docker container.

manylinux is an older linux with a fairly compatible ABI, so you can make linux binary
wheels that run on many different linux distros.

* https://github.com/pygame/pygame/issues/295
* https://github.com/pypa/auditwheel
* https://github.com/pypa/manylinux
* https://hub.docker.com/u/pygame/

The basic idea is that we build the pygame dependencies, and then bundle them up in a .whl file.
We make base images containing the dependencies (but not pygame itself), so that
we can rebuild pygame without building all the dependencies every time.

This is easiest on a Linux machine with the Docker daemon running. To get the
prebuilt base images with pygame dependencies::

    make pull-x64    # 64 bit, or
    make pull-x86    # 32 bit, or
    make pull        # Both

Then build the wheels::

    make wheels-x64  # 64 bit, or
    make wheels-x86  # 32 bit, or
    make wheels      # both

The wheels will be created in a directory called ``wheelhouse``.

If you have changed the files in ``docker_base``, e.g. to add or update
dependencies, you will need to rebuild the Docker base images::

    make base-image-x64  # 64 bit, or
    make base-image-x86  # 32 bit, or
    make base-images     # both


Virtual Machine
---------------

Below are instructions on using a vagrant virtual machine, so we can build the wheels from
mac, windows or linux boxes.


These aren't meant to be copypasta'd in. Perhaps these can be worked into a script later::

    # You should be in the base of the pygame repo when you run all this.
    $ pwd
    /home/jblogs/pygame

    # Download many megabytes of ubuntu.
    mkdir vagrant.xenial64
    cd vagrant.xenial64
    vagrant init ubuntu/xenial64

    # edit your Vagrantfile to add /vagrant_pygame synced folder.
    # You pygame folder is next to your vagrant
    config.vm.synced_folder "../pygame", "/vagrant_pygame"

    # now start vagrant.
    vagrant up
    vagrant ssh

    # now we are on the vagrant ubuntu host
    # We set up docker following these instructions for ubuntu-xenial
    # https://docs.docker.com/install/linux/docker-ce/ubuntu/#install-docker-ce-1
    sudo apt-get update
    sudo apt-get remove docker docker-engine docker.io
    sudo apt-get install apt-transport-https ca-certificates curl software-properties-common

    # Now edit /etc/hosts so it has a first line with the hostname ubuntu-xenial in it.
    # Otherwise docker does not start.
    # 127.0.0.1 localhost ubuntu-xenial
    # makes a /etc/hosts.bak in case something breaks.
    sudo sed -i".bak" '/127.0.0.1 localhost/s/$/ ubuntu-xenial/' /etc/hosts

    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

    sudo apt-get update

    sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"

    sudo apt-get install docker-ce

    # check that it runs.
    sudo docker run hello-world


    # We should have been in our python package clone root directory before we ran vagrant ssh
    cd /vagrant_pygame

    # We need to be able to run docker as the ubuntu user.
    sudo usermod -aG docker ubuntu
    sudo usermod -aG docker $USER

    # now log out of vagrant. Need to reload it because docker.
    exit

    vagrant reload
    vagrant ssh

    # now we can start docker. Should be started already.
    sudo service docker start


    cd /vagrant_pygame/buildconfig/manylinux-build

    # To make the base docker images and push them to docker hub do these commands.
    # Note, these have already been built, so only needed if rebuilding dependencies.
    # https://hub.docker.com/u/pygame/
    #make base-images
    #make push

    # We use the prebuilt docker images, which should be quicker.
    make wheels

    # List the wheels we've built
    ls -la wheelhouse

    # Testing
    export SDL_AUDIODRIVER=disk
    export SDL_VIDEODRIVER=dummy

    python3.5 -m venv anenv35
    . ./anenv35/bin/activate
    pip install wheelhouse/pygame-*cp35-cp35m-manylinux1_x86_64.whl
    python -m pygame.tests --exclude opengl,music


    # Now upload all the linux wheels to pypi.
    # Make sure your PYPI vars are set. See .travis_osx_upload_whl.py
    # Note you will need to increment the version in setup.py first.
    cd ..
    mkdir -p dist
    rm -f dist/*.whl
    cp buildconfig/manylinux-build/wheelhouse/*.whl dist/

    pip install twine

    twine upload dist/*.whl --user=pygameci


Getting a shell
---------------

To be able to run bash:

    docker run --name manylinux2010_base_x86_64 -it pygame/manylinux2010_base_x86_64
    docker run --name manylinux2010_base_i686 -it pygame/manylinux2010_base_i686

    docker run --name manylinux1_base_x86_64 -it pygame/manylinux1_base_x86_64
    docker run --name manylinux1_base_i686 -it pygame/manylinux1_base_i686


To copy the config.log file off there into a SDL2-2.0.12/config.log locally:

    docker run pygame/manylinux1_base_i686 tar -c -C /sdl_build SDL2-2.0.12/config.log | tar x



TODO
----

Maybe these need adding?

- wayland, https://wayland.freedesktop.org/building.html http://www.linuxfromscratch.org/blfs/view/svn/general/wayland-protocols.html
- vulkan, via mesa?
- xinput,
- xrandr,

