This is for building linux binary wheels. So "pip install pygame" works on linux.

manylinux is an older linux with a fairly compatable ABI, so you can make linux binary
wheels that run on many different linux distros.

* https://bitbucket.org/pygame/pygame/issues/295/build-linux-wheels-with-manylinux
* https://github.com/pypa/auditwheel
* https://www.python.org/dev/peps/pep-0513/


The basic idea is that we build the pygame dependencies, and then bundle them up in a .whl file.
To do this we use docker containers so that the dependencies do not need to be built every time.

Below are instructions on using a vagrant virtual machine, so we can build the wheels from
mac, windows or linux boxes.


These aren't meant to be copypasta'd in. Perhaps these can be worked into a script later::

    # You should be in the base of the pygame repo when you run all this.

    # Download many megabytes of ubuntu.
    vagrant init ubuntu/xenial64
    vagrant up
    vagrant ssh

    # now we are on the vagrant ubuntu host
    # We set up docker following these instructions for ubuntu-xenial
    # https://docs.docker.com/engine/installation/linux/ubuntulinux/
    sudo apt-get install apt-transport-https ca-certificates
    sudo apt-key adv --keyserver hkp://p80.pool.sks-keyservers.net:80 --recv-keys 58118E89F3A912897C070ADBF76221572C52609D
    vi /etc/apt/sources.list.d/docker.list
    sudo vi /etc/apt/sources.list.d/docker.list
    sudo apt-get update
    sudo apt-get purge lxc-docker
    apt-cache policy docker-engine
    sudo apt-get install docker-engine

    # Now edit /etc/hosts so it has a first line with the hostname ubuntu-xenial in it.
    # Otherwise docker does not start.
    # 127.0.0.1 localhost ubuntu-xenial
    # makes a /etc/hosts.bak in case something breaks.
    sudo sed -i".bak" '/127.0.0.1 localhost/s/$/ ubuntu-xenial/' /etc/hosts

    # We should have been in our python package clone root directory before we ran vagrant ssh
    cd /vagrant

    # install auditwheel to create the wheel.
    # https://github.com/pypa/auditwheel
    #sudo apt install python3-pip
    #pip3 install auditwheel

    # We need to be able to run docker as the ubuntu user.
    sudo usermod -aG docker ubuntu
    sudo usermod -aG docker $USER

    # now log out of vagrant. Need to reload it because docker.
    exit

    vagrant reload
    vagrant ssh

    # now we can start docker. Should be started already.
    sudo service docker start


    cd /vagrant/manylinux-build

    # To make the base docker images and push them to dockerhub do
    # Note, that these have already been done, so only needed if rebuilding dependencies.
    #make base-images
    #make push

    # We use the prebuilt docker images, which should be quicker.
    make

    # Now perhaps the whl files are built correctly.
    ls -la wheelhouse

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
    cp manylinux-build/wheelhouse/*.whl dist/

    pip install twine

    python .travis_osx_upload_whl.py --no-git
