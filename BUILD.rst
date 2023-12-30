.. image:: https://raw.githubusercontent.com/pygame/pygame/main/docs/reST/_static/pygame_logo.svg
  :alt: pygame
  :target: https://www.pygame.org/


|AppVeyorBuild| |PyPiVersion| |PyPiLicense|
|Python3| |GithubCommits| |BlackFormatBadge|

.. |AppVeyorBuild| image:: https://ci.appveyor.com/api/projects/status/x4074ybuobsh4myx?svg=true
   :target: https://ci.appveyor.com/project/pygame/pygame

.. |PyPiVersion| image:: https://img.shields.io/pypi/v/pygame.svg?v=1
   :target: https://pypi.python.org/pypi/pygame

.. |PyPiLicense| image:: https://img.shields.io/pypi/l/pygame.svg?v=1
   :target: https://pypi.python.org/pypi/pygame

.. |Python3| image:: https://img.shields.io/badge/python-3-blue.svg?v=1

.. |GithubCommits| image:: https://img.shields.io/github/commits-since/pygame/pygame/2.1.2.svg
   :target: https://github.com/pygame/pygame/compare/2.1.2...main

.. |BlackFormatBadge| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

====================
Building From Source
====================

If you want to use features that are currently in development,
or you want to contribute to pygame, you will need to build pygame
locally from its source code, rather than pip installing it.

Installing from source is fairly automated. The most work will
involve compiling and installing all the pygame dependencies.  Once
that is done, run the ``setup.py`` script which will attempt to
auto-configure, build, and install pygame.

========================
Compiling pygame on PyPy
========================

Compiling pygame on PyPy is pretty straight forward, just follow the
regular compilation instructions for your platform, replacing any
``python`` / ``python3`` commands with ``pypy`` / ``pypy3`` and
``pip`` / ``pip3`` commands with ``pypy -m pip`` / ``pypy3 -m pip``
respectively.

PyPy has to be installed for this to work, and it can be downloaded
from `here`_ or any package manager you prefer.

.. _here: https://www.pypy.org/download.html

==================
OS-specific guides
==================

* `Microsoft Windows`_

* `MacOS`_

* `Ubuntu`_

* `Fedora`_

-----------------
Microsoft Windows
-----------------

There are four steps to compile pygame on Windows.

1. Get a C/C++ compiler.
2. Install Python 3.6+
3. Checkout pygame from github.
4. Run the pygame install commands.

Get a C/C++ compiler
====================

* Install `Microsoft Build Tools for Visual Studio 2017`_.

  If the link doesn't work, here are the links to download the Visual
  Studio Build Tools from Microsoft directly.

  `2017`_  `2019`_  `2022`_

  The setuptools Python package version must be at least 34.4.0.

  ``py -m pip install setuptools -U``

* Make sure this checkbox is ticked: "[ ] MSVC v140 - VS 2015 C++
  build tools (v14.00)", even if it asks you to install a different
  version.

These will download the required dependencies and build for SDL2.

.. _Microsoft Build Tools for Visual Studio 2017:
   https://www.visualstudio.com/downloads/#build-tools-for-visual-studio-2017
.. _2017: https://aka.ms/vs/15/release/vs_buildtools.exe
.. _2019: https://aka.ms/vs/16/release/vs_buildtools.exe
.. _2022: https://aka.ms/vs/17/release/vs_buildtools.exe

Checkout pygame from github
===========================

To get pygame from github, you might need to install git. `Git for Windows`_ is a
good command line option for git checkouts on
windows.

Here is the `pygame github repo`_ where the code lives.

.. _Git for Windows: https://gitforwindows.org/
.. _pygame github repo: https://github.com/pygame/pygame

Run the pygame install commands
===============================

In the Developer Command Prompt for visual studio use::

  set DISTUTILS_USE_SDK=1
  set MSSdk=1
  git clone https://github.com/pygame/pygame.git
  cd pygame
  py -m pip install setuptools requests wheel numpy -U
  py -m buildconfig --download
  py -m pip install .
  py -m pygame.examples.aliens

More information
================

* `pypy windows compilers page`_

* `Python Wiki - WindowsCompilers`_

.. _pypy windows compilers page:
   http://doc.pypy.org/en/latest/windows.html#
   installing-build-tools-for-visual-studio-2015-for-python-3
.. _Python Wiki - WindowsCompilers: https://wiki.python.org/moin/
   WindowsCompilers

-----
MacOS
-----

These should work on both x86_64 and arm64 macs, running MacOS 10 and above

1. Install `Homebrew`_ — instructions found on link.
2. Install SDL dependencies::
     
     brew install sdl2 sdl2_image sdl2_mixer sdl2_ttf pkg-config
     
3. Install XQuartz [Seems to be optional]::
     
     brew install Caskroom/cask/xquartz
     
4. Install portmidi.

   This step is definitely optional, and may fail on your system (it
   fails on mine). If you don't run this, or it fails, pygame.midi
   won't work, which is fine for most people. (step 5, testing, will
   tell you if this worked or not).

   ::
       
      brew install portmidi
       
5. Install latest Pygame from source::
     
     python3 -m pip install git+https://github.com/pygame/pygame.git
     
6. Verify all Pygame Tests::
     
     python3 -m pygame.tests

.. _Homebrew: https://brew.sh/
     
------
Ubuntu
------

To install dependencies, clone pygame, and compile it, enter the
following commands in your terminal. These instructions are tested to
work on Ubuntu 18.04 and higher::

  sudo apt-get update sudo apt-get install libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev libfreetype6-dev libportmidi-dev libjpeg-dev python3-setuptools python3-dev python3-numpy
  git clone https://github.com/pygame/pygame.git
  cd pygame
  python3 setup.py -config -auto
  python3 setup.py build install --user

------
Fedora
------

Note: This procedure has been tested with Fedora 39.

1. Make sure the ``rpmfusion-free-release`` repository has been added (so we can
   add RPM Sphere repository)::

     sudo dnf install https://mirrors.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm

   See the `RPM Fusion docs`_ for additional information.

2. Install the ``rpmsphere-release`` package to add the RPM Sphere
   repository (so we can install ``smpeg-devel``)::

     sudo dnf install https://github.com/rpmsphere/noarch/raw/master/r/rpmsphere-release-38-1.noarch.rpm

   See `pkgs.org`_ and `RPM Sphere`_ for more information.

3. Install pygame dependencies::

     sudo yum install python-devel python3-cython numpy gcc dpkg-dev SDL2_image-devel SDL2_mixer-devel SDL2_ttf-devel SDL2-devel freetype-devel libjpeg-turbo-devel smpeg-devel portmidi-devel

4. Clone and build pygame::

     git clone https://github.com/pygame/pygame.git
     cd pygame
     python3 setup.py -config -auto
     python3 setup.py build install --user

.. _RPM Fusion docs: https://rpmfusion.org/Configuration
.. _pkgs.org: https://pkgs.org/download/smpeg-devel
.. _RPM Sphere: https://rpmsphere.github.io/
