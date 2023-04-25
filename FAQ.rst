.. image:: https://cdn.pixabay.com/photo/2017/01/31/23/00/faq-2027970_960_720.png
  :alt: FAQ 
  :target: https://www.pygame.org/ 


Frequently Asked Questions and Answers 
=======================================

Summary 
-------
The purpose of this document is to provide answers to the most frequently
asked questions of new users and contributors in Pygame. 

What is the license of Pygame?
------------------------------
Pygame is published under the GNU LESSER GENERAL PUBLIC LICENSE version2.1
which among others allows for the free and open distribution of the source
code and protects contributors from any liability if their code is being
used for another project. For more information visit the license page 
in the repository in the following link:
https://github.com/DeanDro/pygame_contr/blob/main/docs/LGPL.txt 


What are some popular games built with Pygame? 
----------------------------------------------
There are several examples of games developed with Pygame and it would be
impossible to evaluate all of them in a list of top 10 or most exchiting,
etc. Moreover, new games are developed every day and some of them are not
becoming available to the wider public soon or ever.

Looking online on different websites, we have prepared a short list of
some popular games built with Pygame, but certaintly there are many we 
have left out:  

* Metin 2: https://gameforge.com/en-US/play/metin2 

* Flappy Bird: https://flappybird.io/ 

* Super Potato Bruh: https://www.pygame.org/project/3687/6102

* SREM: https://github.com/lukasz1985/SREM 

* Frets on Fire: https://fretsonfire.sourceforge.net/ 

The list goes on and on, so this is only a sample of games written with
Pygame. Make sure to check online for more examples to get inspired.


Can I build 3D games with Pygame?
-----------------------------------
The short answer is not easily! Pygame was built as a 2D library for
simple 2D animation and games to help developers build simple fun projects.
However, Pygame has become a very popular library with thousands of users
and hundreds of contributors. As a result, several open source libraries that
are using Pygame in the backend to build 3D game engines. One good example
is PyEngine3D (https://github.com/ubuntunux/PyEngine3D), which is an open
source engine that uses pygame and pyglet at the backend. 

If you are already somewhat familiar with Pygame and you want to start
learning the 3D game development, you can use Pygame to learn the basics
of 3D animation. A good tutorial is available from Peter Collingridge
(https://www.petercollingridge.co.uk/tutorials/3d/pygame/).

Nevertheless, our suggestion if you are a beginner is to start
with Pygame before jumping into the world of 3D development. 


Is Pygame only good for game development? 
-----------------------------------------
While Pygame is primarily focused on 2D game development, it doesn't mean
that this is only what it is good for. Pygame is generally good for other
types of multimedia applications such as photo viewers, visual simulations,
presentation tools, etc. Some examples of such applications are: 

* PyBonFire: https://www.pygame.org/project/162

* PySeen: https://www.pygame.org/project/146

* IMGV: https://www.pygame.org/project/54

* Pyntor: https://www.pygame.org/project/214


Are there any competitions for Pygame developers? 
-------------------------------------------------
Pygame has thousands of users and hundreds of contributors. So it is no
surpise that there are also competitions developed around Pygame. The most
resent one was on March 2023 with pygames hachathons that run until 17th 
April 2023 (https://pygames.devpost.com/). In addition to that there are
several game hackathons that take place online or in-person. 
A good resource if you want to check upcoming online and in-person hackathons
is DevPost (https://devpost.com/).


What if I am unable to install Pygame using pip?
------------------------------------------------
If you are running on Linux the easiest way to install Pygame using pip is

1. Install build dependencies:

``sudo apt-get build-dep python-pygame``

2. Install mercurial to use hg 

``sudo apt-get install mercurial``

Windows user can use the mercurial installer, which can be found at 
https://wiki.mercurial-scm.org/Download 

3. Use pip to install Pygame 

``pip install hg+http://bitbucket.org/pygame/pygame``

If the above command gives freetype-config: not found error then
try: sudo apt-get install libfreetype6-dev and repeat step 3. 


What is the easiest way of installing Pygame on Ubuntu?
--------------------------------------------------------------------
The easiest way of installing Pygame on Ubuntu is through apt. Specifically,
run the command ``sudo apt-get install python-pygame``. 

``pip`` is better for pure python dependencies or if you are using it 
in virtual environment. 


Where can I find tutorials for Pygame?
--------------------------------------
There are plenty of tutorials available in this repository that are also
available on the website version of Pygame. The links to these locations
are: https://github.com/DeanDro/pygame_contr/tree/main/docs/reST/tut 
and https://www.pygame.org/docs/  


How to create a virtual environment and install Pygame?
--------------------------------------------------------
The easiest way of using Pygame locally is by installing it within a virtual
environment.

If you work on Windows this is a simple process following these steps: 

1. Open ``Command Prompt`` and navigate in the folder you want to store your
project.

2. Within the folder give the following command: 

``python -m venv <name_of_environment>``

3. Now you have created the virtal environment, but you need to move into
it. Give the command: ``cd <name_of_environmnt>``. 

Once you are in the virtual environment, you can activate it by running
the following commands: 

Windows: ``python Scripts\activate.bat``

Linux/MacOS: ``python bin/activate``

Now that have activated the the environment type: ``pip install pygame`` 

This will download pygame and install it in your environment for use.

Optional 
~~~~~~~~~
It is common that our virtual environment is outdated using older versions
of the dependencies. To ensure that everything is up-to-date run the command
``venv --upgrade-deps`` 
This wll update pip and the setup tools to the latest version of PyPI. 


What if I get ImportError: No module named pygame found? 
---------------------------------------------------------
If you get an ImportError, it means your program cannot find the pygame 
library in the modules folder. To ensure everything has been installed 
correctly and to fix the problem follow these steps: 

1. Open command prompt and navigate to the folder where your project lives.
Once you are in the folder give the following command: 
``python -m pip install pygame`` 

This should install pygame manually in your project. 

2. To ensure pygame has been installed correctly, create a new file in your
project with the following content: 

``import pygame``
``print(pygame.ver)``


How to solve DLL load failed error in Win32 application? 
---------------------------------------------------------
If you are getting an error that says: 
``DLL load failed: 1% is not a valid Win32 application``

that possibly be due to your OS architecture. If your system is 64 Bt,
then you need to install both the 32 bit version Python 3.9 and Pygame for
32 Bit. 

Alternatively, ensure that you have installed the Pygame 64 Bit version and
if you haven't then replace it with Pygame 64 Bit. 


What if I get ModuleNotFoundError: No module named pygame.base error?
----------------------------------------------------------------------
If you are getting this error don't try re-installing pygame before you have
removed the previous version. Start by typing: ``pip3 uninstall pygame``. 
This will remove the existing version of pygame. 

Then give the command: ``using pip3 install pygame``. 
If that doesn't work, try ``pip install pygame-menu==2.0.1``. 

