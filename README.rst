import pygame
import random
import sys

# পাইগেম শুরু করা
pygame.init()

# স্ক্রিনের সাইজ নির্ধারণ
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 400
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Zombie Chase: Infinite Run")

# রঙের কোড
WHITE = (255, 255, 255)
GRAY = (50, 50, 50)
RED = (200, 30, 30)
GREEN = (30, 200, 30)
BLACK = (0, 0, 0)

# ক্লক সেটআপ (এফপিএস নিয়ন্ত্রণের জন্য)
clock = pygame.time.Clock()
FPS = 60

# প্লেয়ার (মানুষ) ক্লাস
class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((40, 60))
        self.image.fill(GREEN)  # মানুষটিকে সবুজ ব্লকে দেখানো হয়েছে
        self.rect = self.image.get_rect()
        self.rect.x = 200
        self.rect.y = SCREEN_HEIGHT - 110
        self.gravity = 1
        self.jump_speed = -16
        self.velocity_y = 0
        self.is_jumping = False

    def handle_jump(self):
        if not self.is_jumping:
            self.velocity_y = self.jump_speed
            self.is_jumping = True

    def update(self):
        # গ্র্যাভিটি বা অভিকর্ষ বল প্রয়োগ
        self.velocity_y += self.gravity
        self.rect.y += self.velocity_y

        # মাটিতে ল্যান্ড করা নিশ্চিত করা
        floor_level = SCREEN_HEIGHT - 110
        if self.rect.y >= floor_level:
            self.rect.y = floor_level
            self.velocity_y = 0
            self.is_jumping = False

# জম্বি ক্লাস (প্লেয়ারের পেছনে থাকবে, কখনো ছুঁতে পারবে না)
class Zombie(pygame.sprite.Sprite):
    def __init__(self, x_offset):
        super().__init__()
        self.image = pygame.Surface((35, 55))
        self.image.fill(RED)  # জম্বি লাল ব্লকে দেখানো হয়েছে
        self.rect = self.image.get_rect()
        self.rect.x = x_offset
        self.rect.y = SCREEN_HEIGHT - 105

# বাধা (Obstacle) ক্লাস
class Obstacle(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.width = random.choice([20, 30, 40])
        self.height = random.choice([30, 50, 65])
        self.image = pygame.Surface((self.width, self.height))
        self.image.fill(BLACK)  # বাধা কালো ব্লকে দেখানো হয়েছে
        self.rect = self.image.get_rect()
        self.rect.x = SCREEN_WIDTH + random.randint(100, 300)
        self.rect.y = SCREEN_HEIGHT - 50 - self.height

    def update(self, speed):
        self.rect.x -= speed
        # স্ক্রিনের বাইরে চলে গেলে মুছে ফেলা
        if self.rect.x < -self.width:
            self.kill()

# গেমের মূল ফাংশন
def main():
    player = Player()
    player_group = pygame.sprite.GroupSingle(player)

    # পেছনে অসংখ্য জম্বি তৈরি (তারা প্লেয়ারের পেছনে তাড়া করবে কিন্তু ছুঁতে পারবে না)
    zombie_group = pygame.sprite.Group()
    for i in range(15):
        zombie = Zombie(random.randint(10, 120))
        zombie_group.add(zombie)

    obstacle_group = pygame.sprite.Group()

    score = 0
    game_speed = 6
    font = pygame.font.SysFont("Arial", 24)
    game_over_font = pygame.font.SysFont("Arial", 48)

    # বাধা তৈরির টাইমার
    obstacle_timer = pygame.USEREVENT + 1
    pygame.time.set_timer(obstacle_timer, 1500)

    running = True
    game_over = False

    while running:
        # ইভেন্ট হ্যান্ডলিং
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()
            
            if event.type == obstacle_timer and not game_over:
                # নতুন বাধা তৈরি
                obstacle_group.add(Obstacle())

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:  # স্পেসবার চেপে লাফানো
                    if game_over:
                        # গেম ওভার হলে স্পেস টিপলে আবার নতুন করে শুরু হবে
                        main()
                    else:
                        player.handle_jump()

        if not game_over:
            # আপডেট
            player_group.update()
            obstacle_group.update(game_speed)

            # স্কোর এবং স্পিড বৃদ্ধি
            score += 1
            if score % 500 == 0:
                game_speed += 1

            # কলিশন চেক (মানুষটি যদি বাধার সাথে ধাক্কা খায় তবে সে থেমে যাবে)
            if pygame.sprite.spritecollide(player, obstacle_group, False):
                game_over = True

        # ড্রয়িং (স্ক্রিনে সবকিছু দেখানো)
        screen.fill(WHITE)

        # মাটি তৈরি করা
        pygame.draw.rect(screen, GRAY, (0, SCREEN_HEIGHT - 50, SCREEN_WIDTH, 50))

        # অবজেক্টগুলো স্ক্রিনে আঁকা
        zombie_group.draw(screen)
        player_group.draw(screen)
        obstacle_group.draw(screen)

        # স্কোর প্রদর্শন
        score_surface = font.render(f"Score: {score // 10}", True, BLACK)
        screen.blit(score_surface, (10, 10))

        # গেম ওভার স্ক্রিন (মানুষটি থেমে গেলে জম্বিরা তাকে ধরে ফেলবে)
        if game_over:
            over_surface = game_over_font.render("GAME OVER - Zombies Caught You!", True, RED)
            restart_surface = font.render("Press SPACE to Restart", True, BLACK)
            screen.blit(over_surface, (100, SCREEN_HEIGHT // 2 - 30))
            screen.blit(restart_surface, (300, SCREEN_HEIGHT // 2 + 30))

        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    main()
.. image:: https://raw.githubusercontent.com/pygame/pygame/main/docs/reST/_static/pygame_logo.svg
  :alt: pygame
  :target: https://www.pygame.org/


|AppVeyorBuild| |PyPiVersion| |PyPiLicense|
|Python3| |GithubCommits| |BlackFormatBadge|

Pygame_ is a free and open-source cross-platform library
for the development of multimedia applications like video games using Python.
It uses the `Simple DirectMedia Layer library`_ and several other
popular libraries to abstract the most common functions, making writing
these programs a more intuitive task.

`We need your help`_ to make pygame the best it can be!
New contributors are welcome.


Installation
------------

Before installing pygame, you must check that Python is installed
on your machine. To find out, open a command prompt (if you have
Windows) or a terminal (if you have MacOS or Linux) and type this:
::

   python --version


If a message such as "Python 3.8.10" appears, it means that Python
is correctly installed. If an error message appears, it means that
it is not installed yet. You must then go to the `official website
<https://www.python.org/downloads/>`_ to download it.

Once Python is installed, you have to perform a final check: you have
to see if pip is installed. Generally, pip is pre-installed with
Python but we are never sure. Same as for Python, type the following
command:
::

   pip --version


If a message such as "pip 20.0.2 from /usr/lib/python3/dist-packages/pip
(python 3.8)" appears, you are ready to install pygame! To install
it, enter this command:
::

   pip install pygame

Once pygame is installed, quickly test your library by entering the following
command, which opens one of the many example games that comes pre-installed: 
::

	python3 -m pygame.examples.aliens


If this doesn’t work, the `Getting Started 
<https://www.pygame.org/wiki/GettingStarted/>`_ section of the official 
website has more information for platform specific issues, such as adding
python to your machine’s PATH settings


Help
----

If you are just getting started with pygame, you should be able to
get started fairly quickly.  Pygame comes with many tutorials and
introductions.  There is also full reference documentation for the
entire library. Browse the documentation on the `docs page`_. You
can also browse the documentation locally by running
``python -m pygame.docs`` in your terminal. If the docs aren't found
locally, it'll launch the online website instead.

The online documentation stays up to date with the development version
of pygame on GitHub.  This may be a bit newer than the version of pygame
you are using. To upgrade to the latest full release, run
``pip install pygame --upgrade`` in your terminal.

Best of all, the examples directory has many playable small programs
which can get you started playing with the code right away.


Features
----------

Pygame is a powerful library for game development, offering a wide
range of features to simplify your coding journey. Let's delve into
what pygame has to offer:

Graphics - With pygame, creating dynamic and engaging graphics has
never been easier. The library provides simple yet effective tools for
2D graphics and animation, including support for images, rectangles,
and polygon shapes. Whether you're a seasoned game developer or just
starting out, pygame has you covered.

Sound - Pygame also includes support for playing and manipulating sound
and music, making it easy to add sound effects and background music to
your games. With support for WAV, MP3, and OGG file formats, you have
plenty of options to choose from.

Input - Pygame provides intuitive functions for handling keyboard, mouse,
and joystick input, allowing you to quickly and easily implement player
controls in your games. No more struggling with complex input code, pygame
makes it simple.

Game Development - Lastly, pygame provides a comprehensive suite of tools
and features specifically designed for game development. From collision
detection to sprite management, pygame has everything you need to create
exciting and engaging games. Whether you're building a platformer, puzzle
game, or anything in between, pygame has you covered.


Building From Source
--------------------

If you want to use features that are currently in development,
or you want to contribute to pygame, you will need to build pygame
locally from its source code, rather than pip installing it.

Installing from source is fairly automated. The most work will
involve compiling and installing all the pygame dependencies.  Once
that is done, run the ``setup.py`` script which will attempt to
auto-configure, build, and install pygame.

Much more information about installing and compiling is available
on the `Compilation wiki page`_.

Contribute
----------

* `Documentation Contributions <https://github.com/pygame/pygame/tree/main/docs>`_ - Guidelines for contributing to the main documentations
* `Writing your first unit test <http://renesd.blogspot.com/2019/11/draft-2-of-lets-write-unit-test.html>`_ - Step by step guide on how to write your first unit test in Python for Pygame.
* `How to Hack Pygame <https://www.pygame.org/wiki/Hacking>`_ - Information on hacking, developing, and modifying Pygame
* `Issue Tracker for beginners <https://github.com/pygame/pygame/labels/good%20first%20issue>`_ - A way for beginners to contribute to the project
* `Bugs & Patches <https://www.pygame.org/wiki/patchesandbugs>`_ - Report bugs
* `Communication tools <https://www.pygame.org/wiki/info>`_ - More information and ways to get in touch with the Pygame team


Credits
-------

Thanks to everyone who has helped contribute to this library.
Special thanks are also in order.

* Marcus Von Appen: many changes, and fixes, 1.7.1+ freebsd maintainer
* Lenard Lindstrom: the 1.8+ windows maintainer, many changes, and fixes
* Brian Fisher for svn auto builder, bug tracker and many contributions
* Rene Dudfield: many changes, and fixes, 1.7+ release manager/maintainer
* Phil Hassey for his work on the pygame.org website
* DR0ID for his work on the sprite module
* Richard Goedeken for his smoothscale function
* Ulf Ekström for his pixel perfect collision detection code
* Pete Shinners: original author
* David Clark for filling the right-hand-man position
* Ed Boraas and Francis Irving: Debian packages
* Maxim Sobolev: FreeBSD packaging
* Bob Ippolito: MacOS and OS X porting (much work!)
* Jan Ekhol, Ray Kelm, and Peter Nicolai: putting up with early design ideas
* Nat Pryce for starting our unit tests
* Dan Richter for documentation work
* TheCorruptor for his incredible logos and graphics
* Nicholas Dudfield: many test improvements
* Alex Folkner for pygame-ctypes

Thanks to those sending in patches and fixes: Niki Spahiev, Gordon
Tyler, Nathaniel Pryce, Dave Wallace, John Popplewell, Michael Urman,
Andrew Straw, Michael Hudson, Ole Martin Bjoerndalen, Herve Cauwelier,
James Mazer, Lalo Martins, Timothy Stranex, Chad Lester, Matthias
Spiller, Bo Jangeborg, Dmitry Borisov, Campbell Barton, Diego Essaya,
Eyal Lotem, Regis Desgroppes, Emmanuel Hainry, Randy Kaelber
Matthew L Daniel, Nirav Patel, Forrest Voight, Charlie Nolan,
Frankie Robertson, John Krukoff, Lorenz Quack, Nick Irvine,
Michael George, Saul Spatz, Thomas Ibbotson, Tom Rothamel, Evan Kroske,
Cambell Barton.

And our bug hunters above and beyond: Angus, Guillaume Proux, Frank
Raiser, Austin Henry, Kaweh Kazemi, Arturo Aldama, Mike Mulcheck,
Michael Benfield, David Lau

There's many more folks out there who've submitted helpful ideas, kept
this project going, and basically made our life easier.  Thanks!

Many thank you's for people making documentation comments, and adding to the
pygame.org wiki.

Also many thanks for people creating games and putting them on the
pygame.org website for others to learn from and enjoy.

Lots of thanks to James Paige for hosting the pygame bugzilla.

Also a big thanks to Roger Dingledine and the crew at SEUL.ORG for our
excellent hosting.

Dependencies
------------

Pygame is obviously strongly dependent on SDL and Python.  It also
links to and embeds several other smaller libraries.  The font
module relies on SDL_ttf, which is dependent on freetype.  The mixer
(and mixer.music) modules depend on SDL_mixer.  The image module
depends on SDL_image, which also can use libjpeg and libpng.  The
transform module has an embedded version of SDL_rotozoom for its
own rotozoom function.  The surfarray module requires the Python
NumPy package for its multidimensional numeric arrays.
Dependency versions:


+----------+------------------------+
| CPython  | >= 3.6 (Or use PyPy3)  |
+----------+------------------------+
| SDL      | >= 2.0.8               |
+----------+------------------------+
| SDL_mixer| >= 2.0.0               |
+----------+------------------------+
| SDL_image| >= 2.0.2               |
+----------+------------------------+
| SDL_ttf  | >= 2.0.11              |
+----------+------------------------+
| SDL_gfx  | (Optional, vendored in)|
+----------+------------------------+
| NumPy    | >= 1.6.2 (Optional)    |
+----------+------------------------+



License
-------

This library is distributed under `GNU LGPL version 2.1`_, which can
be found in the file ``docs/LGPL.txt``.  We reserve the right to place
future versions of this library under a different license.

This basically means you can use pygame in any project you want,
but if you make any changes or additions to pygame itself, those
must be released with a compatible license (preferably submitted
back to the pygame project).  Closed source and commercial games are fine.

The programs in the ``examples`` subdirectory are in the public domain.

See docs/licenses for licenses of dependencies.


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

.. _pygame: https://www.pygame.org
.. _Simple DirectMedia Layer library: https://www.libsdl.org
.. _We need your help: https://www.pygame.org/contribute.html
.. _Compilation wiki page: https://www.pygame.org/wiki/Compilation
.. _docs page: https://www.pygame.org/docs/
.. _GNU LGPL version 2.1: https://www.gnu.org/copyleft/lesser.html
