import pygame
import random
import sys

# Inisialisasi Pygame
pygame.init()

# Konstanta Layar & Warna
WIDTH, HEIGHT = 800, 400
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Spongebob Indonesia Run")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (240, 200, 0)
RED = (200, 50, 50)
SKY_BLUE = (135, 206, 235)
BROWN = (139, 69, 19)

FPS = 60
clock = pygame.time.Clock()

# Karakter Spongebob
class Player:
    def __init__(self):
        self.x = 80
        self.y = 300
        self.width = 40
        self.height = 50
        self.vel_y = 0
        self.is_jumping = False
        self.gravity = 0.8

    def jump(self):
        if not self.is_jumping:
            self.vel_y = -14
            self.is_jumping = True

    def update(self):
        self.vel_y += self.gravity
        self.y += self.vel_y

        # Batas tanah
        if self.y >= 300:
            self.y = 300
            self.is_jumping = False

    def draw(self):
        # Menggunakan bentuk sederhana (Kotak Kuning)
        pygame.draw.rect(SCREEN, YELLOW, (self.x, self.y, self.width, self.height))
        # Mata Spongebob
        pygame.draw.circle(SCREEN, WHITE, (self.x + 12, self.y + 15), 6)
        pygame.draw.circle(SCREEN, WHITE, (self.x + 28, self.y + 15), 6)

# Rintangan (Lubang/Bus/Batu)
class Obstacle:
    def __init__(self, speed):
        self.x = WIDTH
        self.width = random.choice([30, 50])
        self.height = random.choice([40, 60])
        self.y = 350 - self.height
        self.speed = speed

    def update(self):
        self.x -= self.speed

    def draw(self):
        pygame.draw.rect(SCREEN, RED, (self.x, self.y, self.width, self.height))

def main():
    player = Player()
    obstacles = []
    spawn_timer = 0
    game_speed = 6
    score = 0
    font = pygame.font.SysFont("Arial", 24)
    running = True

    while running:
        clock.tick(FPS)
        SCREEN.fill(SKY_BLUE)

        # Gambar Jalanan
        pygame.draw.rect(SCREEN, BROWN, (0, 350, WIDTH, 50))

        # Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE or event.key == pygame.K_UP:
                    player.jump()

        # Update Player
        player.update()
        player.draw()

        # Spawn Rintangan
        spawn_timer += 1
        if spawn_timer > random.randint(60, 110):
            obstacles.append(Obstacle(game_speed))
            spawn_timer = 0

        # Update & Cek Tabrakan Rintangan
        for obstacle in list(obstacles):
            obstacle.update()
            obstacle.draw()

            # Deteksi Tabrakan (Hitbox)
            player_rect = pygame.Rect(player.x, player.y, player.width, player.height)
            obstacle_rect = pygame.Rect(obstacle.x, obstacle.y, obstacle.width, obstacle.height)

            if player_rect.colliderect(obstacle_rect):
                print(f"Game Over! Skor Akhir: {score}")
                running = False

            # Hapus rintangan keluar layar & tambah skor
            if obstacle.x + obstacle.width < 0:
                obstacles.remove(obstacle)
                score += 10
                if score % 50 == 0:
                    game_speed += 0.5  # Makin lama makin cepat

        # Tampilkan Skor
        score_text = font.render(f"Skor: {score}", True, BLACK)
        SCREEN.blit(score_text, (20, 20))

        pygame.display.update()

if __name__ == "__main__":
    main()
Pygame Front Page
=================

.. toctree::
   :maxdepth: 2
   :glob:
   :hidden:

   ref/*
   tut/*
   tut/en/**/*
   tut/ko/**/*
   c_api
   filepaths
   logos

Quick start
-----------

Welcome to pygame! Once you've got pygame installed (:code:`pip install pygame` or
:code:`pip3 install pygame` for most people), the next question is how to get a game
loop running. Pygame, unlike some other libraries, gives you full control of program
execution. That freedom means it is easy to mess up in your initial steps.

Here is a good example of a basic setup (opens the window, updates the screen, and handles events)--

.. literalinclude:: ref/code_examples/base_script.py

Here is a slightly more fleshed out example, which shows you how to move something
(a circle in this case) around on screen--

.. literalinclude:: ref/code_examples/base_script_example.py

For more in depth reference, check out the :ref:`tutorials-reference-label`
section below, check out a video tutorial (`I'm a fan of this one
<https://www.youtube.com/watch?v=AY9MnQ4x3zk>`_), or reference the API
documentation by module.

Documents
---------

`Readme`_
  Basic information about pygame: what it is, who is involved, and where to find it.

`Install`_
  Steps needed to compile pygame on several platforms.
  Also help on finding and installing prebuilt binaries for your system.

:doc:`filepaths`
  How pygame handles file system paths.

:doc:`Pygame Logos <logos>`
   The logos of Pygame in different resolutions.


`LGPL License`_
  This is the license pygame is distributed under.
  It provides for pygame to be distributed with open source and commercial software.
  Generally, if pygame is not changed, it can be used with any type of program.

.. _tutorials-reference-label:

Tutorials
---------

:doc:`Introduction to Pygame <tut/PygameIntro>`
  An introduction to the basics of pygame.
  This is written for users of Python and appeared in volume two of the Py magazine.

:doc:`Import and Initialize <tut/ImportInit>`
  The beginning steps on importing and initializing pygame.
  The pygame package is made of several modules.
  Some modules are not included on all platforms.

:doc:`How do I move an Image? <tut/MoveIt>`
  A basic tutorial that covers the concepts behind 2D computer animation.
  Information about drawing and clearing objects to make them appear animated.

:doc:`Chimp Tutorial, Line by Line <tut/ChimpLineByLine>`
  The pygame examples include a simple program with an interactive fist and a chimpanzee.
  This was inspired by the annoying flash banner of the early 2000s.
  This tutorial examines every line of code used in the example.

:doc:`Sprite Module Introduction <tut/SpriteIntro>`
  Pygame includes a higher level sprite module to help organize games.
  The sprite module includes several classes that help manage details found in almost all games types.
  The Sprite classes are a bit more advanced than the regular pygame modules,
  and need more understanding to be properly used.

:doc:`Surfarray Introduction <tut/SurfarrayIntro>`
  Pygame used the NumPy python module to allow efficient per pixel effects on images.
  Using the surface arrays is an advanced feature that allows custom effects and filters.
  This also examines some of the simple effects from the pygame example, arraydemo.py.

:doc:`Camera Module Introduction <tut/CameraIntro>`
  Pygame, as of 1.9, has a camera module that allows you to capture images,
  watch live streams, and do some basic computer vision.
  This tutorial covers those use cases.

:doc:`Newbie Guide <tut/newbieguide>`
  A list of thirteen helpful tips for people to get comfortable using pygame.

:doc:`Making Games Tutorial <tut/MakeGames>`
  A large tutorial that covers the bigger topics needed to create an entire game.

:doc:`Display Modes <tut/DisplayModes>`
  Getting a display surface for the screen.

:doc:`한국어 튜토리얼 (Korean Tutorial) <tut/ko/빨간블록 검은블록/개요>`
  빨간블록 검은블록


Reference
---------

:ref:`genindex`
  A list of all functions, classes, and methods in the pygame package.

:doc:`ref/bufferproxy`
  An array protocol view of surface pixels

:doc:`ref/color`
  Color representation.

:doc:`ref/cursors`
  Loading and compiling cursor images.

:doc:`ref/display`
  Configure the display surface.

:doc:`ref/draw`
  Drawing simple shapes like lines and ellipses to surfaces.

:doc:`ref/event`
  Manage the incoming events from various input devices and the windowing platform.

:doc:`ref/examples`
  Various programs demonstrating the use of individual pygame modules.

:doc:`ref/font`
  Loading and rendering TrueType fonts.

:doc:`ref/freetype`
  Enhanced pygame module for loading and rendering font faces.

:doc:`ref/gfxdraw`
  Anti-aliasing draw functions.

:doc:`ref/image`
  Loading, saving, and transferring of surfaces.

:doc:`ref/joystick`
  Manage the joystick devices.

:doc:`ref/key`
  Manage the keyboard device.

:doc:`ref/locals`
  Pygame constants.

:doc:`ref/mixer`
  Load and play sounds

:doc:`ref/mouse`
  Manage the mouse device and display.

:doc:`ref/music`
  Play streaming music tracks.

:doc:`ref/pygame`
  Top level functions to manage pygame.

:doc:`ref/pixelarray`
  Manipulate image pixel data.

:doc:`ref/rect`
  Flexible container for a rectangle.

:doc:`ref/scrap`
  Native clipboard access.

:doc:`ref/sndarray`
  Manipulate sound sample data.

:doc:`ref/sprite`
  Higher level objects to represent game images.

:doc:`ref/surface`
  Objects for images and the screen.

:doc:`ref/surfarray`
  Manipulate image pixel data.

:doc:`ref/tests`
  Test pygame.

:doc:`ref/time`
  Manage timing and framerate.

:doc:`ref/transform`
  Resize and move images.

:doc:`pygame C API <c_api>`
  The C api shared amongst pygame extension modules.

:ref:`search`
  Search pygame documents by keyword.

.. _Readme: ../wiki/about

.. _Install: ../wiki/GettingStarted#Pygame%20Installation

.. _LGPL License: LGPL.txt
