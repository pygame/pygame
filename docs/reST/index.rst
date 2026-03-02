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

.. _LGPL License: LGPL.txtimport pygame
import random
import sys

# Inisialisasi pygame
pygame.init()

# Ukuran layar
WIDTH = 800
HEIGHT = 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Balapan Sederhana - Tekan SPASE untuk Gas!")

clock = pygame.time.Clock()

# Warna
PUTIH = (255, 255, 255)
HITAM = (0, 0, 0)
MERAH = (220, 20, 60)
HIJAU = (50, 205, 50)
ABU = (100, 100, 100)
KUNING = (255, 215, 0)

# Posisi mobil pemain
player_x = WIDTH // 2 - 40
player_y = HEIGHT - 120
player_speed = 0
player_max_speed = 12
acceleration = 0.4
friction = 0.2

# Musuh (mobil lain)
enemies = []
enemy_speed_base = 5
spawn_timer = 0
spawn_interval = 80   # semakin kecil semakin sering muncul

# Skor & status
score = 0
game_over = False
font = pygame.font.SysFont("arial", 36, bold=True)
font_small = pygame.font.SysFont("arial", 24)

def create_enemy():
    x = random.randint(80, WIDTH-140)
    return {"x": x, "y": -80, "speed": enemy_speed_base + random.uniform(-2, 4)}

def draw_road():
    screen.fill((40, 40, 40))
    # garis tengah jalan
    for i in range(-1, 15):
        y = (pygame.time.get_ticks() // 8) % 100 + i * 100
        pygame.draw.rect(screen, KUNING, (WIDTH//2 - 15, y - 400, 30, 60))

def draw_car(x, y, color, is_player=False):
    # badan mobil
    pygame.draw.rect(screen, color, (x, y, 80, 50))
    # atap
    pygame.draw.rect(screen, (color[0]//2, color[1]//2, color[2]//2), (x+10, y-20, 60, 30))
    # jendela
    pygame.draw.rect(screen, HITAM, (x+20, y-15, 40, 20))
    # roda
    pygame.draw.circle(screen, HITAM, (x+15, y+50), 18)
    pygame.draw.circle(screen, HITAM, (x+65, y+50), 18)
    pygame.draw.circle(screen, (180,180,180), (x+15, y+50), 10)
    pygame.draw.circle(screen, (180,180,180), (x+65, y+50), 10)
    
    if is_player:
        # spoiler kecil
        pygame.draw.polygon(screen, MERAH, [(x+30, y-25), (x+50, y-25), (x+40, y-45)])

# Game loop utama
running = True
while running:
    dt = clock.tick(60)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE and game_over:
                # Reset game
                score = 0
                player_y = HEIGHT - 120
                player_speed = 0
                enemies.clear()
                game_over = False
                spawn_timer = 0

    if not game_over:
        keys = pygame.key.get_pressed()
        
        # Kontrol gas & rem
        if keys[pygame.K_SPACE] or keys[pygame.K_UP]:
            player_speed = min(player_max_speed, player_speed + acceleration)
        else:
            player_speed = max(0, player_speed - friction)
            
        # Geser kiri-kanan (selama tidak terlalu cepat)
        if player_speed < 8:
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                player_x = max(60, player_x - 6)
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                player_x = min(WIDTH-140, player_x + 6)

        # Spawn musuh
        spawn_timer += 1
        if spawn_timer >= spawn_interval:
            enemies.append(create_enemy())
            spawn_timer = 0
            # Semakin lama makin susah
            if score > 200 and spawn_interval > 40:
                spawn_interval -= 1

        # Update posisi musuh
        for e in enemies[:]:
            e["y"] += e["speed"] + (score // 80)  # makin cepat seiring skor
            
            # Tabrakan
            if (player_x < e["x"] + 80 and player_x + 80 > e["x"] and
                player_y < e["y"] + 50  and player_y + 50 > e["y"]):
                game_over = True
            
            # Musuh keluar layar → tambah skor
            if e["y"] > HEIGHT + 50:
                enemies.remove(e)
                score += 1

    # Gambar semuanya
    draw_road()
    draw_car(player_x, player_y, HIJAU, is_player=True)

    for e in enemies:
        draw_car(e["x"], e["y"], MERAH)

    # UI
    score_text = font.render(f"SKOR: {score}", True, PUTIH)
    screen.blit(score_text, (20, 20))
    
    speed_text = font_small.render(f"Kecepatan: {int(player_speed*10)} km/jam", True, KUNING)
    screen.blit(speed_text, (20, 70))

    if game_over:
        go_text = font.render("GAME OVER!", True, MERAH)
        restart_text = font_small.render("Tekan SPASI untuk main lagi", True, PUTIH)
        screen.blit(go_text, (WIDTH//2 - go_text.get_width()//2, HEIGHT//2 - 60))
        screen.blit(restart_text, (WIDTH//2 - restart_text.get_width()//2, HEIGHT//2 + 20))

    pygame.display.flip()

pygame.quit()
sys.exit()
