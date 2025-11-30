.. TUTORIAL:Import and Initialize

.. include:: commonimport pygame
import time
import random

# Initialize Pygame
pygame.init()

# Screen settings
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake Game")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Snake settings
snake_block = 10
snake_speed = 15

# Fonts
font_style = pygame.font.SysFont("bahnschrift", 25)
score_font = pygame.font.SysFont("comicsansms", 35)


def score_display(score):
    value = score_font.render("Score: " + str(score), True, GREEN)
    screen.blit(value, [0, 0])


def draw_snake(snake_block, snake_list):
    for x in snake_list:
        pygame.draw.rect(screen, BLUE, [x[0], x[1], snake_block, snake_block])


def message(msg, color):
    msg_surface = font_style.render(msg, True, color)
    screen.blit(msg_surface, [WIDTH / 6, HEIGHT / 3])


def game_loop():
    game_over = False
    game_close = False

    x1, y1 = WIDTH / 2, HEIGHT / 2
    x1_change, y1_change = 0, 0

    snake_list = []
    length_of_snake = 1

    # Food position
    foodx = round(random.randrange(0, WIDTH - snake_block) / 10.0) * 10.0
    foody = round(random.randrange(0, HEIGHT - snake_block) / 10.0) * 10.0

    while not game_over:

        while game_close:
            screen.fill(BLACK)
            message("You lost! Press C to Play Again or Q to Quit", RED)
            score_display(length_of_snake - 1)
            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        game_over = True
                        game_close = False
                    if event.key == pygame.K_c:
                        game_loop()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    x1_change = -snake_block
                    y1_change = 0
                elif event.key == pygame.K_RIGHT:
                    x1_change = snake_block
                    y1_change = 0
                elif event.key == pygame.K_UP:
                    y1_change = -snake_block
                    x1_change = 0
                elif event.key == pygame.K_DOWN:
                    y1_change = snake_block
                    x1_change = 0

        if x1 >= WIDTH or x1 < 0 or y1 >= HEIGHT or y1 < 0:
            game_close = True
        x1 += x1_change
        y1 += y1_change
        screen.fill(BLACK)
        pygame.draw.rect(screen, GREEN, [foodx, foody, snake_block, snake_block])

        snake_head = [x1, y1]
        snake_list.append(snake_head)
        if len(snake_list) > length_of_snake:
            del snake_list[0]

        for block in snake_list[:-1]:
            if block == snake_head:
                game_close = True

        draw_snake(snake_block, snake_list)
        score_display(length_of_snake - 1)

        pygame.display.update()

        if x1 == foodx and y1 == foody:
            foodx = round(random.randrange(0, WIDTH - snake_block) / 10.0) * 10.0
            foody = round(random.randrange(0, HEIGHT - snake_block) / 10.0) * 10.0
            length_of_snake += 1

        time.sleep(0.1)

    pygame.quit()
    quit()


game_loop().txt

********************************************
  Pygame Tutorials - Import and Initialize
********************************************
 
Import and Initialize
=====================

.. rst-class:: docinfo

:Author: Pete Shinners
:Contact: pete@shinners.org


Getting pygame imported and initialized is a very simple process. It is also
flexible enough to give you control over what is happening. Pygame is a
collection of different modules in a single python package. Some of the
modules are written in C, and some are written in python. Some modules
are also optional, and might not always be present.

This is just a quick introduction on what is going on when you import pygame.
For a clearer explanation definitely see the pygame examples.


Import
------

First we must import the pygame package. Since pygame version 1.4 this
has been updated to be much easier. Most games will import all of pygame like this. ::

  import pygame
  from pygame.locals import *

The first line here is the only necessary one. It imports all the available pygame
modules into the pygame package. The second line is optional, and puts a limited
set of constants and functions into the global namespace of your script.

An important thing to keep in mind is that several pygame modules are optional.
For example, one of these is the font module. When  you "import pygame", pygame
will check to see if the font module is available. If the font module is available
it will be imported as "pygame.font". If the module is not available, "pygame.font"
will be set to None. This makes it fairly easy to later on test if the font module is available.


Init
----

Before you can do much with pygame, you will need to initialize it. The most common
way to do this is just make one call. ::

  pygame.init()

This will attempt to initialize all the pygame modules for you. Not all pygame modules
need to be initialized, but this will automatically initialize the ones that do. You can
also easily initialize each pygame module by hand. For example to only initialize the
font module you would just call. ::

  pygame.font.init()

Note that if there is an error when you initialize with "pygame.init()", it will silently fail.
When hand initializing modules like this, any errors will raise an exception. Any
modules that must be initialized also have a "get_init()" function, which will return true
if the module has been initialized.

It is safe to call the init() function for any module more than once.


Quit
----

Modules that are initialized also usually have a quit() function that will clean up.
There is no need to explicitly call these, as pygame will cleanly quit all the
initialized modules when python finishes.
