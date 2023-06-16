.. TUTORIAL:Line by Line Descriptions of the Chimp Example

.. include:: common.txt

*************************************************
  Pygame Tutorials - Line By Line Chimp Example
*************************************************


Line By Line Chimp
==================

.. rst-class:: docinfo

:Author: Pete Shinners
:Contact: pete@shinners.org

.. toctree::
   :hidden:

   chimp.py


Introduction
------------

In the *pygame* examples there is a simple example named "chimp".
This example simulates a punchable monkey moving around the screen with
promises of riches and reward. The example itself is very simple, and a
bit thin on error-checking code. This example program demonstrates many of
pygame's abilities, like creating a window, loading images and sounds,
rendering text, and basic event and mouse handling.

The program and images can be found inside the standard source distribution
of pygame. You can run it by running `python -m pygame.examples.chimp` in
your terminal.

This tutorial will go through the code block by block. Explaining how
the code works. There will also be mention of how the code could be improved
and what error checking could help out.

This is an excellent tutorial for people getting their first look at
the *pygame* code. Once *pygame* is fully installed, you can find
and run the chimp demo for yourself in the examples directory.

.. container:: fullwidth leading trailing

   .. rst-class:: small-heading

   (no, this is not a banner ad, it's the screenshot)

   .. image:: chimpshot.gif
      :alt: chimp game banner

   :doc:`Full Source <chimp.py>`


Import Modules
--------------

This is the code that imports all the needed modules into your program.
It also checks for the availability of some of the optional pygame modules. ::

    # Import Modules
    from pathlib import Path
    import sys
    import pygame as pg


First, we import the ``Path`` class from the standard ``pathlib`` python
module. This allows us to do things like create platform independent file
paths. In the next line, we import ``sys`` module in order to use
``sys.exit()``.

Finally we import the ``pygame`` package as ``pg``, so that all of the
functionality of pygame is able to be referenced from the namespace ``pg``.


Loading Resources
-----------------

Here we have two static methods we can use to load images and sounds. We will
look at each method individually in this section. ::

    def load_image(filename, colorkey=None, scale=1):
        filename = Path('data').joinpath(filename)
        image = pg.image.load(filename).convert()

        x, y = image.get_size()
        x = x * scale
        y = y * scale
        image = pg.transform.scale(image, (x,y))

        if colorkey == -1:
            colorkey = image.get_at((0, 0))

        image.set_colorkey(colorkey, pg.RLEACCEL)

        return image, image.get_rect()


This method takes the name of an image to load. It also optionally takes an
argument it can use to set a colorkey for the image, and an argument to scale
the image. A colorkey is used in graphics to represent a color of the image
that is transparent.

The first thing this function does is create a full Path to the filename.
In this example all the resources are in a "data" subdirectory. By using
the ``Path`` class with the ``joinpath`` method, a Path will be created
that works for whatever platform the game is running on.

Next we load the image using the :func:`pygame.image.load` function. We also
chain an important call to the `convert()` method, which gets called on the
Surface object, immediately after the image is loaded. This makes a new copy
of a Surface and converts its color format and depth to match the display.
This means blitting the image to the screen will happen as quickly as
possible.

We then scale the image, using the :func:`pygame.transform.scale` function.
This function takes a Surface and the size it should be scaled to. To scale
by a scalar, we can get the size and scale the x and y by the scalar.

Last, we set the colorkey for the image. If the user supplied an argument
for the colorkey argument we use that value as the colorkey for the image.
This would usually just be a color RGB value, like (255, 255, 255) for
white. You can also pass a value of -1 as the colorkey. In this case the
function will lookup the color at the topleft pixel of the image, and use
that color for the colorkey. ::

    def load_sound(filename):
        if not pg.mixer or not pg.mixer.get_init():
            print("Warning, sound disabled")

            class NoneSound:
                def play(self):
                    pass

            return NoneSound()
        else:
            filename = Path('data').joinpath(filename)
            return pg.mixer.Sound(filename)


Next is the method to load a sound file. The first thing this method does is
check to see if the :mod:`pygame.mixer` module was imported correctly. If not,
it creates & returns a small class instance that has a dummy play method. This
will act enough like a normal Sound object for this game to run without any
extra error checking.

This method is similar to the image loading method, but handles some different
problems. First we create a full Path to the sound image, and load the sound
file. Then we simply return the loaded Sound object.


Game Object Classes
-------------------

Here we create two classes to represent the objects in our game. Almost
all the logic for the game goes into these two classes. We will look over
them one at a time here. ::

    class Fist(pg.sprite.Sprite):
        """Clenched fist (follows mouse)"""

        def __init__(self):
            pg.sprite.Sprite.__init__(self)
            self.image, self.rect = Media.load_image("fist.png", colorkey=-1)
            self.whiff_sound = Media.load_sound("whiff.wav")
            self.punch_sound = Media.load_sound("punch.wav")
            self.state = self.FIST_RETRACTED = (-235, -80)
            self.FIST_PUNCHING = (-220, -55)
            self.add(Game.allsprites)

        def update(self, mouse_buttons):
            # Move fist to mouse position
            self.rect.topleft = pg.mouse.get_pos()
            self.rect.move_ip(self.state)

            # Handle punches
            for event in mouse_buttons:
                if event.type == pg.MOUSEBUTTONDOWN:
                    self.state = self.FIST_PUNCHING
                    self.rect.move_ip((15, 25))
                    self.punch()
                if event.type == pg.MOUSEBUTTONUP:
                    self.state = self.FIST_RETRACTED
                    self.rect.move_ip((-15, -25))

        def punch(self):
            punched = pg.sprite.spritecollide(
                          self,
                          Game.punchables,
                          False,
                          collided=pg.sprite.collide_rect_ratio(.9))

            if len(punched) == 0:
                self.whiff_sound.play()
            else:
                for each in punched:
                    each.get_punched(self)
                    self.punch_sound.play()


Here we create a class to represent the players fist. It is derived from the
`Sprite` class included in the :mod:`pygame.sprite` module. The `__init__`
method is called when new instances of this class are created. The first thing
we do is be sure to call the `__init__` method for our base class. This allows
the Sprite's `__init__` method to prepare our object for use as a sprite.
This game uses one of the sprite drawing Group classes. These classes can draw
sprites that have an "image" and "rect" attribute. By simply changing these
two attributes, the renderer will draw the current image at the current
position.

All sprites have an `update()` method. This method is typically called once
per frame. It is where you should put code that moves and updates the
variables for the sprite. The `update()` method for the fist moves the fist to
the location of the mouse pointer. It also changes the punching state of the
fist, based on the state of the mouse buttons, which offsets the fist position
slightly if the fist is in the "FIST_PUNCHING" state and restores it in the
"FIST_RETRACTED" state.

The `punch()` method checks if the fist Sprite has collided with any of the
sprites in the `Game.punchables` sprite Group and stores the result in the
`punched` list. If the `punched` list is empty then a wiff sound is played.
Otherwise each sprite gets punched via its `get_punched` method, causing it to
react, and a punch sound is played. ::

    class Chimp(pg.sprite.Sprite):
        """Monkey (moves across the screen)
        Spins when punched.
        """

        def __init__(self, topleft=(10,90)):
            pg.sprite.Sprite.__init__(self)
            self.image, self.rect = Media.load_image("chimp.png", colorkey=-1, scale=4)
            self.image_original = self.image
            self.rect.topleft = topleft
            self.rotation = 0
            self.delta = 18
            self.move = self._walk
            self.add((Game.allsprites, Game.punchables))

        def update(self, *nevermint):
            """Walk or spin, depending on state"""
            self.move()

        def _walk(self):
            """Move monkey across screen (turning at ends)"""
            newpos = self.rect.move((self.delta, 0))
            if not Game.screen_rect.contains(newpos):
                self.delta = -self.delta
                newpos = self.rect.move((self.delta, 0))
                self.image = pg.transform.flip(self.image, True, False)

            self.rect = newpos

        def _spin(self):
            """Spin monkey image"""
            center = self.rect.center
            self.rotation += 12
            if self.rotation >= 360:
                self.rotation = 0
                self.move = self._walk
                self.image = self.image_original
            else:
                rotate = pg.transform.rotate
                self.image = rotate(self.image_original, self.rotation)

            self.rect = self.image.get_rect(center=center)

        def get_punched(self, obj):
            """Cause monkey to spin"""
            self.move = self._spin


The `Chimp` class is doing a little more work than the fist, but nothing
more complex. This class will move the chimp back and forth across the
screen.  When the monkey is punched, he will spin around to exciting effect.
This class is also derived from the base :class:`Sprite <pygame.sprite.Sprite>`
class, and is initialized the same as the fist. While initializing, the class
also makes a copy of the original `image` named `image_original`. Finally, it
sets its `move` attribute to `_walk` and then adds itself to the Game's
`allsprites` and `punchables` groups. 

The `update` function for the chimp effectively calls the method that's
currently assigned to the `move` attribute. The method assigned to the `move`
attribute represents either a walking or spinning state. These `_walk` and
`_spin` methods are prefixed with an underscore, following a standard python
idiom, which suggests these methods should only be used by the `Chimp` class.
We could go so far as to give them a double underscore, which would tell
python to really try to make them private methods, but we don't need such
protection. :)

The `_walk` method creates a new position for the monkey by moving the current
rect by a given offset. If this new position crosses outside the display area
of the screen, it reverses the `delta` attribute. It also mirrors the image
using the :func:`pygame.transform.flip` function. This is a crude effect that
makes the monkey look like he's turning the direction he is moving.

The `_spin` method is called when the monkey is currently "dizzy". The
`rotation` attribute is used to store the current amount of rotation. When the
monkey has rotated all the way around (360 degrees) it resets the monkey image
back to the original, non-rotated version. Before calling the
:func:`pygame.transform.rotate` function, you'll see the code makes a local
reference to the function simply named "rotate". There is no need to do that
for this example, it is just done here to keep the following line's length a
little shorter. Note that when calling the `rotate` function, we are always
rotating from the original monkey image. When rotating, there is a slight loss
of quality. Repeatedly rotating the same image and the quality would get worse
each time. Also, when rotating an image, the size of the image will actually
change. This is because the corners of the image will be rotated out, making
the image bigger. We make sure the center of the new image matches the center
of the old image, so it rotates without moving.

The last method is `get_punched()` which tells the sprite to enter its dizzy
state. This will cause the image to start spinning.


Initialize Everything
---------------------

Before we can do much with pygame, we need to make sure its modules
are initialized. In this case we will also open a simple graphics window.
Now we are in the `Game` class of the program, which actually runs everything. ::

    pg.init()
    Game.screen = pg.display.set_mode((1280, 480))
    Game.screen_rect = Game.screen.get_rect()
    pg.display.set_caption("Monkey Fever")
    pg.mouse.set_visible(False)

The first line to initialize *pygame* takes care of a bit of
work for us. It checks through the imported *pygame* modules and attempts
to initialize each one of them. It is possible to go back and check if modules
failed to initialize, but we won't bother here. It is also possible to
take a lot more control and initialize each specific module by hand. That
type of control is generally not needed, but is available if you desire.

Next we set up the display graphics mode. Note that the :mod:`pygame.display`
module is used to control all the display settings. In this case we are
asking for a 1280 by 480 window. We could add the ``SCALED`` display flag
which automatically scales up the window for displays much larger than the
window.

Last we set the window title and turn off the mouse cursor for our
window. Very basic to do, and now we have a small black window ready to
do our bidding. Usually the cursor defaults to visible, so there is no need
to really set the state unless we want to hide it.


Create The Background
---------------------

Our program is going to have text message in the background. It would
be nice for us to create a single surface to represent the background and
repeatedly use that. The first step is to create the surface. ::

    Game.background = pg.Surface(Game.screen.get_size())
    Game.background = Game.background.convert()
    Game.background.fill((170, 238, 187))

This creates a new surface for us that is the same size as the display
window. Note the extra call to `convert()` after creating the Surface. The
convert with no arguments will make sure our background is the same format
as the display window, which will give us the fastest results.

We also fill the entire background with a certain green color. The fill()
function usually takes an RGB triplet as arguments, but supports many
input formats. See the :mod:`pygame.Color` for all the color formats.


Put Text On The Background, Centered
------------------------------------

Now that we have a background surface, lets get the text rendered to it. We
only do this if we see the :mod:`pygame.font` module has imported properly.
If not, we just skip this section. ::

    if pg.font:
        font = pg.font.Font(None, 64)
        text = font.render("Pummel The Chimp, And Win $$$",
                           True,
                           (10, 10, 10))
        # Centered
        textpos = text.get_rect(centerx=Game.background.get_width() / 2,
                                y=10)
        Game.background.blit(text, textpos)
    else:
        print("Warning, fonts disabled")

As you see, there are a couple steps to getting this done. First we
must create the font object and render it into a new surface. We then find
the center of that new surface and blit (paste) it onto the background.

The font is created with the `font` module's `Font()` constructor. Usually
you will pass the name of a TrueType font file to this function, but we
can also pass `None`, which will use a default font. The `Font` constructor
also needs to know the size of font we want to create.

We then render that font into a new surface. The `render` function creates
a new surface that is the appropriate size for our text. In this case
we are also telling render to create antialiased text (for a nice smooth
look) and to use a dark grey color.

Next we need to find the centered position of the text on our display.
We create a "Rect" object from the text dimensions, which allows us to
easily assign it to the screen center.

Finally we blit (blit is like a copy or paste) the text onto the background
image.

Some pygame modules are optional, and if they aren't found, they evaluate to
``False``. Because of that, we decide to print a nice warning message if the
:mod:`font<pygame.font>` module in pygame is not available. (Although they
will only be unavailable in very uncommon situations).


Display The Background While Setup Finishes
-------------------------------------------

We still have a black window on the screen. Lets show our background
while we wait for the other resources to load. ::

    Game.screen.blit(Game.background, (0, 0))
    pg.display.flip()

This will blit our entire background onto the display window. The
blit is self explanatory, but what about this flip routine?

In pygame, changes to the display surface are not immediately visible.
Normally, a display must be updated in areas that have changed for them
to be visible to the user. In this case the `flip()` function works nicely
because it simply handles the entire window area.


Prepare Game Object
-------------------

Here we create all the objects that the game is going to need.

::

    Game.allsprites = pg.sprite.RenderPlain()
    Game.punchables = pg.sprite.Group()
    Game.chimp = Chimp()
    Game.fist = Fist()
    Game.clock = pg.time.Clock()

First we create a sprite :class:`Group <pygame.sprite.Group>` which will
contain all our sprites. We actually use a special sprite group named
:class:`RenderPlain<pygame.sprite.RenderPlain>`. This sprite group can draw
all the sprites it contains to the screen. It is called `RenderPlain` because
there are actually more advanced Render groups. But for our game, we just need
simple drawing.

Next we we create an instance of each of our sprite classes.  And lastly we
create a `clock` object to help control our game's framerate. we will use it
in the main loop of our game to make sure it doesn't run too fast.


Game Loop
---------

Nothing much here, just an infinite loop. ::

    while True:
        Game.clock.tick(60)

All games run in some sort of loop. The usual order of things is to
check on the state of the computer and user input, move and update the
state of all the objects, and then draw them to the screen. You'll see
that this example is no different.

We also make a call to our `clock` object, which will make sure our game
doesn't run faster than 60 frames per second.


Handle All Input Events
-----------------------

This is an extremely simple case of working the event queue. ::

    mouse_buttons = pg.event.get(
            eventtype=(pg.MOUSEBUTTONDOWN,
                       pg.MOUSEBUTTONUP))

    for event in pg.event.get():
        if event.type == pg.QUIT:
            pg.quit()
            sys.exit()
        elif (event.type == pg.KEYDOWN
              and event.key == pg.K_ESCAPE):
            pg.quit()
            sys.exit()

First we get any mouse button Events from pygame and save them. Then we loop
through each of the remaining events. The first two tests see if the user has
quit our game, or pressed the escape key. In these cases we just quit & exit,
allowing us out of the infinite loop.

Cleaning up the running game in *pygame* is extremely simple.
Since all variables are automatically destructed, we don't really have to do
anything, but calling `pg.quit()` explicitly cleans up pygame's internals.


Update the Sprites
------------------

::

    Game.allsprites.update(mouse_buttons)

Sprite groups have an `update()` method, which simply calls the update method
for all the sprites it contains. Each of the objects will move around, depending
on which state they are in. This is where the chimp will move one step side
to side, or spin a little farther if he was recently punched.


Draw The Entire Scene
---------------------

Now that all the objects are in the right place, time to draw them. ::

    Game.screen.blit(Game.background, (0, 0))
    Game.allsprites.draw(Game.screen)
    pg.display.flip()

The first blit call will draw the background onto the entire screen. This
erases everything we saw from the previous frame (slightly inefficient, but
good enough for this game). Next we call the `draw()` method of the sprite
container. Since this sprite container is really an instance of the "RenderPlain"
sprite group, it knows how to draw our sprites. Lastly, we `flip()` the contents
of pygame's software double buffer to the screen. This makes everything we've
drawn visible all at once.
