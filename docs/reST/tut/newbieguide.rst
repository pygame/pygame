.. TUTORIAL:David Clark's Newbie Guide To Pygame

A Newbie Guide to pygame
************************

**or**

Things I learned by trial and error so you don't have to.
=========================================================

**or**

How I learned to stop worrying and love the blit.
=================================================


Pygame_ is a python wrapper for SDL_, written by Pete Shinners.  What this
means is that, using pygame, you can write games or other multimedia
applications in Python that will run unaltered on any of SDL's supported
platforms (Windows, Unix, Mac, beOS and others).

Pygame may be easy to learn, but the world of graphics programming can be
pretty confusing to the newcomer.  I wrote this to try to distill the practical
knowledge I've gained over the past year or so of working with pygame, and it's
predecessor, pySDL.  I've tried to rank these suggestions in order of
importance, but how relevent any particular hint is will depend on your own
background and the details of your project.


Get comfortable working in Python.
----------------------------------

The most important thing is to feel confident using python. Learning something
as potentially complicated as graphics programming will be a real chore if
you're also unfamiliar with the language you're using. Write a few sizable
non-graphical programs in python -- parse some text files, write a guessing
game or a journal-entry program or something. Get comfortable with string and
list manipulation -- know how to split, slice and combine strings and lists.
Know how ``import`` works -- try writing a program that is spread across
several source files.  Write your own functions, and practice manipulating
numbers and characters; know how to convert between the two.  Get to the point
where the syntax for using lists and dictionaries is second-nature -- you don't
want to have to run to the documentation every time you need to slice a list or
sort a set of keys.  Resist the temptation to run to a mailing list,
comp.lang.python, or irc when you run into trouble.  Instead, fire up the
interpreter and play with the problem for a few hours.  Print out the `Python
2.0 Quick Reference`_ and keep it by your computer.

This may sound incredibly dull, but the confidence you'll gain through your
familiarity with python will work wonders when it comes time to write your
game.  The time you spend making python code second-nature will be nothing
compared to the time you'll save when you're writing real code.


Recognize which parts of pygame you really need.
------------------------------------------------

Looking at the jumble of classes at the top of the pygame Documentation index
may be confusing.  The important thing is to realize that you can do a great
deal with only a tiny subset of functions.  Many classes you'll probably never
use -- in a year, I haven't touched the ``Channel``, ``Joystick``, ``cursors``,
``Userrect``, ``surfarray`` or ``version`` functions.


Know what a surface is.
-----------------------

The most important part of pygame is the surface.  Just think of a surface as a
blank piece of paper.  You can do a lot of things with a surface -- you can
draw lines on it, fill parts of it with color, copy images to and from it, and
set or read individual pixel colors on it.  A surface can be any size (within
reason) and you can have as many of them as you like (again, within reason).
One surface is special -- the one you create with
``pygame.display.set_mode()``.  This 'display surface' represents the screen;
whatever you do to it will appear on the user's screen.  You can only have one
of these -- that's an SDL limitation, not a pygame one.

So how do you create surfaces?  As mentioned above, you create the special
'display surface' with ``pygame.display.set_mode()``.  You can create a surface
that contains an image by using ``image.load()``, or you can make a surface
that contains text with ``font.render()``.  You can even create a surface that
contains nothing at all with ``Surface()``.

Most of the surface functions are not critical. Just learn ``blit()``,
``fill()``, ``set_at()`` and ``get_at()``, and you'll be fine.


Use surface.convert().
----------------------

When I first read the documentation for ``surface.convert()``, I didn't think
it was something I had to worry about. 'I only use pngs, therefore everything I
do will be in the same format. So I don't need ``convert()``';. It turns out I
was very, very wrong.

The 'format' that ``convert()`` refers to isn't the *file* format (ie png,
jpeg, gif), it's what's called the 'pixel format'.  This refers to the
particular way that a surface records individual colors in a specific pixel.
If the surface format isn't the same as the display format, SDL will have to
convert it on-the-fly for every blit -- a fairly time-consuming process.  Don't
worry too much about the explanation; just note that ``convert()`` is necessary
if you want to get any kind of speed out of your blits.

How do you use convert? Just call it after creating a surface with the
``image.load()`` function. Instead of just doing::

    surface = pygame.image.load('foo.png')

Do::

    surface = pygame.image.load('foo.png').convert()

It's that easy. You just need to call it once per surface, when you load an
image off the disk.  You'll be pleased with the results; I see about a 6x
increase in blitting speed by calling ``convert()``.

The only times you don't want to use ``convert()`` is when you really need to
have absolute control over an image's internal format -- say you were writing
an image conversion program or something, and you needed to ensure that the
output file had the same pixel format as the input file.  If you're writing a
game, you need speed.  Use ``convert()``.


Dirty rect animation.
---------------------

The most common cause of inadequate frame rates in pygame programs results from
misunderstanding the ``pygame.display.update()`` function.  With pygame, merely
drawing something to the display surface doesn't cause it to appear on the
screen -- you need to call ``pygame.display.update()``.  There are three ways
of calling this function:


 * ``pygame.display.update()`` -- This updates the whole window (or the whole screen for fullscreen displays).
 * ``pygame.display.flip()`` -- This does the same thing, and will also do the right thing if you're using ``doublebuffered`` hardware acceleration, which you're not, so on to...
 * ``pygame.display.update(a rectangle or some list of rectangles)`` -- This updates just the rectangular areas of the screen you specify.


Most people new to graphics programming use the first option -- they update the
whole screen every frame.  The problem is that this is unacceptably slow for
most people.  Calling ``update()`` takes 35 milliseconds on my machine, which
doesn't sound like much, until you realize that 1000 / 35 = 28 frames per
second *maximum*. And that's with no game logic, no blits, no input, no AI,
nothing.  I'm just sitting there updating the screen, and 28 fps is my maximum
framerate. Ugh.

The solution is called 'dirty rect animation'.  Instead of updating the whole
screen every frame, only the parts that changed since the last frame are
updated.  I do this by keeping track of those rectangles in a list, then
calling ``update(the_dirty_rectangles)`` at the end of the frame.  In detail
for a moving sprite, I:

 * Blit a piece of the background over the sprite's current location, erasing it.
 * Append the sprite's current location rectangle to a list called dirty_rects.
 * Move the sprite.
 * Draw the sprite at it's new location.
 * Append the sprite's new location to my dirty_rects list.
 * Call ``display.update(dirty_rects)``

The difference in speed is astonishing. Consider that Solarwolf_ has dozens of
constantly moving sprites updating smoothly, and still has enough time left
over to display a parallax starfield in the background, and update that too.

There are two cases where this technique just won't work. The first is where
the whole window or screen really is being updated every frame -- think of a
smooth-scrolling engine like an overhead real-time strategy game or a
side-scroller.  So what do you do in this case?  Well, the short answer is --
don't write this kind of game in pygame.  The long answer is to scroll in steps
of several pixels at a time; don't try to make scrolling perfectly smooth.
Your player will appreciate a game that scrolls quickly, and won't notice the
background jumping along too much.

A final note -- not every game requires high framerates. A strategic wargame
could easily get by on just a few updates per second -- in this case, the added
complexity of dirty rect animation may not be necessary.


There is NO rule six.
---------------------


Hardware surfaces are more trouble than they're worth.
------------------------------------------------------

If you've been looking at the various flags you can use with
``pygame.display.set_mode()``, you may have thought like this: `Hey,
HWSURFACE! Well, I want that -- who doesn't like hardware acceleration. Ooo...
DOUBLEBUF; well, that sounds fast, I guess I want that too!`.  It's not
your fault; we've been trained by years of 3-d gaming to believe that hardware
acceleration is good, and software rendering is slow.

Unfortunately, hardware rendering comes with a long list of drawbacks:

 * It only works on some platforms. Windows machines can usually get hardware surfaces if you ask for them. Most other platforms can't. Linux, for example, may be able to provide a hardware surface if X4 is installed, if DGA2 is working properly, and if the moons are aligned correctly. If a hardware surface is unavailable, SDL will silently give you a software surface instead.

 * It only works fullscreen.

 * It complicates per-pixel access.  If you have a hardware surface, you need to Lock the surface before writing or reading individual pixel values on it.  If you don't, Bad Things Happen. Then you need to quickly Unlock the surface again, before the OS gets all confused and starts to panic.  Most of this process is automated for you in pygame, but it's something else to take into account.

 * You lose the mouse pointer. If you specify ``HWSURFACE`` (and actually get it), your pointer will usually just vanish (or worse, hang around in a half-there, half-not flickery state).  You'll need to create a sprite to act as a manual mouse pointer, and you'll need to worry about pointer acceleration and sensitivity. What a pain.

 * It might be slower anyway. Many drivers are not accelerated for the types of drawing that we do, and since everything has to be blitted across the video bus (unless you can cram your source surface into video memory as well), it might end up being slower than software access anyway.

Hardware rendering has it's place. It works pretty reliably under Windows, so
if you're not interested in cross-platform performance, it may provide you with
a substantial speed increase.  However, it comes at a cost -- increased
headaches and complexity.  It's best to stick with good old reliable
``SWSURFACE`` until you're sure you know what you're doing.


Don't get distracted by side issues.
------------------------------------

Sometimes, new game programmers spend too much time worrying about issues that
aren't really critical to their game's success.  The desire to get secondary
issues 'right' is understandable, but early in the process of creating a game,
you cannot even know what the important questions are, let alone what answers
you should choose.  The result can be a lot of needless prevarication.

For example, consider the question of how to organize your graphics files.
Should each frame have its own graphics file, or each sprite?  Perhaps all the
graphics should be zipped up into one archive?  A great deal of time has been
wasted on a lot of projects, asking these questions on mailing lists, debating
the answers, profiling, etc, etc.  This is a secondary issue; any time spent
discussing it should have been spent coding the actual game.

The insight here is that it is far better to have a 'pretty good' solution that
was actually implemented, than a perfect solution that you never got around to
writing.


Rects are your friends.
-----------------------

Pete Shinners' wrapper may have cool alpha effects and fast blitting speeds,
but I have to admit my favorite part of pygame is the lowly ``Rect`` class.  A
rect is simply a rectangle -- defined only by the position of its top left
corner, its width, and its height.  Many pygame functions take rects as
arguments, and they also take 'rectstyles', a sequence that has the same values
as a rect. So if I need a rectangle that defines the area between 10, 20 and
40, 50, I can do any of the following::

    rect = pygame.Rect(10, 20, 30, 30)
    rect = pygame.Rect((10, 20, 30, 30))
    rect = pygame.Rect((10, 20), (30, 30))
    rect = (10, 20, 30, 30)
    rect = ((10, 20, 30, 30))

If you use any of the first three versions, however, you get access to Rect's
utility functions.  These include functions to move, shrink and inflate rects,
find the union of two rects, and a variety of collision-detection functions.

For example, suppose I'd like to get a list of all the sprites that contain a
point (x, y) -- maybe the player clicked there, or maybe that's the current
location of a bullet. It's simple if each sprite has a .rect member -- I just
do::

    sprites_clicked = [sprite for sprite in all_my_sprites_list if sprite.rect.collidepoint(x, y)]

Rects have no other relation to surfaces or graphics functions, other than the
fact that you can use them as arguments.  You can also use them in places that
have nothing to do with graphics, but still need to be defined as rectangles.
Every project I discover a few new places to use rects where I never thought
I'd need them.


Don't bother with pixel-perfect collision detection.
----------------------------------------------------

So you've got your sprites moving around, and you need to know whether or not they're bumping into one another. It's tempting to write something like the following:

 * Check to see if the rects are in collision. If they aren't, ignore them.
 * For each pixel in the overlapping area, see if the corresponding pixels from both sprites are opaque. If so, there's a collision.

There are other ways to do this, with ANDing sprite masks and so on, but any
way you do it in pygame, it's probably going to be too slow. For most games,
it's probably better just to do 'sub-rect collision' -- create a rect for each
sprite that's a little smaller than the actual image, and use that for
collisions instead. It will be much faster, and in most cases the player won't
notice the inprecision.


Managing the event subsystem.
-----------------------------

Pygame's event system is kind of tricky.  There are actually two different ways
to find out what an input device (keyboard, mouse or joystick) is doing.

The first is by directly checking the state of the device.  You do this by
calling, say, ``pygame.mouse.get_pos()`` or ``pygame.key.get_pressed()``.
This will tell you the state of that device *at the moment you call the
function.*

The second method uses the SDL event queue.  This queue is a list of events --
events are added to the list as they're detected, and they're deleted from the
queue as they're read off.

There are advantages and disadvantages to each system.  State-checking (system
1) gives you precision -- you know exactly when a given input was made -- if
``mouse.get_pressed([0])`` is 1, that means that the left mouse button is
down *right at this moment*.  The event queue merely reports that the
mouse was down at some time in the past; if you check the queue fairly often,
that can be ok, but if you're delayed from checking it by other code, input
latency can grow.  Another advantage of the state-checking system is that it
detects "chording" easily; that is, several states at the same time.  If you
want to know whether the ``t`` and ``f`` keys are down at the same time, just
check::

    if (key.get_pressed[K_t] and key.get_pressed[K_f]):
        print "Yup!"

In the queue system, however, each keypress arrives in the queue as a
completely separate event, so you'd need to remember that the ``t`` key was
down, and hadn't come up yet, while checking for the ``f`` key.  A little more
complicated.

The state system has one great weakness, however. It only reports what the
state of the device is at the moment it's called; if the user hits a mouse
button then releases it just before a call to ``mouse.get_pressed()``, the
mouse button will return 0 -- ``get_pressed()`` missed the mouse button press
completely.  The two events, ``MOUSEBUTTONDOWN`` and ``MOUSEBUTTONUP``, will
still be sitting in the event queue, however, waiting to be retrieved and
processed.

The lesson is: choose the system that meets your requirements.  If you don't
have much going on in your loop -- say you're just sitting in a ``while 1``
loop, waiting for input, use ``get_pressed()`` or another state function; the
latency will be lower.  On the other hand, if every keypress is crucial, but
latency isn't as important -- say your user is typing something in an editbox,
use the event queue.  Some keypresses may be slightly late, but at least you'll
get them all.

A note about ``event.poll()`` vs. ``wait()`` -- ``poll()`` may seem better,
since it doesn't block your program from doing anything while it's waiting for
input -- ``wait()`` suspends the program until an event is received.
However, ``poll()`` will consume 100% of available cpu time while it runs,
and it will fill the event queue with ``NOEVENTS``.  Use ``set_blocked()`` to
select just those event types you're interested in -- your queue will be much
more manageable.


Colorkey vs. Alpha.
-------------------

There's a lot of confusion around these two techniques, and much of it comes from the terminology used.

'Colorkey blitting' involves telling pygame that all pixels of a certain color
in a certain image are transparent instead of whatever color they happen to be.
These transparent pixels are not blitted when the rest of the image is blitted,
and so don't obscure the background.  This is how we make sprites that aren't
rectangular in shape.  Simply call ``surface.set_colorkey(color)``, where
color is a rgb tuple -- say (0,0,0). This would make every pixel in the source
image transparent instead of black.

'Alpha' is different, and it comes in two flavors. 'Image alpha' applies to the
whole image, and is probably what you want.  Properly known as 'translucency',
alpha causes each pixel in the source image to be only *partially* opaque.
For example, if you set a surface's alpha to 192 and then blitted it onto a
background, 3/4 of each pixel's color would come from the source image, and 1/4
from the background.  Alpha is measured from 255 to 0, where 0 is completely
transparent, and 255 is completely opaque.  Note that colorkey and alpha
blitting can be combined -- this produces an image that is fully transparent in
some spots, and semi-transparent in others.

'Per-pixel alpha' is the other flavor of alpha, and it's more complicated.
Basically, each pixel in the source image has its own alpha value, from 0 to
255.  Each pixel, therefore, can have a different opacity when blitted onto a
background.  This type of alpha can't be mixed with colorkey blitting,
and it overrides per-image alpha.  Per-pixel alpha is rarely used in
games, and to use it you have to save your source image in a graphic
editor with a special *alpha channel*.  It's complicated -- don't use it
yet.


Do things the pythony way.
--------------------------

A final note (this isn't the least important one; it just comes at the end).
Pygame is a pretty lightweight wrapper around SDL, which is in turn a pretty
lightweight wrapper around your native OS graphics calls.  Chances are pretty
good that if your code is still slow, and you've done the things I've mentioned
above, then the problem lies in the way you're addressing your data in python.
Certain idioms are just going to be slow in python no matter what you do.
Luckily, python is a very clear language -- if a piece of code looks awkward or
unweildy, chances are its speed can be improved, too.  Read over `Python
Performance Tips`_ for some great advice on how you can improve the speed of
your code.  That said, premature optimisation is the root of all evil; if it's
just not fast enough, don't torture the code trying to make it faster.  Some
things are just not meant to be :)


There you go. Now you know practically everything I know about using pygame.
Now, go write that game!

----

David Clark is an avid pygame user and the editor of the Pygame Code
Repository, a showcase for community-submitted python game code.  He is also
the author of Twitch, an entirely average pygame arcade game.

.. _Pygame: http://www.pygame.org/
.. _SDL: http://libsdl.org
.. _Python 2.0 Quick Reference: http://www.brunningonline.net/simon/python/quick-ref2_0.html
.. _Solarwolf: http://shredwheat.zopesite.com/solarwolf
.. _Python Performance Tips: http://musi-cal.mojam.com/~skip/python/fastpython.html
