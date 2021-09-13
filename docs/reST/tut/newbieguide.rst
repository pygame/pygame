.. TUTORIAL: David Clark's Newbie Guide To Pygame

.. include:: common.txt

**************************
  Newbie Guide to Pygame
**************************

.. title:: A Newbie Guide to pygame


A Newbie Guide to pygame
========================

or **Things I learned by trial and error so you don't have to,**

or **How I learned to stop worrying and love the blit.**

Pygame_ is a python wrapper for SDL_, written by Pete Shinners.  What this
means is that, using pygame, you can write games or other multimedia
applications in Python that will run unaltered on any of SDL's supported
platforms (Windows, Unix, Mac, BeOS and others).

Pygame may be easy to learn, but the world of graphics programming can be
pretty confusing to the newcomer.  I wrote this to try to distill the practical
knowledge I've gained over the past year or so of working with pygame, and it's
predecessor, PySDL.  I've tried to rank these suggestions in order of
importance, but how relevant any particular hint is will depend on your own
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
sort a set of keys.  Resist the temptation to ask for direct help online when 
you run into trouble.  Instead, fire up the interpreter and play with the 
problem for a few hours, or use print statements and debugging tools to find out
what's going wrong in your code.  Get into the habit of looking things up in the 
official _Python Docs, and Googling error messages to figure out what they 
mean.

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
``surfarray`` or ``version`` functions. 


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
it was something I had to worry about. 'I only use PNGs, therefore everything I
do will be in the same format. So I don't need ``convert()``';. It turns out I
was very, very wrong.

The 'format' that ``convert()`` refers to isn't the *file* format (ie PNG,
JPEG, GIF), it's what's called the 'pixel format'.  This refers to the
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


Some advice you'll encounter is outdated, obsolete, or optional.
----------------------------------------------------------------

 **Dirty Rects**

When you read older bits of pygame documentation or guides online, you may see 
some emphasis on only updating portions of the screen that are dirty for the 
sake of performance (in this context, "dirty" means the region has changed since 
the previous frame was drawn).  

Generally this entails calling ``display.update(dirty_rects)`` instead of 
``display.flip()``, not having scrolling backgrounds, or even not clearing the 
screen every frame because otherwise pygame supposedly can't handle it.  Some of 
pygame's API is designed to support this paradigm as well (e.g. 
``pygame.sprite.RenderUpdates``), which made a lot of sense in the early 2000s 
when pygame was first released.

In the current year though, even modest computers are powerful enough to refresh 
the entire display once per frame at 60 FPS and beyond.  You can have a moving 
camera, or dynamic backgrounds and your game should run totally fine.  CPUs are 
more powerful nowadays, and you can use `display.flip()` without fear.

That being said though, there are still plenty of ways to accidentally tank your 
game's performance with poorly optimized rendering logic.  For example, even on 
modern hardware it's probably too slow to call ``set_at`` once per pixel on the 
display surface.  Being mindful of performance is still something you'll have to 
do.

 **HWSURFACE and DOUBLEBUF** 

These ``display.set_mode()`` flags do nothing in pygame 2.  There's no reason to 
use them anymore.

 ** The Sprite class**
 
You don't need to use the built-in Sprite or Group classes if you don't want to.  
If you watch a lot of tutorials it may seem like Sprite is the fundamental 
"GameObject" of pygame, from which all other objects must derive, but in reality 
it's pretty much just a wrapper around a rect and a surface, with some 
convenience methods.  You may find it more intuitive (and fun) to design your 
game's core classes from scratch.


There is NO rule six.
---------------------


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

So you've got your sprites moving around, and you need to know whether or not 
they're bumping into one another. It's tempting to write something like the 
following:

 * Check to see if the rects are in collision. If they aren't, ignore them.
 * For each pixel in the overlapping area, see if the corresponding pixels from both sprites are opaque. If so, there's a collision.

There are other ways to do this, with ANDing sprite masks and so on, but any
way you do it in pygame, it's probably going to be too slow. For most games,
it's probably better just to do 'sub-rect collision' -- create a rect for each
sprite that's a little smaller than the actual image, and use that for
collisions instead. It will be much faster, and in most cases the player won't
notice the imprecision.


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

    if key.get_pressed[K_t] and key.get_pressed[K_f]:
        print("Yup!")

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
However, ``poll()`` will consume 100% of available CPU time while it runs,
and it will fill the event queue with ``NOEVENTS``.  Use ``set_blocked()`` to
select just those event types you're interested in -- your queue will be much
more manageable.

Another note about the event queue -- even if you don't want to use it, you must 
still clear it periodically because it's still going to be filling up with events 
in the background as the user presses keys and mouses over the window. On Windows, 
if your game goes too long without clearing the queue, the operating system will 
think it has frozen and show a "The application is not responding" message. 
Iterating over ``event.get()`` or simply calling ``event.clear()`` once per frame 
will avoid this.


Colorkey vs. Alpha.
-------------------

There's a lot of confusion around these two techniques, and much of it comes from the terminology used.

'Colorkey blitting' involves telling pygame that all pixels of a certain color
in a certain image are transparent instead of whatever color they happen to be.
These transparent pixels are not blitted when the rest of the image is blitted,
and so don't obscure the background.  This is how we make sprites that aren't
rectangular in shape.  Simply call ``surface.set_colorkey(color)``, where
color is an RGB tuple -- say (0,0,0). This would make every pixel in the source
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
unwieldy, chances are its speed can be improved, too.  Read over `Why Pygame is 
Slow`_ for some deeper insight into why pygame might be considered slower than 
other frameworks/engines, and what that actually means in practice.  
And if you're truly stumped by performance problems, profilers like cProfile_ or 
SnakeViz_ can help identify bottlenecks (they'll tell you which parts of the 
code are taking the longest to execute). That said, premature optimisation is 
the root of all evil; if it's already fast enough, don't torture the code trying 
to make it faster.  If it's fast enough, let it be :)


There you go. Now you know practically everything I know about using pygame.
Now, go write that game!

----

*David Clark is an avid pygame user and the editor of the Pygame Code
Repository, a showcase for community-submitted python game code.  He is also
the author of Twitch, an entirely average pygame arcade game.*

.. _Pygame: https://www.pygame.org/
.. _SDL: http://libsdl.org
.. _Python Documentation: https://docs.python.org/3/
.. _SolarWolf: https://www.pygame.org/shredwheat/solarwolf/index.shtml
.. _Why Pygame is Slow: https://blubberquark.tumblr.com/post/630054903238262784/why-pygame-is-slow
.. _cProfile: https://docs.python.org/3/library/profile.html
.. _SnakeViz: https://jiffyclub.github.io/snakeviz/
