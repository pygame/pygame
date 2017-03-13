.. TUTORIAL: Sprite Module Introduction

.. include:: common.txt

*************************************************
  Pygame Tutorials - Sprite Module Introduction
*************************************************


Sprite Module Introduction
==========================

.. rst-class:: docinfo

:Author: Pete Shinners
:Contact: pete@shinners.org
:Revision: 1.2, 2002-04-12 (Updated 2017-03-13)

Pygame version 1.3 comes with a new module, ``pygame.sprite``. This module is
written in Python and includes some higher-level classes to manage your game
objects. By using this module to its full potential, you can easily manage and
draw your game objects. The sprite classes are very optimized, so it's likely
your game will run faster with the sprite module than without.

The sprite module is also meant to be very generic. It turns out you can use it
with nearly any type of gameplay. All this flexibility comes with a slight
penalty, it needs a little understanding to properly use it. The
:mod:`reference documentation <pygame.sprite>` for the sprite module can keep
you running, but you'll probably need a bit more explanation of how to use
``pygame.sprite`` in your own game.

Several of the pygame examples (like "chimp" and "aliens") have been updated to
use the sprite module. You may want to look into those first to see what this
sprite module is all about. The chimp module even has it's own line-by-line
tutorial, which may help get more an understanding of programming with python
and pygame.

Note that this introduction will assume you have a bit of experience
programming with python, and are somewhat familiar with the different parts of
creating a simple game.  In this tutorial the word "reference" is occasionally
used.  This represents a python variable. Variables in python are references,
so you can have several variables all pointing to the same object.

History Lesson
--------------

The term "sprite" is a holdover from older computer and game machines.  These
older boxes were unable to draw and erase normal graphics fast enough for them
to work as games. These machines had special hardware to handle game like
objects that needed to animate very quickly. These objects were called
"sprites" and had special limitations, but could be drawn and updated very
fast. They usually existed in special overlay buffers in the video.  These days
computers have become generally fast enough to handle sprite like objects
without dedicated hardware. The term sprite is still used to represent just
about anything in a 2D game that is animated.

The Classes
-----------

The sprite module comes with two main classes. The first is :class:`Sprite
<pygame.sprite.Sprite>`, which should be used as a base class for all your game
objects. This class doesn't really do anything on its own, it just includes
several functions to help manage the game object. The other type of class is
:class:`Group <pygame.sprite.Group>`. The ``Group`` class is a container for
different ``Sprite`` objects. There are actually several different types of
group classes. Some of the ``Groups`` can draw all the elements they contain,
for example.

This is all there really is to it. We'll start with a description of what each
type of class does, and then discuss the proper ways to use these two classes.


The Sprite Class
----------------

As mentioned before, the Sprite class is designed to be a base class for all
your game objects. You cannot really use it on its own, as it only has several
methods to help it work with the different ``Group`` classes. The sprite keeps
track of which groups it belongs to.
The class constructor (``__init__`` method) takes an argument of a
``Group`` (or list of ``Groups``) the ``Sprite`` instance should belong to.
You can also change the ``Group`` membership for the ``Sprite`` with the
:meth:`add() <pygame.sprite.Sprite.add>` and
:meth:`remove() <pygame.sprite.Sprite.remove>` methods.
there is also a :meth:`groups() <pygame.sprite.Sprite.groups>` method,
which returns a list of the current groups containing the sprite.

When using the your Sprite classes it's best to think of them as "valid" or
"alive" when they are belonging to one or more ``Groups``. When you remove the
instance from all groups pygame will clean up the object. (Unless you have your
own references to the instance somewhere else.) The :meth:`kill()
<pygame.sprite.Sprite.kill>` method removes the sprite from all groups it
belongs to. This will cleanly delete the sprite object. If you've put some
little games together, you'll know sometimes cleanly deleting a game object can
be tricky. The sprite also comes with an :meth:`alive()
<pygame.sprite.Sprite.alive>` method, which returns true if it is still a
member of any groups.


The Group Class
---------------

The ``Group`` class is just a simple container. Similar to the sprite, it has
an :meth:`add() <pygame.sprite.Group.add>` and :meth:`remove()
<pygame.sprite.Group.remove>` method which can change which sprites belong to
the group. You also can pass a sprite or list of sprites to the constructor
(``__init__()`` method) to create a ``Group`` instance that contains some
initial sprites.


The ``Group`` has a few other methods like :meth:`empty()
<pygame.sprite.Group.empty>` to remove all sprites from the group and
:meth:`copy() <pygame.sprite.Group.copy>` which will return a copy of the group
with all the same members. Also the :meth:`has() <pygame.sprite.Group.has>`
method will quickly check if the ``Group`` contains a sprite or list of
sprites.

The other function you will use frequently is the :meth:`sprites()
<pygame.sprite.Group.sprites>` method. This returns an object that can be
looped on to access every sprite the group contains.  Currently this is just a
list of the sprites, but in later version of python this will likely use
iterators for better performance.

As a shortcut, the ``Group`` also has an :meth:`update()
<pygame.sprite.Group.update>` method, which will call an ``update()`` method on
every sprite in the group. Passing the same arguments to each one. Usually in a
game you need some function that updates the state of a game object. It's very
easy to call your own methods using the ``Group.sprites()`` method, but this is
a shortcut that's used enough to be included. Also note that the base
``Sprite`` class has a "dummy" ``update()`` method that takes any sort of
arguments and does nothing.

Lastly, the Group has a couple other methods that allow you to use it with
the builtin ``len()`` function, getting the number of sprites it contains, and
the "truth" operator, which allows you to do "if mygroup:" to check if the
group has any sprites.


Mixing Them Together
--------------------

At this point the two classes seem pretty basic. Not doing a lot more than you
can do with a simple list and your own class of game objects. But there are
some big advantages to using the ``Sprite`` and ``Group`` together. A sprite
can belong to as many groups as you want. Remember as soon as it belongs to no
groups, it will usually be cleared up (unless you have other "non-group"
references to that object).

The first big thing is a fast simple way to categorize sprites. For example,
say we had a Pacman-like game. We could make separate groups for the different
types of objects in the game. Ghosts, Pac, and Pellets. When Pac eats a power
pellet, we can change the state for all ghost objects by effecting everything
in the Ghost group. This is quicker and simpler than looping through a list
of all the game objects and checking which ones are ghosts.

Adding and removing groups and sprites from each other is a very fast
operation, quicker than using lists to store everything. Therefore you can very
efficiently change group memberships. Groups can be used to work like simple
attributes for each game object. Instead of tracking some attribute like
"close_to_player" for a bunch of enemy objects, you could add them to a
separate group. Then when you need to access all the enemies that are near the
player, you already have a list of them, instead of going through a list of all
the enemies, checking for the "close_to_player" flag. Later on your game could
add multiple players, and instead of adding more "close_to_player2",
"close_to_player3" attributes, you can easily add them to different groups or
each player.

Another important benefit of using the ``Sprites`` and ``Groups``, the groups
cleanly handle the deleting (or killing) of game objects. In a game where many
objects are referencing other objects, sometimes deleting an object can be the
hardest part, since it can't go away until it is not referenced by anyone. Say
we have an object that is "chasing" another object. The chaser can keep a
simple Group that references the object (or objects) it is chasing. If the
object being chased happens to be destroyed, we don't need to worry about
notifying the chaser to stop chasing. The chaser can see for itself that its
group is now empty, and perhaps find a new target.

Again, the thing to remember is that adding and removing sprites from groups is
a very cheap/fast operation. You may be best off by adding many groups to
contain and organize your game objects. Some could even be empty for large
portions of the game, there isn't any penalties for managing your game like
this.


The Many Group Types
--------------------

The above examples and reasons to use ``Sprites`` and ``Groups`` are only a tip
of the iceberg. Another advantage is that the sprite module comes with several
different types of ``Groups``. These groups all work just like a regular old
``Group``, but they also have added functionality (or slightly different
functionality).  Here's a list of the ``Group`` classes included with the
sprite module.

  :class:`Group <pygame.sprite.Group>`

    This is the standard "no frills" group mainly explained above. Most of the
    other ``Groups`` are derived from this one, but not all.

  :class:`GroupSingle <pygame.sprite.GroupSingle>`

    This works exactly like the regular ``Group`` class, but it only contains
    the most recently added sprite. Therefore when you add a sprite to this group,
    it "forgets" about any previous sprites it had. Therefore it always contains
    only one or zero sprites.

  :class:`RenderPlain <pygame.sprite.RenderPlain>`

    This is a standard group derived from ``Group``. It has a draw() method
    that draws all the sprites it contains to the screen (or any ``Surface``). For
    this to work, it requires all sprites it contains to have a "image" and "rect"
    attributes. It uses these to know what to blit, and where to blit it.

  :class:`RenderClear <pygame.sprite.RenderClear>`

    This is derived from the ``RenderPlain`` group, and adds a method named
    ``clear()``. This will erase the previous position of all drawn sprites. It
    uses a background image to fill in the areas where the sprite were. It is smart
    enough to handle deleted sprites and properly clear them from the screen when
    the ``clear()`` method is called.

  :class:`RenderUpdates <pygame.sprite.RenderUpdates>`

    This is the Cadillac of rendering ``Groups``. It is inherited from
    ``RenderClear``, but changes the ``draw()`` method to also return a list of
    pygame ``Rects``, which represent all the areas on screen that have been
    changed.

That is the list of different groups available We'll discuss more about these
rendering groups in the next section. There's nothing stopping you from
creating your own Group classes as well. They are just python code, so you can
inherit from one of these and add/change whatever you want. In the future I
hope we can add a couple more ``Groups`` to this list. A ``GroupMulti`` which
is like the ``GroupSingle``, but can hold up to a given number of sprites (in
some sort of circular buffer?). Also a super-render group that can clear the
position of the old sprites without needing a background image to do it (by
grabbing a copy of the screen before blitting). Who knows really, but in the
future we can add more useful classes to this list.


The Rendering Groups
--------------------

From above we can see there are three different rendering groups. We could
probably just get away with the ``RenderUpdates`` one, but it adds overhead not
really needed for something like a scrolling game. So we have a couple tools
here, pick the right one for the right job.

For a scrolling type game, where the background completely changes every frame.
We obviously don't need to worry about python's update rectangles in the call
to ``display.update()``. You should definitely go with the ``RenderPlain``
group here to manage your rendering.

For games where the background is more stationary, you definitely don't want
pygame updating the entire screen (since it doesn't need to). This type of game
usually involves erasing the old position of each object, then drawing it in a
new place for each frame. This way we are only changing what is necessary.
Most of the time you will just want to use the ``RenderUpdates`` class here.
Since you will also want to pass this list of changes to the
``display.update()`` function.

The ``RenderUpdates`` class also does a good job an minimizing overlapping
areas in the list of updated rectangles. If the previous position and current
position of an object overlap, it will merge them into a single rectangle.
Combine this with the fact that is properly handles deleted objects and this is
one powerful ``Group`` class. If you've written a game that manages the changed
rectangles for the objects in a game, you know this the cause for a lot of
messy code in your game. Especially once you start to throw in objects that can
be deleted at any time. All this work is reduced down to a ``clear()`` and
``draw()`` method with this monster class. Plus with the overlap checking, it
is likely faster than if you did it yourself.

Also note that there's nothing stopping you from mixing and matching these
render groups in your game. You should definitely use multiple rendering groups
when you want to do layering with your sprites. Also if the screen is split
into multiple sections, perhaps each section of the screen should use an
appropriate render group?


Collision Detection
-------------------

The sprite module also comes with two very generic collision detection
functions.  For more complex games, these really won't work for you, but you
can easily grab the source code for them, and modify them as needed.

Here's a summary of what they are, and what they do.

  :func:`spritecollide(sprite, group, dokill) -> list <pygame.sprite.spritecollide>`

    This checks for collisions between a single sprite and the sprites in a group.
    It requires a "rect" attribute for all the sprites used. It returns a list of
    all the sprites that overlap with the first sprite. The "dokill" argument is a
    boolean argument. If it is true, the function will call the ``kill()`` method
    on all the sprites. This means the last reference to each sprite is probably in
    the returned list. Once the list goes away so do the sprites.  A quick example
    of using this in a loop ::

      >>> for bomb in sprite.spritecollide(player, bombs, 1):
      ...     boom_sound.play()
      ...     Explosion(bomb, 0)

    This finds all the sprites in the "bomb" group that collide with the player.
    Because of the "dokill" argument it deletes all the crashed bombs. For each
    bomb that did collide, it plays a "boom" sound effect, and creates a new
    ``Explosion`` where the bomb was. (Note, the ``Explosion`` class here knows to
    add each instance to the appropriate class, so we don't need to store it in a
    variable, that last line might feel a little "funny" to you python programmers.

  :func:`groupcollide(group1, group2, dokill1, dokill2) -> dictionary <pygame.sprite.groupcollide>`

    This is similar to the ``spritecollide`` function, but a little more complex.
    It checks for collisions for all the sprites in one group, to the sprites in
    another. There is a ``dokill`` argument for the sprites in each list. When
    ``dokill1`` is true, the colliding sprites in ``group1`` will be ``kill()``ed.
    When ``dokill2`` is true, we get the same results for ``group2``. The
    dictionary it returns works like this; each key in the dictionary is a sprite
    from ``group1`` that had a collision.  The value for that key is a list of the
    sprites that it collided with. Perhaps another quick code sample explains it
    best ::

      >>> for alien in sprite.groupcollide(aliens, shots, 1, 1).keys()
      ...     boom_sound.play()
      ...     Explosion(alien, 0)
      ...     kills += 1

    This code checks for the collisions between player bullets and all the aliens
    they might intersect. In this case we only loop over the dictionary keys, but
    we could loop over the ``values()`` or ``items()`` if we wanted to do something
    to the specific shots that collided with aliens. If we did loop over the
    ``values()`` we would be looping through lists that contain sprites. The same
    sprite may even appear more than once in these different loops, since the same
    "shot" could have collided against multiple "aliens".

Those are the basic collision functions that come with pygame. It should be
easy to roll your own that perhaps use something different than the "rect"
attribute. Or maybe try to fine-tweak your code a little more by directly
effecting the collision object, instead of building a list of the collision?
The code in the sprite collision functions is very optimized, but you could
speed it up slightly by taking out some functionality you don't need.


Common Problems
---------------

Currently there is one main problem that catches new users. When you derive
your new sprite class with the Sprite base, you **must** call the
``Sprite.__init__()`` method from your own class ``__init__()`` method.  If you
forget to call the ``Sprite.__init__()`` method, you get a cryptic error, like
this ::

    AttributeError: 'mysprite' instance has no attribute '_Sprite__g'


Extending Your Own Classes *(Advanced)*
---------------------------------------

Because of speed concerns, the current ``Group`` classes try to only do exactly
what they need, and not handle a lot of general situations. If you decide you
need extra features, you may want to create your own ``Group`` class.

The ``Sprite`` and ``Group`` classes were designed to be extended, so feel free
to create your own ``Group`` classes to do specialized things. The best place
to start is probably the actual python source code for the sprite module.
Looking at the current ``Sprite`` groups should be enough example on how to
create your own.

For example, here is the source code for a rendering ``Group`` that calls a
``render()`` method for each sprite, instead of just blitting an "image"
variable from it.  Since we want it to also handle updated areas, we will start
with a copy of the original ``RenderUpdates`` group, here is the code::

    class RenderUpdatesDraw(RenderClear):
        """call sprite.draw(screen) to render sprites"""
        def draw(self, surface):
            dirty = self.lostsprites
            self.lostsprites = []
            for s, r in self.spritedict.items():
                newrect = s.draw(screen) #Here's the big change
                if r is 0:
                    dirty.append(newrect)
                else:
                    dirty.append(newrect.union(r))
                self.spritedict[s] = newrect
            return dirty

Following is more information on how you could create your own ``Sprite`` and
``Group`` objects from scratch.

The ``Sprite`` objects only "require" two methods. "add_internal()" and
"remove_internal()".  These are called by the ``Group`` classes when they are
removing a sprite from themselves. The ``add_internal()`` and
``remove_internal()`` have a single argument which is a group. Your ``Sprite``
will need some way to also keep track of the ``Groups`` it belongs to. You will
likely want to try to match the other methods and arguments to the real
``Sprite`` class, but if you're not going to use those methods, you sure don't
need them.

It is almost the same requirements for creating your own ``Group``. In fact, if
you look at the source you'll see the ``GroupSingle`` isn't derived from the
``Group`` class, it just implements the same methods so you can't really tell
the difference. Again you need an "add_internal()" and "remove_internal()"
method that the sprites call when they want to belong or remove themselves from
the group. The ``add_internal()`` and ``remove_internal()`` have a single
argument which is a sprite. The only other requirement for the ``Group``
classes is they have a dummy attribute named "_spritegroup". It doesn't matter
what the value is, as long as the attribute is present. The Sprite classes can
look for this attribute to determine the difference between a "group" and any
ordinary python container. (This is important, because several sprite methods
can take an argument of a single group, or a sequence of groups. Since they
both look similar, this is the most flexible way to "see" the difference.)

You should through the code for the sprite module. While the code is a bit
"tuned", it's got enough comments to help you follow along.  There's even a
TODO section in the source if you feel like contributing.

