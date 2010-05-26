##    pygame - Python Game Library
##    Copyright (C) 2000-2003, 2007  Pete Shinners
##              (C) 2004 Joe Wreschnig
##    This library is free software; you can redistribute it and/or
##    modify it under the terms of the GNU Library General Public
##    License as published by the Free Software Foundation; either
##    version 2 of the License, or (at your option) any later version.
##
##    This library is distributed in the hope that it will be useful,
##    but WITHOUT ANY WARRANTY; without even the implied warranty of
##    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
##    Library General Public License for more details.
##
##    You should have received a copy of the GNU Library General Public
##    License along with this library; if not, write to the Free
##    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
##

"""
Basic game object classes.

This module contains several simple classes to be used within games. There
is the main Sprite class and several Group classes that contain Sprites.
The use of these classes is entirely optional when using Pygame. The classes
are fairly lightweight and only provide a starting place for the code
that is common to most games.

The Sprite class is intended to be used as a base class for the different
types of objects in the game. There is also a base Group class that simply
stores sprites. A game could create new types of Group classes that operate
on specially customized Sprite instances they contain.

The basic Sprite class can draw the Sprites it contains to a Surface. The
Group.draw() method requires that each Sprite have a Surface.image attribute
and a Surface.rect. The Group.clear() method requires these same attributes,
and can be used to erase all the Sprites with background. There are also
more advanced Groups: pygame2.sprite.RenderUpdates() and
pygame2.sprite.OrderedUpdates().

Lastly, this module contains several collision functions. These help find
sprites inside multiple groups that have intersecting bounding rectangles.
To find the collisions, the Sprites are required to have a Surface.rect
attribute assigned.

The groups are designed for high efficiency in removing and adding Sprites
to them. They also allow cheap testing to see if a Sprite already exists in
a Group. A given Sprite can exist in any number of groups. A game could use 
some groups to control object rendering, and a completely separate set of 
groups to control interaction or player movement. Instead of adding type 
attributes or bools to a derived Sprite class, consider keeping the 
Sprites inside organized Groups. This will allow for easier lookup later 
in the game.

Sprites and Groups manage their relationships with the add() and remove()
methods. These methods can accept a single or multiple targets for 
membership.  The default initializers for these classes also takes a 
single or list of targets for initial membership. It is safe to repeatedly 
add and remove the same Sprite from a Group.

While it is possible to design sprite and group classes that don't derive 
from the Sprite and AbstractGroup classes below, it is strongly recommended 
that you extend those when you add a Sprite or Group class.

Sprites are not thread safe.  So lock them yourself if using threads.
"""

import pygame2.compat
pygame2.compat.deprecation \
    ("The sprite package is deprecated and will change in future versions")

##todo
## a group that holds only the 'n' most recent elements.
## sort of like the GroupSingle class, but holding more
## than one sprite
##
## drawing groups that can 'automatically' store the area
## underneath, so the can "clear" without needing a background
## function. obviously a little slower than normal, but nice
## to use in many situations. (also remember it must "clear"
## in the reverse order that it draws :])
##
## the drawing groups should also be able to take a background
## function, instead of just a background surface. the function
## would take a surface and a rectangle on that surface to erase.
##
## perhaps more types of collision functions? the current two
## should handle just about every need, but perhaps more optimized
## specific ones that aren't quite so general but fit into common
## specialized cases.

from pygame2.sprite.collision import *
from pygame2.sprite.groups import *
from pygame2.sprite.layeredgroups import *
from pygame2.sprite.rendergroups import *
from pygame2.sprite.sprites import *
