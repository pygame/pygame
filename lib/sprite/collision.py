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

# some different collision detection functions that could be used.

"""Collision handling for sprite objects"""

import pygame2.compat
pygame2.compat.deprecation \
    ("The sprite package is deprecated and will change in future versions")

try:
    from pygame2.mask import from_surface
except:
    pass

def collide_rect(left, right):
    """collision detection between two sprites, using rects.
    pygame2.sprite.collide_rect(left, right): return bool

    Tests for collision between two sprites. Uses the
    pygame rect colliderect function to calculate the
    collision. Intended to be passed as a collided
    callback function to the *collide functions.
    Sprites must have a "rect" attributes.

    New in pygame 1.8.0
    """
    return left.rect.colliderect(right.rect)

class collide_rect_ratio (object):
    """A callable class that checks for collisions between
    two sprites, using a scaled version of the sprites
    rects.

    Is created with a ratio, the instance is then intended
    to be passed as a collided callback function to the
    *collide functions.

    New in pygame 1.8.1
    """
    
    def __init__( self, ratio ):
        """Creates a new collide_rect_ratio callable. ratio is
        expected to be a floating point value used to scale
        the underlying sprite rect before checking for
        collisions.
        """

        self.ratio = ratio

    def __call__( self, left, right ):
        """pygame2.sprite.collide_rect_ratio(ratio)(left, right): bool
        collision detection between two sprites, using scaled rects.

        Tests for collision between two sprites. Uses the
        pygame rect colliderect function to calculate the
        collision, after scaling the rects by the stored ratio.
        Sprites must have a "rect" attributes.
        """

        ratio = self.ratio
        
        leftrect = left.rect
        width = leftrect.width
        height = leftrect.height
        leftrect = leftrect.inflate(width * ratio - width,
                                    height * ratio - height)
        
        rightrect = right.rect
        width = rightrect.width
        height = rightrect.height
        rightrect = rightrect.inflate(width * ratio - width,
                                      height * ratio - height)
        
        return leftrect.colliderect( rightrect )

def collide_circle( left, right ):
    """collision detection between two sprites, using circles.
    pygame2.sprite.collide_circle(left, right): return bool

    Tests for collision between two sprites, by testing to
    see if two circles centered on the sprites overlap. If
    the sprites have a "radius" attribute, that is used to
    create the circle, otherwise a circle is created that
    is big enough to completely enclose the sprites rect as
    given by the "rect" attribute. Intended to be passed as
    a collided callback function to the *collide functions.
    Sprites must have a "rect" and an optional "radius"
    attribute.

    New in pygame 1.8.0
    """

    xdistance = left.rect.centerx - right.rect.centerx
    ydistance = left.rect.centery - right.rect.centery
    distancesquared = xdistance ** 2 + ydistance ** 2
    try:
        leftradiussquared = left.radius ** 2
    except AttributeError:
        leftrect = left.rect
        leftradiussquared = ( leftrect.width ** 2 + leftrect.height ** 2 ) / 4
    try:
        rightradiussquared = right.radius ** 2
    except AttributeError:
        rightrect = right.rect
        rightradiussquared = (rightrect.width ** 2 + rightrect.height ** 2) / 4
    return distancesquared < leftradiussquared + rightradiussquared

class collide_circle_ratio( object ):
    """A callable class that checks for collisions between
    two sprites, using a scaled version of the sprites radius.

    Is created with a ratio, the instance is then intended
    to be passed as a collided callback function to the
    *collide functions.

    New in pygame 1.8.1
    """
    
    def __init__( self, ratio ):
        """Creates a new collide_circle_ratio callable. ratio is
        expected to be a floating point value used to scale
        the underlying sprite radius before checking for
        collisions.
        """
        self.ratio = ratio
        # Constant value that folds in division for diameter to radius,
        # when calculating from a rect.
        self.halfratio = ratio ** 2 / 4.0

    def __call__( self, left, right ):
        """pygame2.sprite.collide_circle_radio(ratio)(left, right): return bool
        collision detection between two sprites, using scaled circles.

        Tests for collision between two sprites, by testing to
        see if two circles centered on the sprites overlap, after
        scaling the circles radius by the stored ratio. If
        the sprites have a "radius" attribute, that is used to
        create the circle, otherwise a circle is created that
        is big enough to completely enclose the sprites rect as
        given by the "rect" attribute. Intended to be passed as
        a collided callback function to the *collide functions.
        Sprites must have a "rect" and an optional "radius"
        attribute.
        """

        ratio = self.ratio
        xdistance = left.rect.centerx - right.rect.centerx
        ydistance = left.rect.centery - right.rect.centery
        distancesquared = xdistance ** 2 + ydistance ** 2
        # Optimize for not containing radius attribute, as if radius was
        # set consistently, would probably be using collide_circle instead.
        if hasattr( left, "radius" ):
            leftradiussquared = (left.radius * ratio) ** 2
            
            if hasattr( right, "radius" ):
                rightradiussquared = (right.radius * ratio) ** 2
            else:
                halfratio = self.halfratio
                rightrect = right.rect
                rightradiussquared = \
                    (rightrect.width ** 2 + rightrect.height ** 2) * halfratio
        else:
            halfratio = self.halfratio
            leftrect = left.rect
            leftradiussquared = \
                (leftrect.width ** 2 + leftrect.height ** 2) * halfratio
            
            if hasattr( right, "radius" ):
                rightradiussquared = (right.radius * ratio) ** 2
            else:
                rightrect = right.rect
                rightradiussquared = \
                    (rightrect.width ** 2 + rightrect.height ** 2) * halfratio
        return distancesquared < leftradiussquared + rightradiussquared

def collide_mask(left, right):
    """collision detection between two sprites, using masks.
    pygame2.sprite.collide_mask(SpriteLeft, SpriteRight): bool

    Tests for collision between two sprites, by testing if
    thier bitmasks overlap. If the sprites have a "mask"
    attribute, that is used as the mask, otherwise a mask is
    created from the sprite image. Intended to be passed as
    a collided callback function to the *collide functions.
    Sprites must have a "rect" and an optional "mask"
    attribute.

    New in pygame 1.8.0
    """
    xoffset = right.rect[0] - left.rect[0]
    yoffset = right.rect[1] - left.rect[1]
    try:
        leftmask = left.mask
    except AttributeError:
        leftmask = from_surface(left.image)
    try:
        rightmask = right.mask
    except AttributeError:
        rightmask = from_surface(right.image)
    return leftmask.overlap(rightmask, (xoffset, yoffset))

def spritecollide(sprite, group, dokill, collided = None):
    """find Sprites in a Group that intersect another Sprite
    pygame2.sprite.spritecollide(sprite, group, dokill, collided = None): return Sprite_list

    Return a list containing all Sprites in a Group that intersect with another
    Sprite. Intersection is determined by comparing the Sprite.rect attribute
    of each Sprite.

    The dokill argument is a bool. If set to True, all Sprites that collide
    will be removed from the Group.

    The collided argument is a callback function used to calculate if
    two sprites are colliding. it should take two sprites as values, and
    return a bool value indicating if they are colliding. If collided is
    not passed, all sprites must have a "rect" value, which is a
    rectangle of the sprite area, which will be used to calculate the
    collision.
    """
    crashed = []
    if collided is None:
        # Special case old behaviour for speed.
        collide = sprite.rect.colliderect
        if dokill:
            for s in group.sprites():
                if collide(s.rect):
                    s.kill()
                    crashed.append(s)
        else:
            for s in group:
                if collide(s.rect):
                    crashed.append(s)
    else:
        if dokill:
            for s in group.sprites():
                if collided(sprite, s):
                    s.kill()
                    crashed.append(s)
        else:
            for s in group:
                if collided(sprite, s):
                    crashed.append(s)
    return crashed

def groupcollide(groupa, groupb, dokilla, dokillb, collided = None):
    """pygame2.sprite.groupcollide(groupa, groupb, dokilla, dokillb) -> dict
       collision detection between group and group

       given two groups, this will find the intersections
       between all sprites in each group. it returns a
       dictionary of all sprites in the first group that
       collide. the value for each item in the dictionary
       is a list of the sprites in the second group it
       collides with. the two dokill arguments control if
       the sprites from either group will be automatically
       removed from all groups.
       collided is a callback function used to calculate if
       two sprites are colliding. it should take two sprites
       as values, and return a bool value indicating if
       they are colliding. if collided is not passed, all
       sprites must have a "rect" value, which is a
       rectangle of the sprite area, which will be used
       to calculate the collision."""
    crashed = {}
    SC = spritecollide
    if dokilla:
        for s in groupa.sprites():
            c = SC(s, groupb, dokillb, collided)
            if c:
                crashed[s] = c
                s.kill()
    else:
        for s in groupa:
            c = SC(s, groupb, dokillb, collided)
            if c:
                crashed[s] = c
    return crashed

def spritecollideany(sprite, group, collided = None):
    """pygame2.sprite.spritecollideany(sprite, group) -> sprite
       finds any sprites that collide

       given a sprite and a group of sprites, this will
       return return any single sprite that collides with
       with the given sprite. If there are no collisions
       this returns None.

       if you don't need all the features of the
       spritecollide function, this function will be a
       bit quicker.

       collided is a callback function used to calculate if
       two sprites are colliding. it should take two sprites
       as values, and return a bool value indicating if
       they are colliding. if collided is not passed, all
       sprites must have a "rect" value, which is a
       rectangle of the sprite area, which will be used
       to calculate the collision."""
    if collided is None:
        # Special case old behaviour for speed.
        collide = sprite.rect.colliderect
        for s in group:
            if collide(s.rect):
                return s
    else:
        for s in group:
            if collided(sprite, s):
                return s
    return None
