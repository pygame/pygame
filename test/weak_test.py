from weakref import WeakSet

class Sprite:
    """simple base class for visible game objects

    pygame.sprite.Sprite(*groups): return Sprite

    The base class for visible game objects. Derived classes will want to
    override the Sprite.update() method and assign Sprite.image and Sprite.rect
    attributes.  The initializer can accept any number of Group instances that
    the Sprite will become a member of.

    When subclassing the Sprite class, be sure to call the base initializer
    before adding the Sprite to Groups.

    """

    def __init__(self, *groups):
        self.__g = set()  # The groups the sprite is in
        if groups:
            self.add(*groups)

    def add(self, *groups):
        """add the sprite to groups

        Sprite.add(*groups): return None

        Any number of Group instances can be passed as arguments. The
        Sprite will be added to the Groups it is not already a member of.

        """
        has = self.__g.__contains__
        for group in groups:
            if hasattr(group, "_spritegroup"):
                if not has(group):
                    group.add_internal(self)
                    self.add_internal(group)
            else:
                self.add(*group)

    def remove(self, *groups):
        """remove the sprite from groups

        Sprite.remove(*groups): return None

        Any number of Group instances can be passed as arguments. The Sprite
        will be removed from the Groups it is currently a member of.

        """
        has = self.__g.__contains__
        for group in groups:
            if hasattr(group, "_spritegroup"):
                if has(group):
                    group.remove_internal(self)
                    self.remove_internal(group)
            else:
                self.remove(*group)

    def add_internal(self, group):
        """
        For adding this sprite to a group internally.

        :param group: The group we are adding to.
        """
        self.__g.add(group)

    def remove_internal(self, group):
        """
        For removing this sprite from a group internally.

        :param group: The group we are removing from.
        """
        self.__g.remove(group)

    def update(self, *args, **kwargs):
        """method to control sprite behavior

        Sprite.update(*args, **kwargs):

        The default implementation of this method does nothing; it's just a
        convenient "hook" that you can override. This method is called by
        Group.update() with whatever arguments you give it.

        There is no need to use this method if not using the convenience
        method by the same name in the Group class.

        """

    def kill(self):
        """remove the Sprite from all Groups

        Sprite.kill(): return None

        The Sprite is removed from all the Groups that contain it. This won't
        change anything about the state of the Sprite. It is possible to
        continue to use the Sprite after this method has been called, including
        adding it to Groups.

        """
        for group in self.__g:
            group.remove_internal(self)
        self.__g.clear()

    def groups(self):
        """list of Groups that contain this Sprite

        Sprite.groups(): return group_list

        Returns a list of all the Groups that contain this Sprite.

        """
        return list(self.__g)

    def alive(self):
        """does the sprite belong to any groups

        Sprite.alive(): return bool

        Returns True when the Sprite belongs to one or more Groups.
        """
        return bool(self.__g)

    def __repr__(self):
        return f"<{self.__class__.__name__} Sprite(in {len(self.__g)} groups)>"

    @property
    def layer(self):
        """
        Dynamic, read only property for protected _layer attribute.
        This will get the _layer variable if it exists.

        If you try to get it before it is set it will raise an attribute error.

        Layer property can only be set before the sprite is added to a group,
        after that it is read only and a sprite's layer in a group should be
        set via the group's change_layer() method.

        :return: layer as an int, or raise AttributeError.
        """
        return self._layer

    @layer.setter
    def layer(self, value):
        if not self.alive():
            self._layer = value
        else:
            raise AttributeError(
                "Can't set layer directly after "
                "adding to group. Use "
                "group.change_layer(sprite, new_layer) "
                "instead."
            )


class WeakSprite(Sprite):
    def __init__(self, *groups):
        super().__init__(*groups)
        self.__dict__["_Sprite__g"] = WeakSet(self._Sprite__g)

class AbstractGroup:
    """base class for containers of sprites

    AbstractGroup does everything needed to behave as a normal group. You can
    easily subclass a new group class from this or the other groups below if
    you want to add more features.

    Any AbstractGroup-derived sprite groups act like sequences and support
    iteration, len, and so on.

    """

    # dummy val to identify sprite groups, and avoid infinite recursion
    _spritegroup = True

    def __init__(self):
        self.spritedict = {}
        self.lostsprites = []

    def sprites(self):
        """get a list of sprites in the group

        Group.sprites(): return list

        Returns an object that can be looped over with a 'for' loop. (For now,
        it is always a list, but this could change in a future version of
        pygame.) Alternatively, you can get the same information by iterating
        directly over the sprite group, e.g. 'for sprite in group'.

        """
        return list(self.spritedict)

    def add_internal(
        self,
        sprite,
        layer=None,  # noqa pylint: disable=unused-argument; supporting legacy derived classes that override in non-pythonic way
    ):
        """
        For adding a sprite to this group internally.

        :param sprite: The sprite we are adding.
        :param layer: the layer to add to, if the group type supports layers
        """
        self.spritedict[sprite] = None

    def remove_internal(self, sprite):
        """
        For removing a sprite from this group internally.

        :param sprite: The sprite we are removing.
        """
        lost_rect = self.spritedict[sprite]
        if lost_rect:
            self.lostsprites.append(lost_rect)
        del self.spritedict[sprite]

    def has_internal(self, sprite):
        """
        For checking if a sprite is in this group internally.

        :param sprite: The sprite we are checking.
        """
        return sprite in self.spritedict

    def copy(self):
        """copy a group with all the same sprites

        Group.copy(): return Group

        Returns a copy of the group that is an instance of the same class
        and has the same sprites in it.

        """
        return self.__class__(  # noqa pylint: disable=too-many-function-args
            self.sprites()  # Needed because copy() won't work on AbstractGroup
        )

    def __iter__(self):
        return iter(self.sprites())

    def __contains__(self, sprite):
        return self.has(sprite)

    def add(self, *sprites):
        """add sprite(s) to group

        Group.add(sprite, list, group, ...): return None

        Adds a sprite or sequence of sprites to a group.

        """
        for sprite in sprites:
            # It's possible that some sprite is also an iterator.
            # If this is the case, we should add the sprite itself,
            # and not the iterator object.
            if isinstance(sprite, Sprite):
                if not self.has_internal(sprite):
                    self.add_internal(sprite)
                    sprite.add_internal(self)
            else:
                try:
                    # See if sprite is an iterator, like a list or sprite
                    # group.
                    self.add(*sprite)
                except (TypeError, AttributeError):
                    # Not iterable. This is probably a sprite that is not an
                    # instance of the Sprite class or is not an instance of a
                    # subclass of the Sprite class. Alternately, it could be an
                    # old-style sprite group.
                    if hasattr(sprite, "_spritegroup"):
                        for spr in sprite.sprites():
                            if not self.has_internal(spr):
                                self.add_internal(spr)
                                spr.add_internal(self)
                    elif not self.has_internal(sprite):
                        self.add_internal(sprite)
                        sprite.add_internal(self)

    def remove(self, *sprites):
        """remove sprite(s) from group

        Group.remove(sprite, list, or group, ...): return None

        Removes a sprite or sequence of sprites from a group.

        """
        # This function behaves essentially the same as Group.add. It first
        # tries to handle each argument as an instance of the Sprite class. If
        # that fails, then it tries to handle the argument as an iterable
        # object. If that fails, then it tries to handle the argument as an
        # old-style sprite group. Lastly, if that fails, it assumes that the
        # normal Sprite methods should be used.
        for sprite in sprites:
            if isinstance(sprite, Sprite):
                if self.has_internal(sprite):
                    self.remove_internal(sprite)
                    sprite.remove_internal(self)
            else:
                try:
                    self.remove(*sprite)
                except (TypeError, AttributeError):
                    if hasattr(sprite, "_spritegroup"):
                        for spr in sprite.sprites():
                            if self.has_internal(spr):
                                self.remove_internal(spr)
                                spr.remove_internal(self)
                    elif self.has_internal(sprite):
                        self.remove_internal(sprite)
                        sprite.remove_internal(self)

    def has(self, *sprites):
        """ask if group has a sprite or sprites

        Group.has(sprite or group, ...): return bool

        Returns True if the given sprite or sprites are contained in the
        group. Alternatively, you can get the same information using the
        'in' operator, e.g. 'sprite in group', 'subgroup in group'.

        """
        if not sprites:
            return False  # return False if no sprites passed in

        for sprite in sprites:
            if isinstance(sprite, Sprite):
                # Check for Sprite instance's membership in this group
                if not self.has_internal(sprite):
                    return False
            else:
                try:
                    if not self.has(*sprite):
                        return False
                except (TypeError, AttributeError):
                    if hasattr(sprite, "_spritegroup"):
                        for spr in sprite.sprites():
                            if not self.has_internal(spr):
                                return False
                    else:
                        if not self.has_internal(sprite):
                            return False

        return True

    def update(self, *args, **kwargs):
        """call the update method of every member sprite

        Group.update(*args, **kwargs): return None

        Calls the update method of every member sprite. All arguments that
        were passed to this method are passed to the Sprite update function.

        """
        for sprite in self.sprites():
            sprite.update(*args, **kwargs)

    def draw(
        self, surface, bgsurf=None, special_flags=0
    ):  # noqa pylint: disable=unused-argument; bgsurf arg used in LayeredDirty
        """draw all sprites onto the surface

        Group.draw(surface, special_flags=0): return Rect_list

        Draws all of the member sprites onto the given surface.

        """
        sprites = self.sprites()
        if hasattr(surface, "blits"):
            self.spritedict.update(
                zip(
                    sprites,
                    surface.blits(
                        (spr.image, spr.rect, None, special_flags) for spr in sprites
                    ),
                )
            )
        else:
            for spr in sprites:
                self.spritedict[spr] = surface.blit(
                    spr.image, spr.rect, None, special_flags
                )
        self.lostsprites = []
        dirty = self.lostsprites

        return dirty

    def clear(self, surface, bgd):
        """erase the previous position of all sprites

        Group.clear(surface, bgd): return None

        Clears the area under every drawn sprite in the group. The bgd
        argument should be Surface which is the same dimensions as the
        screen surface. The bgd could also be a function which accepts
        the given surface and the area to be cleared as arguments.

        """
        if callable(bgd):
            for lost_clear_rect in self.lostsprites:
                bgd(surface, lost_clear_rect)
            for clear_rect in self.spritedict.values():
                if clear_rect:
                    bgd(surface, clear_rect)
        else:
            surface_blit = surface.blit
            for lost_clear_rect in self.lostsprites:
                surface_blit(bgd, lost_clear_rect, lost_clear_rect)
            for clear_rect in self.spritedict.values():
                if clear_rect:
                    surface_blit(bgd, clear_rect, clear_rect)

    def empty(self):
        """remove all sprites

        Group.empty(): return None

        Removes all the sprites from the group.

        """
        for sprite in self.sprites():
            self.remove_internal(sprite)
            sprite.remove_internal(self)

    def __bool__(self):
        return bool(self.sprites())

    def __len__(self):
        """return number of sprites in group

        Group.len(group): return int

        Returns the number of sprites contained in the group.

        """
        return len(self.sprites())

    def __repr__(self):
        return f"<{self.__class__.__name__}({len(self)} sprites)>"


class Group(AbstractGroup):
    """container class for many Sprites

    pygame.sprite.Group(*sprites): return Group

    A simple container for Sprite objects. This class can be subclassed to
    create containers with more specific behaviors. The constructor takes any
    number of Sprite arguments to add to the Group. The group supports the
    following standard Python operations:

        in      test if a Sprite is contained
        len     the number of Sprites contained
        bool    test if any Sprites are contained
        iter    iterate through all the Sprites

    The Sprites in the Group are not ordered, so the Sprites are drawn and
    iterated over in no particular order.

    """

    def __init__(self, *sprites):
        AbstractGroup.__init__(self)
        self.add(*sprites)

sprite = WeakSprite()
group = Group(sprite)

del group
import gc
gc.collect()

assert not sprite.alive()
