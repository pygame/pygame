import warnings

warnings.warn(
    "The ._sprite submodule is deprecated since 2.4.0. Please use pygame.sprite instead.",
    category=DeprecationWarning,
    stacklevel=2,
)

from pygame.sprite import (
    AbstractGroup,
    DirtySprite,
    Group,
    GroupSingle,
    LayeredDirty,
    LayeredUpdates,
    RenderClear,
    RenderPlain,
    RenderUpdates,
    Sprite,
    collide_circle,
    collide_circle_ratio,
    collide_mask,
    collide_rect,
    collide_rect_ratio,
    groupcollide,
    spritecollide,
    spritecollideany,
)
