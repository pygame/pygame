.. include:: ../common.txt

.. highlight:: c

************************************************************************
Slots and c_api - Making functions and data available from other modules
************************************************************************


One example is pg_RGBAFromObj where the implementation is defined in base.c, and also exported in base.c (and _pygame.h).

base.c has this exposing the pg_RGBAFromObj function to the `c_api` structure:

    c_api[12] = pg_RGBAFromObj;


Then in src_c/include/_pygame.h there is an

	#define pg_RGBAFromObj.

Also in _pygame.h, it needs to define the number of slots the base module uses. This is PYGAMEAPI_BASE_NUMSLOTS. So if you were adding another function, you need to increment this PYGAMEAPI_BASE_NUMSLOTS number.

Then to use the pg_RGBAFromObj in other files,

1) include the "pygame.h" file,
2) they have to make sure base is imported with:

	import_pygame_base();

Examples that use pg_RGBAFromObj are: _freetype.c, color.c, gfxdraw.c, and surface.c.
