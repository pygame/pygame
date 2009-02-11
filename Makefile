PREFIX ?= /usr/local
LIBDIR ?= $(PREFIX)/lib
INCDIR ?= $(PREFIX)/include

CC ?= gcc
CFLAGS ?= -W -Wall -g
INCLUDES ?= -I/usr/local/include
LIBS ?= 

INSTALL ?= install
INSTALL_DATA ?= $(INSTALL) -c -m 444

LIBTOOL ?= libtool
LTCOMPILE ?= $(LIBTOOL) --mode=compile $(CC)
LTLINK ?= $(LIBTOOL) --mode=link --tag=CC
LTINSTALL ?= $(LIBTOOL) --mode=install
LTFINISH ?= $(LIBTOOL) --finish

# Changed or added interfaces: CURRENT++ and AGE++ and REVISION = 0
# binary compatibility broken: CURRENT++, REVISION and AGE = 0
# interfaces stay the same as before: REVISION++
#
# CURRENT = most recent version of the library that the library supports
# REVISION = implementation number of the current interface
# AGE = number of versions back this version is still backwards
# compatible with.
CURRENT = 0
AGE = 0
REVISION = 0

OBJDIR = obj
BLDDIR = lib
SRCDIR = src
HEADERS = src/draw.h \
	src/filters.h \
	src/jpg.h \
	src/pgpng.h \
	src/scrap.h \
	src/surface.h \
	src/tga.h \
	src/transform.h

SRC_DRAW = draw.c
TGT_DRAW = libSDL_pgdraw.la
LT_DRAW = $(SRC_DRAW:%.c=$(OBJDIR)/%.lo)
OBJ_DRAW = $(SRC_DRAW:%.c=%.o)
CFLAGS_DRAW ?=
LIBS_DRAW ?=

SRC_SCRAP = scrap.c scrap_win.c scrap_x11.c
TGT_SCRAP = libSDL_pgscrap.la
LT_SCRAP = $(SRC_SCRAP:%.c=$(OBJDIR)/%.lo)
OBJ_SCRAP = $(SRC_SCRAP:%.c=%.o)
CFLAGS_SCRAP ?=
LIBS_SCRAP ?= -lX11

SRC_SURFACE = jpg.c png.c surface_blit.c surface_fill.c surface_save.c tga.c
TGT_SURFACE = libSDL_pgsurface.la
LT_SURFACE = $(SRC_SURFACE:%.c=$(OBJDIR)/%.lo)
OBJ_SURFACE = $(SRC_SURFACE:%.c=%.o)
CFLAGS_SURFACE ?= -DHAVE_JPG -DHAVE_PNG `pkg-config --cflags libpng`
LIBS_SURFACE ?= -ljpeg `pkg-config --libs libpng`

SRC_TRANSFORM = filters.c transform.c
TGT_TRANSFORM = libSDL_pgtransform.la
LT_TRANSFORM = $(SRC_TRANSFORM:%.c=$(OBJDIR)/%.lo)
OBJ_TRANSFORM = $(SRC_TRANSFORM:%.c=%.o)
CFLAGS_TRANSFORM ?=
LIBS_TRANSFORM ?=

SOURCES= $(SRC_DRAW) $(SRC_SCRAP) $(SRC_SURFACE) $(SRC_TRANSFORM)
TARGETS= $(TGT_DRAW) $(TGT_SCRAP) $(TGT_SURFACE) $(TGT_TRANSFORM)

SDLFLAGS = `sdl-config --cflags`
SDLLIBS = `sdl-config --libs`
LIBS += $(SDLLIBS) -lm
CFLAGS += $(SDLFLAGS) $(INCLUDES)

all: clean dirs $(TARGETS)

dirs:
	@mkdir -p $(OBJDIR) $(BLDDIR)

# draw library
$(OBJ_DRAW):
	@$(LTCOMPILE) $(CFLAGS) $(CFLAGS_DRAW) \
		-c $(SRCDIR)/$*.c -o $(OBJDIR)/$*.o

$(TGT_DRAW): $(OBJ_DRAW)
	$(LTLINK) $(LT_DRAW) -o $(BLDDIR)/$(TGT_DRAW) -rpath $(PREFIX) \
		$(LIBS) $(LIBS_DRAW) \
		-version-info $(CURRENT):$(REVISION):$(AGE)

# scrap library
$(OBJ_SCRAP):
	@$(LTCOMPILE) $(CFLAGS) $(CFLAGS_SCRAP) \
		-c $(SRCDIR)/$*.c -o $(OBJDIR)/$*.o

$(TGT_SCRAP): $(OBJ_SCRAP)
	$(LTLINK) $(LT_SCRAP) -o $(BLDDIR)/$(TGT_SCRAP) -rpath $(PREFIX) \
		$(LIBS) $(LIBS_SCRAP) \
		-version-info $(CURRENT):$(REVISION):$(AGE)

# surface library
$(OBJ_SURFACE):
	@$(LTCOMPILE) $(CFLAGS) $(CFLAGS_SURFACE) \
		-c $(SRCDIR)/$*.c -o $(OBJDIR)/$*.o

$(TGT_SURFACE): $(OBJ_SURFACE)
	$(LTLINK) $(LT_SURFACE) -o $(BLDDIR)/$(TGT_SURFACE) -rpath $(PREFIX) \
		$(LIBS) $(LIBS_SURFACE) \
		-version-info $(CURRENT):$(REVISION):$(AGE)

# transform library
$(OBJ_TRANSFORM):
	@$(LTCOMPILE) $(CFLAGS) $(CFLAGS_TRANSFORM) \
		-c $(SRCDIR)/$*.c -o $(OBJDIR)/$*.o

$(TGT_TRANSFORM): $(OBJ_TRANSFORM)
	$(LTLINK) $(LT_TRANSFORM) -o $(BLDDIR)/$(TGT_TRANSFORM) \
		-rpath $(PREFIX) $(LIBS) $(LIBS_TRANSFORM) \
		-version-info $(CURRENT):$(REVISION):$(AGE)

install:
	$(INSTALL) -d $(INCDIR)/SDL/pgame
	$(INSTALL_DATA) $(HEADERS) $(INCDIR)/SDL/pgame
	$(INSTALL) -d $(LIBDIR)
	for target in $(TARGETS); do \
		$(LTINSTALL) install -c -s $(BLDDIR)/$$target \
			 $(LIBDIR)/$$target; \
	done
	$(LTFINISH) $(PREFIX)

clean:
	rm -rf $(SRCDIR)/*~ *~ $(OBJDIR) $(BLDDIR) $(TARGETS) .libs
