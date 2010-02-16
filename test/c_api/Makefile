SOURCES = \
	base_tests.c \
	mask_tests.c \
	sdl_tests.c

OBJECTS = $(SOURCES:%.c=%.o)
TARGETS = base_tests mask_tests sdl_tests

CC ?= gcc
CFLAGS = -W -Wall -g 

# Windows includes (MinGW)
#INCLUDES = -DWIN32 -I. -Ic:\\Python26\\include
#LIBS = -Lc:\\Python25\\libs -mwindows -lm -lpython26 

# Linux/Unix includes
INCLUDES = -I. -I/usr/local/include/ -I/usr/local/include/python2.6
LIBS = -L/usr/local/lib -lpython2.6 -lm

CFLAGS += $(INCLUDES)
LDFLAGS += $(LIBS)

SDLCFLAGS = `pkg-config --cflags sdl`
SDLLDFLAGS = `pkg-config --libs sdl`

all: $(OBJECTS) $(TARGETS)

base_tests.o: base_tests.c
	$(CC) $(CFLAGS) -c -o $@ $*.c
base_tests: base_tests.o
	$(CC) base_tests.o -o base_tests $(LDFLAGS) 

mask_tests.o: mask_tests.c
	$(CC) $(CFLAGS) -c -o $@ $*.c
mask_tests: mask_tests.o
	$(CC) mask_tests.o -o mask_tests $(LDFLAGS) 

sdl_tests.o: sdl_tests.c
	$(CC) $(CFLAGS) $(SDLCFLAGS) -c -o $@ $*.c
sdl_tests: sdl_tests.o
	$(CC) sdl_tests.o -o sdl_tests $(LDFLAGS) $(SDLLDFLAGS)

clean:
	@rm -f *.o *.orig *.core *~ $(TARGETS)
