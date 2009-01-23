SOURCES = \
	base_tests.c \
	mask_tests.c \
	physics_tests.c
OBJECTS = $(SOURCES:%.c=%.o)
TARGETS = base_tests mask_tests physics_tests

CC ?= gcc
CFLAGS = -W -Wall -g 

# Windows includes (MinGW)
#INCLUDES = -DWIN32 -I. -Ic:\\Python25\\include
#LIBS = -Lc:\\Python25\\libs -mwindows -lm -lpython25 

# Linux/Unix includes
INCLUDES = -I. -I/usr/local/include/ -I/usr/local/include/python2.5
LIBS = -L/usr/local/lib -lpython2.5 -lm

CFLAGS += $(INCLUDES)
LDFLAGS += $(LIBS)

all: $(OBJECTS) $(TARGETS)

base_tests.o: base_tests.c
	$(CC) $(CFLAGS) -c -o $@ $*.c
base_tests: base_tests.o
	$(CC) base_tests.o -o base_tests $(LDFLAGS) 

mask_tests.o: mask_tests.c
	$(CC) $(CFLAGS) -c -o $@ $*.c
mask_tests: mask_tests.o
	$(CC) mask_tests.o -o mask_tests $(LDFLAGS) 

physics_tests.o: physics_tests.c
	$(CC) $(CFLAGS) -c -o $@ $*.c
physics_tests: physics_tests.o
	$(CC) physics_tests.o -o physics_tests $(LDFLAGS) 

clean:
	rm -f *.o *.core *~ $(TARGETS)
