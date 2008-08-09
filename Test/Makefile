SOURCES = Main.c pgPhysicsRenderer.c
OBJECTS = $(SOURCES:%.c=%.o)
TARGET = physics_test

CC ?= gcc
CFLAGS ?= -W -Wall -g 

# Windows includes (MinGW)
#INCLUDES = -DWIN32 -I. -Ic:\\Python25\\include
#LIBS = -Lc:\\Python25\\libs -mwindows -lm -lpython25 -lopengl32 -lglu32 -lglut32win 

# Linux/Unix includes
INCLUDES = -I. -I/usr/local/include/ -I/usr/local/include/python2.5
LIBS = -L/usr/local/lib -lGL -lGLU -lglut -lpython2.5 -lm

CFLAGS += $(INCLUDES)
LDFLAGS += $(LIBS)

all: $(OBJECTS) $(TARGET)

$(OBJECTS): %.o:%.c
	$(CC) $(CFLAGS) -c -o $@ $*.c

$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -o $(TARGET) $(LDFLAGS) 

clean:
	rm -f *.o $(TARGET)