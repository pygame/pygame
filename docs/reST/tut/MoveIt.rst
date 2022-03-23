.. TUTORIAL:Help! How Do I Move An Image?

.. include:: common.txt

****************************************************
  Pygame Tutorials - Help! How Do I Move An Image?
****************************************************

Help! How Do I Move An Image?
=============================

.. rst-class:: docinfo

:Author: Pete Shinners
:Contact: pete@shinners.org


Many people new to programming and graphics have a hard time figuring
out  how to make an image move around the screen. Without understanding
all  the  concepts, it can be very confusing. You're not the first person
to be  stuck  here, I'll do my best to take things step by step. We'll even
try to end with methods of keeping your animations efficient.

Note that we won't be teaching you to program with python in this article,
just introduce you to some of the basics with pygame.


Just Pixels On The Screen
-------------------------

Pygame has a display Surface. This is basically an image that is visible
on the screen, and the image is made up of pixels. The main way you change
these pixels is by calling the blit() function. This copies the pixels
from   one image onto another.

This is the first thing to understand. When you blit an image onto the
screen, you are simply changing the color of the pixels on the screen.
Pixels aren't added or moved, we just change the colors of the pixels already
on the screen. These images you blit to the screen are also Surfaces in
pygame, but they are in no way connected to the display Surface. When they
are blitted  to the screen they are copied into the display, but you still
have a unique  copy of the original.

With this brief description. Perhaps you can already understand what
is needed to "move" an image. We don't actually move anything at all. We
simply blit the image in a new position. But before we draw the image in
the new position, we'll need to "erase" the old one. Otherwise the image
will be visible in two places on the screen. By rapidly erasing the image
and redrawing it in a new place, we achieve the "illusion" of movement.

Through the rest of this tutorial we will break this process down into
simpler steps. Even explaining the best ways to have multiple images moving
around the screen. You probably already have questions. Like, how do we
"erase" the image before drawing it in a new position? Perhaps you're still
totally lost? Well hopefully the rest of this tutorial can straighten things
out for you.


Let's Go Back A Step
--------------------

Perhaps the concept of pixels and images is still a little foreign to
you?  Well good news, for the next few sections we are going to use code that
does  everything we want, it just doesn't use pixels. We're going to create
a small  python list of 6 numbers, and imagine it represents some fantastic
graphics  we could see on the screen. It might actually be surprising how
closely this  represents exactly what we'll later be doing with real graphics.

So let's begin by creating our screen list and fill it with a beautiful
landscape of 1s and 2s. ::

  >>> screen = [1, 1, 2, 2, 2, 1]
  >>> print(screen)
  [1, 1, 2, 2, 2, 1]


Now we've created our background. It's not going to be very exciting
unless   we also draw a player on the screen. We'll create a mighty hero
that looks   like the number 8. Let's stick him near the middle of the map
and see what   it looks like. ::

  >>> screen[3] = 8
  >>> print(screen)
  [1, 1, 2, 8, 2, 1]


This might have been as far as you've gotten if you jumped right in doing
some graphics programming with pygame. You've got some nice looking stuff
on the screen, but it cannot move anywhere. Perhaps now that our screen
is  just a list of numbers, it's easier to see how to move him?


Making The Hero Move
--------------------

Before we can start moving the character. We need to keep track of some
sort of position for him. In the last section when we drew him, we just picked
an arbitrary position. Let's do it a little more officially this time. ::

  >>> playerpos = 3
  >>> screen[playerpos] = 8
  >>> print(screen)
  [1, 1, 2, 8, 2, 1]


Now it is pretty easy to move him to a new position. We simply change
the  value of playerpos, and draw him on the screen again. ::

  >>> playerpos = playerpos - 1
  >>> screen[playerpos] = 8
  >>> print(screen)
  [1, 1, 8, 8, 2, 1]


Whoops. Now we can see two heroes. One in the old position, and one
in his new position. This is exactly the reason we need to "erase" the hero
in his old position before we draw him in the new position. To erase him,
we need to change that value in the list back to what it was before the hero
was there. That means we need to keep track of the values on the screen before
the hero replaced them. There's several way you could do this, but the easiest
is usually to keep a separate copy of the screen background. This means
we  need to make some changes to our little game.


Creating A Map
--------------

What we want to do is create a separate list we will call our background.
We will create the background so it looks like our original screen did,
with  1s and 2s. Then we will copy each item from the background to the screen.
After that we can finally draw our hero back onto the screen. ::

  >>> background = [1, 1, 2, 2, 2, 1]
  >>> screen = [0]*6                         #a new blank screen
  >>> for i in range(6):
  ...     screen[i] = background[i]
  >>> print(screen)
  [1, 1, 2, 2, 2, 1]
  >>> playerpos = 3
  >>> screen[playerpos] = 8
  >>> print(screen)
  [1, 1, 2, 8, 2, 1]


It may seem like a lot of extra work. We're no farther off than we were
before the last time we tried to make him move. But this time we have the
extra information we need to move him properly.


Making The Hero Move (Take 2)
-----------------------------

This time it will be easy to move the hero around. First we will erase
the  hero from his old position. We do this by copying the correct value
from the background onto the screen. Then we will draw the character in his
new position on the screen


  >>> print(screen)
  [1, 1, 2, 8, 2, 1]
  >>> screen[playerpos] = background[playerpos]
  >>> playerpos = playerpos - 1
  >>> screen[playerpos] = 8
  >>> print(screen)
  [1, 1, 8, 2, 2, 1]


There it is. The hero has moved one space to the left. We can use this
same  code to move him to the left again. ::

  >>> screen[playerpos] = background[playerpos]
  >>> playerpos = playerpos - 1
  >>> screen[playerpos] = 8
  >>> print(screen)
  [1, 8, 2, 2, 2, 1]


Excellent! This isn't exactly what you'd call smooth animation. But with
a couple small changes, we'll make this work directly with graphics on
the   screen.


Definition: "blit"
------------------

In the next sections we will transform our program from using lists to
using real graphics on the screen. When displaying the graphics we will
use the term **blit** frequently. If you are new to doing graphics
work, you are probably unfamiliar with this common term.

BLIT: Basically, blit means to copy graphics from one image
to another. A more formal definition is to copy an array of data
to a bitmapped array destination. You can think of blit as just
*"assigning"* pixels. Much like setting values in our screen-list
above, blitting assigns the color of pixels in our image.

Other graphics libraries will use the word *bitblt*, or just *blt*,
but they are talking about the same thing. It is basically copying
memory from one place to another. Actually, it is a bit more advanced than
straight copying of memory, since it needs to handle things like pixel
formats, clipping, and scanline pitches. Advanced blitters can also
handle things like transparency and other special effects.


Going From The List To The Screen
---------------------------------

To take the code we see in the above to examples and make them work with
pygame is very straightforward. We'll pretend we have loaded some pretty
graphics and named them "terrain1", "terrain2", and "hero". Where before
we assigned numbers to a list, we now blit graphics to the screen. Another
big change, instead of using positions as a single index (0 through 5), we
now need a two dimensional coordinate. We'll pretend each of the graphics
in our game is 10 pixels wide. ::

  >>> background = [terrain1, terrain1, terrain2, terrain2, terrain2, terrain1]
  >>> screen = create_graphics_screen()
  >>> for i in range(6):
  ...     screen.blit(background[i], (i*10, 0))
  >>> playerpos = 3
  >>> screen.blit(playerimage, (playerpos*10, 0))


Hmm, that code should seem very familiar, and hopefully more importantly;
the code above should make a little sense. Hopefully my illustration of setting
simple values in a list shows the similarity of setting pixels on the screen
(with blit). The only part that's really extra work is converting the player position
into coordinates on the screen. For now we just use a crude :code:`(playerpos*10, 0)` ,
but we can certainly do better than that. Now let's move the player
image over a space. This code should have no surprises. ::

  >>> screen.blit(background[playerpos], (playerpos*10, 0))
  >>> playerpos = playerpos - 1
  >>> screen.blit(playerimage, (playerpos*10, 0))


There you have it. With this code we've shown how to display a simple background
with a hero's image on it. Then we've properly moved that hero one space
to the left. So where do we go from here? Well for one the code is still
a little awkward. First thing we'll want to do is find a cleaner way to represent
the background and player position. Then perhaps a bit of smoother, real
animation.


Screen Coordinates
------------------

To position an object on the screen, we need to tell the blit() function
where to put the image. In pygame we always pass positions as an (X,Y) coordinate.
This represents the number of pixels to the right, and the number of pixels
down to place the image. The top-left corner of a Surface is coordinate (0,
0). Moving to the right a little would be (10, 0), and then moving down just
as much would be (10, 10). When blitting, the position argument represents
where the topleft corner of the source should be placed on the destination.

Pygame comes with a convenient container for these coordinates, it is a
Rect. The Rect basically represents a rectangular area in these coordinates.
It has topleft corner and a size. The Rect comes with a lot of convenient
methods which help you move and position them. In our next examples we will
represent the positions of our objects with the Rects.

Also know that many functions in pygame expect Rect arguments. All of these
functions can also accept a simple tuple of 4 elements (left, top, width,
height). You aren't always required to use these Rect objects, but you will
mainly want to. Also, the blit() function can accept a Rect as its position
argument, it simply uses the topleft corner of the Rect as the real position.


Changing The Background
-----------------------

In all our previous sections, we've been storing the background as a list
of different types of ground. That is a good way to create a tile-based game,
but we want smooth scrolling. To make that a little easier, we're going to
change the background into a single image that covers the whole screen. This
way, when we want to "erase" our objects (before redrawing them) we only need
to blit the section of the erased background onto the screen.

By passing an optional third Rect argument to blit, we tell blit to only
use that subsection of the source image. You'll see that in use below as we
erase the player image.

Also note, now when we finish drawing to the screen, we call pygame.display.update()
which will show everything we've drawn onto the screen.


Smooth Movement
---------------

To make something appear to move smoothly, we only want to move it a couple
pixels at a time. Here is the code to make an object move smoothly across
the screen. Based on what we already now know, this should look pretty simple. ::

  >>> screen = create_screen()
  >>> player = load_player_image()
  >>> background = load_background_image()
  >>> screen.blit(background, (0, 0))        #draw the background
  >>> position = player.get_rect()
  >>> screen.blit(player, position)          #draw the player
  >>> pygame.display.update()                #and show it all
  >>> for x in range(100):                   #animate 100 frames
  ...     screen.blit(background, position, position) #erase
  ...     position = position.move(2, 0)     #move player
  ...     screen.blit(player, position)      #draw new player
  ...     pygame.display.update()            #and show it all
  ...     pygame.time.delay(100)             #stop the program for 1/10 second


There you have it. This is all the code that is needed to smoothly animate
an object across the screen. We can even use a pretty background character.
Another benefit of doing the background this way, the image for the player
can have transparency or cutout sections and it will still draw correctly
over the background (a free bonus).

We also throw in a call to pygame.time.delay() at the end of our loop above.
This slows down our program a little, otherwise it might run so fast you might
not see it.


So, What Next?
--------------

Well there we have it. Hopefully this article has done everything it promised
to do. But, at this point the code really isn't ready for the next best-selling
game. How do we easily have multiple moving objects? What exactly are those
mysterious functions like load_player_image()? We also need a way to get simple
user input, and loop for more than 100 frames. We'll take the example we
have here, and turn it into an object oriented creation that would make momma
proud.


First, The Mystery Functions
----------------------------

Full information on these types of functions can be found in other tutorials
and reference. The pygame.image module has a load() function which will do
what we want. The lines to load the images should become this. ::

  >>> player = pygame.image.load('player.bmp').convert()
  >>> background = pygame.image.load('liquid.bmp').convert()


We can see that's pretty simple, the load function just takes a filename
and returns a new Surface with the loaded image. After loading we make a call
to the Surface method, convert(). Convert returns us a new Surface of the
image, but now converted to the same pixel format as our display. Since the
images will be the same format at the screen, they will blit very quickly.
If we did not convert, the blit() function is slower, since it has to convert
from one type of pixel to another as it goes.

You may also have noticed that both the load() and convert() return new
Surfaces. This means we're really creating two Surfaces on each of these
lines. In other programming languages, this results in a memory leak (not
a good thing). Fortunately Python is smart enough to handle this, and pygame
will properly clean up the Surface we end up not using.

The other mystery function we saw in the above example was create_screen().
In pygame it is simple to create a new window for graphics. The code to create
a 640x480 surface is below. By passing no other arguments, pygame will just
pick the best color depth and pixel format for us. ::

  >>> screen = pygame.display.set_mode((640, 480))


Handling Some Input
-------------------

We desperately need to change the main loop to look for any user input, (like
when the user closes the window). We need to add "event handling" to our
program. All graphical programs use this Event Based design. The program
gets events like "keyboard pressed" or "mouse moved" from the computer. Then
the program responds to the different events. Here's what the code should
look like. Instead of looping for 100 frames, we'll keep looping until the
user asks us to stop. ::

  >>> while True:
  ...     for event in pygame.event.get():
  ...         if event.type in (QUIT, KEYDOWN):
  ...             sys.exit()
  ...     move_and_draw_all_game_objects()


What this code simply does is, first loop forever, then check if there are
any events from the user. We exit the program if the user presses the keyboard
or the close button on the window. After we've checked all the events we
move and draw our game objects. (We'll also erase them before they move,
too)


Moving Multiple Images
----------------------

Here's the part where we're really going to change things around. Let's
say we want 10 different images moving around on the screen. A good way to
handle this is to use python's classes. We'll create a class that represents
our game object. This object will have a function to move itself, and then
we can create as many as we like. The functions to draw and move the object
need to work in a way where they only move one frame (or one step) at a time.
Here's the python code to create our class. ::

  >>> class GameObject:
  ...     def __init__(self, image, height, speed):
  ...         self.speed = speed
  ...         self.image = image
  ...         self.pos = image.get_rect().move(0, height)
  ...     def move(self):
  ...         self.pos = self.pos.move(0, self.speed)
  ...         if self.pos.right > 600:
  ...             self.pos.left = 0


So we have two functions in our class. The init function constructs our object.
It positions the object and sets its speed. The move method moves the object
one step. If it's gone too far, it moves the object back to the left.


Putting It All Together
-----------------------

Now with our new object class, we can put together the entire game. Here
is what the main function for our program will look like. ::

  >>> screen = pygame.display.set_mode((640, 480))
  >>> player = pygame.image.load('player.bmp').convert()
  >>> background = pygame.image.load('background.bmp').convert()
  >>> screen.blit(background, (0, 0))
  >>> objects = []
  >>> for x in range(10):                    #create 10 objects</i>
  ...     o = GameObject(player, x*40, x)
  ...     objects.append(o)
  >>> while True:
  ...     for event in pygame.event.get():
  ...         if event.type in (QUIT, KEYDOWN):
  ...             sys.exit()
  ...     for o in objects:
  ...         screen.blit(background, o.pos, o.pos)
  ...     for o in objects:
  ...         o.move()
  ...         screen.blit(o.image, o.pos)
  ...     pygame.display.update()
  ...     pygame.time.delay(100)


And there it is. This is the code we need to animate 10 objects on the screen.
The only point that might need explaining is the two loops we use to clear
all the objects and draw all the objects. In order to do things properly,
we need to erase all the objects before drawing any of them. In our sample
here it may not matter, but when objects are overlapping, using two loops
like this becomes important.


Further Improvements
--------------------

pygame.time.delay() is not the best function to be using! One of the most important
aspects of any game is the framerate, as I'm sure many of you know. Currently, our
simple implementation has no limit on the framerate, using pygame.time.delay() to
control the processing. However, pygame has a function that's just as simple, and is
more effective for input control.

First, let's grab the clock element from pygame.time. We can do this with the
following code snipit:
 >>> clock = pygame.time.Clock()

This clock element has the funciton .tick(), which accepts an integer input.
Using the following code, we can set the framerate to a common and respectable 60:
 >>> clock.tick(60)

Of course, you can use any number for the framerate here. Try experimenting to
see what might work best for you, or what happens when you have a very high limit.
Keep in mind, a tick occurs x amount of times per second. In the above example, there
will be 60 ticks per second. Keep this in mind, as you may be surprised when an object
moves 60 times more than you expect!

We need to make sure the clock element is defined outside of the game loop, so
it isn't constantly reinstantiated. clock.tick(), however, must be within the
game loop. With those minor adjustments, we come to the new code implementation:

  >>> screen = pygame.display.set_mode((640, 480))
  >>> player = pygame.image.load('player.bmp').convert()
  >>> background = pygame.image.load('background.bmp').convert()
  >>> screen.blit(background, (0, 0))
  >>> clock = pygame.time.Clock()
  >>> objects = []
  >>> for x in range(10):                    #create 10 objects</i>
  ...     o = GameObject(player, x*40, x)
  ...     objects.append(o)
  >>> while True:
  ...     for event in pygame.event.get():
  ...         if event.type in (QUIT, KEYDOWN):
  ...             sys.exit()
  ...     for o in objects:
  ...         screen.blit(background, o.pos, o.pos)
  ...     for o in objects:
  ...         o.move()
  ...         screen.blit(o.image, o.pos)
  ...     pygame.display.update()
  ...     clock.tick(60)

Now our 10 objects are being moved at a speed controlled by the framerate!


Advanced Player Input
---------------------

Let's say you want to be able to control one of the objects on screen. Where
would we start? Well, it would be a good idea to have a different image to represent
this controlled character. So, we'll add a new image.load() of a different image.
We already have a function for movement, so we can create a special player entity
outside of the object list. But our move function only allows the user to move to
the right! Perhaps there's a better way to handle things...

From here on, things are going to get a bit more complicated, but not too much so.
Let's first redefine the move function as follows:

 >>> # move the object. Defaults to moving right.
 >>> def move(self, up=0, down=0, left=0, right=1):
 ...    if right:
 ...        self.pos.right += self.speed
 ...    if left:
 ...        self.pos.right -= self.speed
 ...    if down:
 ...        self.pos.top += self.speed
 ...    if up:
 ...        self.pos.top -= self.speed
 ...    else: return
        
 ...    # controls the object such that it cannot leave the screen's viewpoint
 ...    if self.pos.right > 640:
 ...        self.pos.left = 0
 ...    if self.pos.top > 420:
 ...        self.pos.top = 0
 ...    if self.pos.right < 79:
 ...        self.pos.right = 640
 ...    if self.pos.top < 0:
 ...        self.pos.top = 420

There's certainly a lot more going on, but it isn't all too complex. Let's take it one step
at a time. For starters, we're now passing some additional parameters: up, down, left, and right. 
We see that the have defaullt conditions, as defined by them being declared in the parameter such as
up=0 and right=1. When a parameter is like this, that means it has a default value. If we call
o.move(), it will take the default direction values into account, and by will move to the right. 
That way, we don't have to make any changes to the objects we already created.

Additionally, we are now checking the direction. If it is moving right, move right. Up, move up.
If any of the values are 0, they won't move. We'll be setting these in the game loop. And finally, we
need to add additional conditionals for moving the object back in bounds. We have the original
of it moving too far right, then being moved back to the left. We will do similar conditions
for all the other directions: up, left, and down. Now, our object can't leave the screen!

Those are all the changes that need to be made to our object class. Now we can move into some
simple user input. One of the most important things to keep in mind when moving forward is:
no matter how tempting, do not directly move an object in the event handling! This will become
really messy, and it won't work nearly as well as you might expect. Instead, the best thing
to do is keep track of conditionals, and handle them after event handling. We'll start
by declaring the basic directions:

 >>> up = 0
 >>> down = 0
 >>> left = 0
 >>> right = 0

I used integers here, however 1s and 0s function identically to True and False. Now, we need to
update our extremely basic event handling while keeping two things in mind:

1: We always want to move while a key is held, and we don't want to override with a different key.
2: When a key is released, we want to stop moving.

pygame has two events that are perfect for the job: KEYDOWN and KEYUP. In fact, we're already
using the keydown event, where if a key is pressed we exit the program. However, we don't want
to be doing that anymore. The code for the new event handler will be displayed below, and I will
do my best to explain everything:

 >>> for e in pg.event.get():

 ...        # if a key is pressed, make sure that a value is set so that it can move in multiple directions
 ...        if e.type == pg.KEYDOWN:
 ...            if e.key == pg.K_DOWN:
 ...                down = 1
 ...            elif e.key == pg.K_UP:
 ...                up = 1
 ...            elif e.key == pg.K_LEFT:
 ...                left = 1
 ...            elif e.key == pg.K_RIGHT:
 ...                right = 1
 ...            elif e.key == pg.K_ESCAPE:
 ...                exit()

 ...        # if a key is lifted, make sure the value is removed so it can stop moving
 ...        if e.type == pg.KEYUP:
 ...            if e.key == pg.K_DOWN:
 ...                down = 0
 ...            elif e.key == pg.K_UP:
 ...                up = 0
 ...            elif e.key == pg.K_LEFT:
 ...                left = 0
 ...            elif e.key == pg.K_RIGHT:
 ...                right = 0
            
 ...        # quit upon screen exit
 ...        if e.type == pg.QUIT:
 ...            exit()

We start our loop with "for e in pg.event.get()" which gets a list of all pygame events currently
happening. From here, we know we need to look for the two specific types of input: KEYDOWN and KEYUP.
Let's start with KEYDOWN. We see that when we press a key, the event type will be equal to KEYDOWN. 
From here, we want to determine what keys are being pressed. For this example, we'll use the arrow
keys. Pygame has all of the keys defined, but we will be using K_DOWN, K_UP, K_LEFT, K_RIGHT. Using
our conditionals we set earlier, we can change their status to 1, or active, if a key is pressed down.
And that's really all we're doing here! We also have K_ESCAPE to quit the code if escape is pressed.

KEYUP is essentially the same! We know that if a key is let go, it's no longer being held. So, we
can simply set the conditionals back to 0, or inactive, when a key is let go. At this point, our
event handling is nearly complete, although it's a good idea to allow the user to close the window
with the red x, which is where the QUIT event comes into play. Now our handling is done.

At this point, all we have to do is move the player character, which we can do simply with:
 >>> p.move(up,down,left,right)

One Final Look at the Code
--------------------------

At this point, our main is more robust. Let's have one last look at it all together:

 >>> pg.init()
 >>> clock = pg.time.Clock()
 >>> screen = pg.display.set_mode((640, 480))

 >>> player = load_image("player1.gif")
 >>> entity = load_image("alien1.gif")
 >>> background = load_image("liquid.bmp")

 >>> # scale the background image so that it fills the window and
 >>> #   successfully overwrites the old sprite position.
 >>> background = pg.transform.scale2x(background)
 >>> background = pg.transform.scale2x(background)

 >>> screen.blit(background, (0, 0))

 >>> objects = []
 >>> p = GameObject(player, 10, 3)
 >>> for x in range(10):
 ...    o = GameObject(entity, x * 40, x)
 ...    objects.append(o)

 >>> # Player controls
 >>> up = 0
 >>> down = 0
 >>> right = 0
 >>> left = 0

 >>> # This is a simple event handler that enables player input.
 >>> while True:
 ...    screen.blit(background, (0, 0))
 ...    for e in pg.event.get():

 ...        # if a key is pressed, make sure that a value is set so that it can move in multiple directions
 ...        if e.type == pg.KEYDOWN:
 ...            if e.key == pg.K_DOWN:
 ...                down = 1
 ...            elif e.key == pg.K_UP:
 ...                up = 1
 ...            elif e.key == pg.K_LEFT:
 ...                left = 1
 ...            elif e.key == pg.K_RIGHT:
 ...                right = 1
 ...            elif e.key == pg.K_ESCAPE:
 ...                exit()

 ...        # if a key is lifted, make sure the value is removed so it can stop moving
 ...        if e.type == pg.KEYUP:
 ...            if e.key == pg.K_DOWN:
 ...                down = 0
 ...            elif e.key == pg.K_UP:
 ...                up = 0
 ...            elif e.key == pg.K_LEFT:
 ...                left = 0
 ...            elif e.key == pg.K_RIGHT:
 ...                right = 0
           
 ...        # quit upon screen exit
 ...        if e.type == pg.QUIT:
 ...            exit()

 ...    # move the player in accordance to the values set in the event handling
 ...    p.move(up,down,left,right)

 ...    for o in objects:
 ...        screen.blit(background, o.pos, o.pos)
 ...    for o in objects:
 ...        o.move()
 ...        screen.blit(o.image, o.pos)
 ...    screen.blit(p.image, p.pos)
 ...    clock.tick(60)
 ...    pg.display.update()

As you can see, a majority of the original code is still in tact. However, our
new and robust event handling takes up a majority of the space. I would personally
make sure that event handling is always done in a function outside of main, especially
with more advanced code. But, with that, our code is complete and fully functional!

You Are On Your Own From Here
-----------------------------

So what would be next on your road to learning? Well first playing around
with this example a bit. The full running version of this example is available
in the pygame examples directory. It is the example named
:func:`moveit.py <pygame.examples.moveit.main>` .
Take a look at the code and play with it, run it, learn it.

Things you may want to work on is maybe having more than one type of object.
Finding a way to cleanly "delete" objects when you don't want to show them
any more. Also updating the display.update() call to pass a list of the areas
on-screen that have changed.

There are also other tutorials and examples in pygame that cover these
issues. So when you're ready to keep learning, keep on reading. :-)

Lastly, you can feel free to come to the pygame mailing list or chatroom
with any questions on this stuff. There's always folks on hand who can help
you out with this sort of business.

Lastly, have fun, that's what games are for!
