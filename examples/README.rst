These examples should help get you started with pygame.  Here is a
brief rundown of what you get.  The source code for all examples
is in the public domain.  Feel free to use for your own projects.

aliens.py
   This started off as a port of the SDL demonstration, Aliens.
   Now it has evolved into something sort of resembling fun.
   This demonstrates a lot of different uses of sprites and
   optimized blitting.  Also transparancy, colorkeys, fonts, sound,
   music, joystick, and more.  (PS, my high score is 117! goodluck)

arraydemo.py
   Another example filled with various surfarray effects.
   It requires the surfarray and image modules to be installed.
   This little demo can also make a good starting point for any of
   your own tests with surfarray

audiocapture.py
   Record sound from a microphone, and play back the recorded sound.

blend_fill.py
   BLEND_ing colors in different ways with Surface.fill().

blit_blends.py
   BLEND_ing colors Surface.blit().

camera.py
   Basic image capturing and display using pygame.camera

cursors.py
   Make custom cursors :)

dropevent.py
   Drag and drop files.  Using the following events.
   DROPBEGIN, DROPCOMPLETE, DROPTEXT, DROPFILE

eventlist.py
   Learn about pygame events and input.
   Watch the events fly by.  Click the mouse, and see the mouse
   event come up.  Press a keyboard key, and see the key up event.

fastevents.py
   Posting events from other threads? Check this out.

font_viewer.py
   Display all available fonts in a scrolling window.

fonty.py
   Super quick, super simple application demonstrating
   the different ways to render fonts with the font module

freetype_misc.py
   FreeType is a world famous font project.

glcube.py
   Using PyOpenGL and Pygame, this creates a spinning 3D multicolored cube.

headless_no_windows_needed.py
   For using pygame in scripts.

liquid.py
   This example was created in a quick comparison with the
   BlitzBasic gaming language.  Nonetheless, it demonstrates a quick
   8-bit setup (with colormap).

mask.py
   Single bit pixel manipulation.  Fast for collision detection,
   and also good for computer vision.

midi.py
   For connecting pygame to musical equipment.

moveit.py
   A very simple example of moving stuff.

music_drop_fade.py
   Fade in and play music from a list while observing
   several events.  Uses fade_ms added in pygame2, as well as set_endevent,
   set_volume, drag and drop events, and the scrap module.

overlay.py
   An old way of displaying video content.

pixelarray.py
   Process whole arrays of pixels at a time.
   Like numpy, but for pixels, and also built into pygame.

playmus.py
   Simple music playing example.

prevent_display_stretching.py
   A windows specific example.

scaletest.py
   Showing how to scale Surfaces.

scrap_clipboard.py
   A simple demonstration example for the clipboard support.

setmodescale.py
   SCALED allows you to work in 320x200 and have it show up big.
   It handles mouse scaling and selection of a good sized window depending
   on the display.

sound.py
   Extremely basic testing of the mixer module.  Load a
   sound and play it.  All from the command shell, no graphics.

sound_array_demos.py
   Echo, delay and other array based processing of sounds.

sprite_texture.py
   Shows how to use hardware Image Textures with pygame.sprite.

stars.py
   A simple starfield example.  You can change the center of
   perspective by leftclicking the mouse on the screen.

testsprite.py
   More of a test example.  If you're interested in how to use sprites,
   then check out the aliens.py example instead.

textinput.py
   A little "console" where you can write in text.
   Shows how to use the TEXTEDITING and TEXTINPUT events.

vgrade.py
   Demonstrates creating a vertical gradient with
   Numpy.  The app will create a new gradient every half
   second and report the time needed to create and display the
   image.  If you're not prepared to start working with the
   Numpy arrays, don't worry about the source for this one :]

video.py
   It explores some new video APIs in pygame 2.
   Including multiple windows, Textures, and such.

data/
   Directory with the resources for the examples.

There's LOTS of examples on the pygame website, and on places like github.

We're always on the lookout for more examples and/or example
requests.  Code like this is probably the best way to start
getting involved with Python gaming.
