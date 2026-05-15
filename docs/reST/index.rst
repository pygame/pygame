mport React, { useEffect, useRef, useState } from "react";
      if (["Space", "ArrowUp", "KeyW"].includes(e.code)) e.preventDefault();
      if (!started) resetGame();
      else if (gameOver && e.code === "Space") resetGame();
    };
    const handleUp = (e) => {
      keysRef.current[e.code] = false;
    };
    window.addEventListener("keydown", handleDown);
    window.addEventListener("keyup", handleUp);
    return () => {
      window.removeEventListener("keydown", handleDown);
      window.removeEventListener("keyup", handleUp);
      cancelAnimationFrame(animationRef.current);
    };
  }, [started, gameOver]);

  useEffect(() => {
    if (started && !gameOver) {
      cancelAnimationFrame(animationRef.current);
      animationRef.current = requestAnimationFrame(updateAndDraw);
    }
    return () => cancelAnimationFrame(animationRef.current);
  }, [started, gameOver]);

  return (
    <div className="min-h-screen bg-emerald-50 flex items-center justify-center p-4">
      <Card className="w-full max-w-5xl rounded-2xl shadow-xl border-emerald-200">
        <CardContent className="p-5">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3 mb-4">
            <div>
              <h1 className="text-3xl font-bold text-emerald-900">Monkey Banana Run</h1>
              <p className="text-emerald-700">Jump over rocks and collect bananas.</p>
            </div>
            <div className="flex gap-3 items-center">
              <div className="px-4 py-2 rounded-xl bg-white shadow text-emerald-900 font-semibold">Score: {score}</div>
              <div className="px-4 py-2 rounded-xl bg-white shadow text-yellow-700 font-semibold">🍌 {bananas}</div>
              <Button onClick={resetGame} className="rounded-xl bg-emerald-700 hover:bg-emerald-800">
                {started ? "Restart" : "Start"}
              </Button>
            </div>
          </div>

          <div className="relative overflow-hidden rounded-2xl shadow-inner border-4 border-emerald-300 bg-sky-100">
            <canvas
              ref={canvasRef}
              width={WIDTH}
              height={HEIGHT}
              onMouseDown={() => (started && !gameOver ? jump() : resetGame())}
              className="w-full cursor-pointer block"
            />

            {!started && (
              <div className="absolute inset-0 flex items-center justify-center bg-black/25 text-white text-center p-6">
                <div className="bg-black/45 rounded-2xl p-6">
                  <h2 className="text-3xl font-bold mb-2">Ready?</h2>
                  <p className="mb-4">Press Space / ↑ / W or click to jump.</p>
                  <Button onClick={resetGame} className="rounded-xl bg-yellow-500 hover:bg-yellow-600 text-black">
                    Play Game
                  </Button>
                </div>
              </div>
            )}

            {gameOver && (
              <div className="absolute inset-0 flex items-center justify-center bg-black/40 text-white text-center p-6">
                <div className="bg-slate-950/80 rounded-2xl p-6 shadow-2xl">
                  <h2 className="text-4xl font-bold mb-2">Game Over</h2>
                  <p className="text-lg mb-4">Final Score: {score} | Bananas: {bananas}</p>
                  <Button onClick={resetGame} className="rounded-xl bg-yellow-500 hover:bg-yellow-600 text-black">
                    Play Again
                  </Button>
                </div>
              </div>
            )}
          </div>

          <p className="mt-4 text-sm text-emerald-700 text-center">
            Controls: Space / Arrow Up / W / Mouse click / Tap screen
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
Pygame Front Page
=================

.. toctree::
   :maxdepth: 2
   :glob:
   :hidden:

   ref/*
   tut/*
   tut/en/**/*
   tut/ko/**/*
   c_api
   filepaths
   logos

Quick start
-----------

Welcome to pygame! Once you've got pygame installed (:code:`pip install pygame` or
:code:`pip3 install pygame` for most people), the next question is how to get a game
loop running. Pygame, unlike some other libraries, gives you full control of program
execution. That freedom means it is easy to mess up in your initial steps.

Here is a good example of a basic setup (opens the window, updates the screen, and handles events)--

.. literalinclude:: ref/code_examples/base_script.py

Here is a slightly more fleshed out example, which shows you how to move something
(a circle in this case) around on screen--

.. literalinclude:: ref/code_examples/base_script_example.py

For more in depth reference, check out the :ref:`tutorials-reference-label`
section below, check out a video tutorial (`I'm a fan of this one
<https://www.youtube.com/watch?v=AY9MnQ4x3zk>`_), or reference the API
documentation by module.

Documents
---------

`Readme`_
  Basic information about pygame: what it is, who is involved, and where to find it.

`Install`_
  Steps needed to compile pygame on several platforms.
  Also help on finding and installing prebuilt binaries for your system.

:doc:`filepaths`
  How pygame handles file system paths.

:doc:`Pygame Logos <logos>`
   The logos of Pygame in different resolutions.


`LGPL License`_
  This is the license pygame is distributed under.
  It provides for pygame to be distributed with open source and commercial software.
  Generally, if pygame is not changed, it can be used with any type of program.

.. _tutorials-reference-label:

Tutorials
---------

:doc:`Introduction to Pygame <tut/PygameIntro>`
  An introduction to the basics of pygame.
  This is written for users of Python and appeared in volume two of the Py magazine.

:doc:`Import and Initialize <tut/ImportInit>`
  The beginning steps on importing and initializing pygame.
  The pygame package is made of several modules.
  Some modules are not included on all platforms.

:doc:`How do I move an Image? <tut/MoveIt>`
  A basic tutorial that covers the concepts behind 2D computer animation.
  Information about drawing and clearing objects to make them appear animated.

:doc:`Chimp Tutorial, Line by Line <tut/ChimpLineByLine>`
  The pygame examples include a simple program with an interactive fist and a chimpanzee.
  This was inspired by the annoying flash banner of the early 2000s.
  This tutorial examines every line of code used in the example.

:doc:`Sprite Module Introduction <tut/SpriteIntro>`
  Pygame includes a higher level sprite module to help organize games.
  The sprite module includes several classes that help manage details found in almost all games types.
  The Sprite classes are a bit more advanced than the regular pygame modules,
  and need more understanding to be properly used.

:doc:`Surfarray Introduction <tut/SurfarrayIntro>`
  Pygame used the NumPy python module to allow efficient per pixel effects on images.
  Using the surface arrays is an advanced feature that allows custom effects and filters.
  This also examines some of the simple effects from the pygame example, arraydemo.py.

:doc:`Camera Module Introduction <tut/CameraIntro>`
  Pygame, as of 1.9, has a camera module that allows you to capture images,
  watch live streams, and do some basic computer vision.
  This tutorial covers those use cases.

:doc:`Newbie Guide <tut/newbieguide>`
  A list of thirteen helpful tips for people to get comfortable using pygame.

:doc:`Making Games Tutorial <tut/MakeGames>`
  A large tutorial that covers the bigger topics needed to create an entire game.

:doc:`Display Modes <tut/DisplayModes>`
  Getting a display surface for the screen.

:doc:`한국어 튜토리얼 (Korean Tutorial) <tut/ko/빨간블록 검은블록/개요>`
  빨간블록 검은블록


Reference
---------

:ref:`genindex`
  A list of all functions, classes, and methods in the pygame package.

:doc:`ref/bufferproxy`
  An array protocol view of surface pixels

:doc:`ref/color`
  Color representation.

:doc:`ref/cursors`
  Loading and compiling cursor images.

:doc:`ref/display`
  Configure the display surface.

:doc:`ref/draw`
  Drawing simple shapes like lines and ellipses to surfaces.

:doc:`ref/event`
  Manage the incoming events from various input devices and the windowing platform.

:doc:`ref/examples`
  Various programs demonstrating the use of individual pygame modules.

:doc:`ref/font`
  Loading and rendering TrueType fonts.

:doc:`ref/freetype`
  Enhanced pygame module for loading and rendering font faces.

:doc:`ref/gfxdraw`
  Anti-aliasing draw functions.

:doc:`ref/image`
  Loading, saving, and transferring of surfaces.

:doc:`ref/joystick`
  Manage the joystick devices.

:doc:`ref/key`
  Manage the keyboard device.

:doc:`ref/locals`
  Pygame constants.

:doc:`ref/mixer`
  Load and play sounds

:doc:`ref/mouse`
  Manage the mouse device and display.

:doc:`ref/music`
  Play streaming music tracks.

:doc:`ref/pygame`
  Top level functions to manage pygame.

:doc:`ref/pixelarray`
  Manipulate image pixel data.

:doc:`ref/rect`
  Flexible container for a rectangle.

:doc:`ref/scrap`
  Native clipboard access.

:doc:`ref/sndarray`
  Manipulate sound sample data.

:doc:`ref/sprite`
  Higher level objects to represent game images.

:doc:`ref/surface`
  Objects for images and the screen.

:doc:`ref/surfarray`
  Manipulate image pixel data.

:doc:`ref/tests`
  Test pygame.

:doc:`ref/time`
  Manage timing and framerate.

:doc:`ref/transform`
  Resize and move images.

:doc:`pygame C API <c_api>`
  The C api shared amongst pygame extension modules.

:ref:`search`
  Search pygame documents by keyword.

.. _Readme: ../wiki/about

.. _Install: ../wiki/GettingStarted#Pygame%20Installation

.. _LGPL License: LGPL.txt
