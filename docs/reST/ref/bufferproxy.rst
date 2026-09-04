.. include:: common.txt

.. default-domain:: py

:class:`pygame.BufferProxy`
===========================

.<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Retro Highway Dodger</title>
    <style>
        body {
            margin: 0;
            background: #111;
            color: #fff;
            font-family: 'Courier New', Courier, monospace;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
            overflow: hidden;
        }

        #game-container {
            position: relative;
            width: 300px;
            height: 500px;
            background: #333;
            border: 4px solid #fff;
            border-radius: 8px;
            overflow: hidden;
        }

        /* Road lanes animation */
        .lane-mark {
            position: absolute;
            left: 50%;
            transform: translateX(-50%);
            width: 8px;
            height: 40px;
            background: #fff;
        }

        /* Player Car */
        #player {
            position: absolute;
            bottom: 20px;
            left: 125px;
            width: 50px;
            height: 80px;
            background: #ff0055;
            border-radius: 8px;
            box-shadow: 0 0 10px #ff0055;
            transition: left 0.1s ease;
        }

        #player::before {
            content: '';
            position: absolute;
            top: 15px;
            left: 5px;
            width: 40px;
            height: 20px;
            background: #111;
            border-radius: 4px;
        }

        /* Enemy Cars */
        .enemy {
            position: absolute;
            width: 50px;
            height: 80px;
            background: #00e5ff;
            border-radius: 8px;
            box-shadow: 0 0 10px #00e5ff;
        }

        .enemy::after {
            content: '';
            position: absolute;
            bottom: 15px;
            left: 5px;
            width: 40px;
            height: 20px;
            background: #111;
            border-radius: 4px;
        }

        /* UI Elements */
        #score-board {
            margin-bottom: 10px;
            font-size: 20px;
            font-weight: bold;
        }

        #game-over-screen {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.85);
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            gap: 15px;
        }

        button {
            padding: 10px 20px;
            font-size: 16px;
            font-family: inherit;
            background: #ff0055;
            color: #fff;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            box-shadow: 0 0 8px #ff0055;
        }

        button:hover {
            background: #ff3377;
        }

        .hidden {
            display: none !important;
        }
    </style>
</head>
<body>

    <div id="score-board">SCORE: <span id="score">0</span></div>

    <div id="game-container">
        <div class="lane-mark" style="top: -40px;"></div>
        <div class="lane-mark" style="top: 80px;"></div>
        <div class="lane-mark" style="top: 200px;"></div>
        <div class="lane-mark" style="top: 320px;"></div>
        <div class="lane-mark" style="top: 440px;"></div>

        <div id="player"></div>

        <div id="game-over-screen">
            <h1>CRASHED!</h1>
            <p>Final Score: <span id="final-score">0</span></p>
            <button onclick="startGame()">PLAY AGAIN</button>
        </div>
    </div>
.. class:: BufferProxy

   | :sl:`pygame object to export a surface buffer through an array protocol`
   | :sg:`BufferProxy(<parent>) -> BufferProxy`

   :class:`BufferProxy` is a pygame support type, designed as the return value
   of the :meth:`Surface.get_buffer` and :meth:`Surface.get_view` methods.
   For all Python versions a :class:`BufferProxy` object exports a C struct
   and Python level array interface on behalf of its parent object's buffer.
   A new buffer interface is also exported.
   In pygame, :class:`BufferProxy` is key to implementing the
   :mod:`pygame.surfarray` module.

   :class:`BufferProxy` instances can be created directly from Python code,
   either for a parent that exports an interface, or from a Python ``dict``
   describing an object's buffer layout. The dict entries are based on the
   Python level array interface mapping. The following keys are recognized:

      ``"shape"`` : tuple
         The length of each array dimension as a tuple of integers. The
         length of the tuple is the number of dimensions in the array.

      ``"typestr"`` : string
         The array element type as a length 3 string. The first character
         gives byteorder, '<' for little-endian, '>' for big-endian, and
         '\|' for not applicable. The second character is the element type,
         'i' for signed integer, 'u' for unsigned integer, 'f' for floating
         point, and 'V' for an chunk of bytes. The third character gives the
         bytesize of the element, from '1' to '9' bytes. So, for example,
         "<u4" is an unsigned 4 byte little-endian integer, such as a
         32 bit pixel on a PC, while "\|V3" would represent a 24 bit pixel,
         which has no integer equivalent.

      ``"data"`` : tuple
         The physical buffer start address and a read-only flag as a length
         2 tuple. The address is an integer value, while the read-only flag
         is a bool—``False`` for writable, ``True`` for read-only.

      ``"strides"`` : tuple : (optional)
         Array stride information as a tuple of integers. It is required
	 only of non C-contiguous arrays. The tuple length must match
	 that of ``"shape"``.

      ``"parent"`` : object : (optional)
         The exporting object. It can be used to keep the parent object
         alive while its buffer is visible.

      ``"before"`` : callable : (optional)
         Callback invoked when the :class:`BufferProxy` instance
         exports the buffer. The callback is given one argument, the
	 ``"parent"`` object if given, otherwise ``None``.
         The callback is useful for setting a lock on the parent.

      ``"after"`` : callable : (optional)
         Callback invoked when an exported buffer is released.
         The callback is passed on argument, the ``"parent"`` object if given,
         otherwise None. The callback is useful for releasing a lock on the
         parent.
      
   The BufferProxy class supports subclassing, instance variables, and weak
   references.

   .. versionadded:: 1.8.0
   .. versionextended:: 1.9.2

   .. attribute:: parent

      | :sl:`Return wrapped exporting object.`
      | :sg:`parent -> Surface`
      | :sg:`parent -> <parent>`

      The :class:`Surface` which returned the :class:`BufferProxy` object or
      the object passed to a :class:`BufferProxy` call.

   .. attribute:: length

      | :sl:`The size, in bytes, of the exported buffer.`
      | :sg:`length -> int`

      The number of valid bytes of data exported. For discontinuous data,
      that is data which is not a single block of memory, the bytes within
      the gaps are excluded from the count. This property is equivalent to
      the ``Py_buffer`` C struct ``len`` field.

   .. attribute:: raw

      | :sl:`A copy of the exported buffer as a single block of bytes.`
      | :sg:`raw -> bytes`

      The buffer data as a ``str``/``bytes`` object.
      Any gaps in the exported data are removed.

   .. method:: write

      | :sl:`Write raw bytes to object buffer.`
      | :sg:`write(buffer, offset=0)`

      Overwrite bytes in the parent object's data. The data must be C or F
      contiguous, otherwise a ValueError is raised. Argument `buffer` is a
      ``str``/``bytes`` object. An optional offset gives a
      start position, in bytes, within the buffer where overwriting begins.
      If the offset is negative or greater that or equal to the buffer proxy's
      :attr:`length` value, an ``IndexException`` is raised.
      If ``len(buffer) > proxy.length + offset``, a ``ValueError`` is raised.
