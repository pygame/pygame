.. include:: ../common.txt

.. highlight:: c

********************************
  API exported by pygame.mixer
********************************

src_c/mixer.c
=============

Python types and module startup/shutdown functions defined in the
:py:mod:`pygame.mixer` extension module.

Header file: src_c/include/pygame_mixer.h


.. c:type:: pgSoundObject

   The :py:class:`pygame.mixer.Sound` instance C structure.

.. c:var:: PyTypeObject *pgSound_Type

   The :py:class:`pygame.mixer.Sound` Python type.

.. c:function:: PyObject* pgSound_New(Mix_Chunk *chunk)

   Return a new :py:class:`pygame.mixer.Sound` instance for the SDL mixer chunk *chunk*.
   On failure, raise a Python exception and return ``NULL``.

.. c:function:: int pgSound_Check(PyObject \*obj)

   Return true if *obj* is an instance of type :c:data:`pgSound_Type`,
   but not a :c:data:`pgSound_Type` subclass instance.
   A macro.

.. c:function:: Mix_Chunk* pgSound_AsChunk(PyObject *x)

   Return the SDL :c:type:`Mix_Chunk` struct associated with the
   :c:data:`pgSound_Type` instance *x*.
   A macro that does no ``NULL`` or Python type check on *x*.

.. c:type:: pgChannelObject

   The :py:class:`pygame.mixer.Channel` instance C structure.

.. c:var:: PyTypeObject *pgChannel_Type

   The :py:class:`pygame.mixer.Channel` Python type.

.. c:function:: PyObject* pgChannel_New(int channelnum)

   Return a new :py:class:`pygame.mixer.Channel` instance for the SDL mixer
   channel *channelnum*.
   On failure, raise a Python exception and return ``NULL``.

.. c:function:: int pgChannel_Check(PyObject \*obj)

   Return true if *obj* is an instance of type :c:data:`pgChannel_Type`,
   but not a :c:data:`pgChannel_Type` subclass instance.
   A macro.

.. c:function:: Mix_Chunk \*pgChannel_AsInt(PyObject \*x)

   Return the SDL mixer music channel number associated with :c:type:`pgChannel_Type` instance *x*.
   A macro that does no ``NULL`` or Python type check on *x*.

.. c:function:: void pgMixer_AutoInit(void)

   Initialize the :py:mod:`pygame.mixer` module and start the SDL mixer.

.. c:function:: void pgMixer_AutoQuit(void)

   Stop all playing channels and close the SDL mixer.
