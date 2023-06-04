.. include:: common.txt

:mod:`pygame.display`
=====================

.. module:: pygame.display
   :synopsis: módulo de pygame para controlar la ventana de visualización y pantalla

| :sl:`pygame module to control the display window and screen`

Este módulo ofrence control sobre la visualización de pygame. Pygame tiene una 
única Surface (superficie) de display que puede estar contenida en una ventana o
ejecutarse en pantalla completa. Una vez que creás la visualización, la tratás 
como una surface normal. Los cambios no son visibles de inmediato en la pantalla;
debés elegir una de las dos funciones de volteo para actualizar la visualización 
real. 

El origen de la visualización, donde x=0 y y=0, es la esquina superior 
izquierda de la pantalla. Ambos ejes aumentan positivamente hacia la 
esquina inferior derecha de la pantalla. 

La visualización de pygame en realidad se puede inicializar en varios modos. 
De forma predeterminada, la visualización es un búfer básico controlado por 
software. Podés solicitar módulos especiales como escalado automático o 
soporte de OpenGL. Estos se controlan mediante indicadores que se pasan a 
``pygame.display.set_mode()``.

Pygame solo puede tener una única visualización activa por vez. Crear una 
nueva con ``pygame.display.set_mode()`` cerrará la anterior. Para detectar 
el número y tamaño de las pantallas conectadas, podés usar 
``pygame.display.get_desktop_sizes`` y luego seleccionar el tamaño de 
ventana adecuado y el índice de visualización para pasar a 
``pygame.display.set_mode()``.

Para una compatibilidad hacia atrás ``pygame.display`` permite precisar 
controles sobre el formato de píxeles o las resoluciones de visualización. 
Esto solía ser necesario con antiguas tarjetas gráficas y pantallas CRT, 
pero generalmente ya no es necesario. Utilizá las funciones 
``pygame.display.mode_ok()``, ``pygame.display.list_modes()``, and 
``pygame.display.Info()`` para obtener información detallada sobre la 
visualización.

Una vez creada la superficie (surface) de visualización, las funciones de este 
módulo afectan a la única visualización existente. La superficie se vuelve 
inválida si se desinicializa el módulo. Si se establece un nuevo modo de 
visualización, la superficie existente cambiará automáticamente para 
operar en la nueva visualización.

Cuando se establece el modo de visualización, se colocan varios eventos en la 
cola de eventos de pygame. Se envía ``pygame.QUIT``cuando el usuario solicita 
cerrar el programa. La ventana recibirá eventos ``pygame.ACTIVEEVENT`` a 
medida que la visualización gane y pierda el enfoque de entrada. Si la 
visualización se establece con la marca ``pygame.RESIZABLE``, se enviarán 
eventos ``pygame.VIDEORESIZE`` cuando el usuario ajuste las dimensiones de 
la ventana. Las pantallas de hardware que dibujan directamente en la pantalla 
recibirán eventos ``pygame.VIDEOEXPOSE`` cuando se deban volver a dibujar 
partes de la ventana.

En pygame 2.0.1 se introdujo una nueva API de eventos de ventana. Consulta la 
documentación del módulo de eventos para obtener más información al respecto.

Algunos entornos de visualización tienen una opción para estirar automáticamente 
todas las ventanas. Cuando esta opción está habilitada, este estiramiento 
automático distoriona la apariencia de la ventana de pygame. En el directorio 
de ejemplos de pygame, hay un código de ejemplo (prevent_display_stretching.py) 
que muestra cómo deshabilitar este estiramiento automático de la visualización 
de pygame en Microsoft Windows (se requiere vista o una versión más nueva).

.. function:: init

   | :sl:`Initialize the display module`
   | :sg:`init() -> None`

   Inicializa el módulo de visualización de pygame. El módulo de visualización 
   no puede hacer nada hasta que se inicialice. Esto generalmente se maneja 
   automáticamente cuando llamas a la función de nivel superior ``pygame.init()``.

   Pygame seleccionará uno de varios backend internos de visualización cuando se 
   inicialice. El modo de visualización se elegirá según la plataforma y los 
   permisos del usuario actual. Antes de inicializar el módulo de visualización, 
   se puede establecer la variable de entorno ``SDL_VIDEODRIVER`` para controlar 
   qué backend se utiliza. Aquí se enumeran los sistemas con múltiples opciones.
  ::

      Windows : windib, directx
      Unix    : x11, dga, fbcon, directfb, ggi, vgl, svgalib, aalib

   En algunas plataformas es posible incrustar la visualización de pygame en 
   una ventana existente. Para hacer esto, la variable de entorno ``SDL_WINDOWID`` 
   debe establecerse como una cadena que contenga el ID o el identificador de 
   la ventana. La variable de entorno se verifica cuando se inicializa la 
   visualización de pygame. Ten en cuenta que puede haber efectos secundarios 
   extraños al ejecutar en una visualización incrustada.

   No tiene ningún efecto perjudicial llamar a esto más de una vez; las llamadas 
   repetidas no tienen efecto.
   

   .. ## pygame.display.init ##

.. function:: quit

   | :sl:`Uninitialize the display module`
   | :sg:`quit() -> None`

   Esto cerrará por completo el módulo de visualización. Esto significa que se 
   cerrarán todas las visualizaciones activas. Esto también se manejará automáticamente 
   cuando el programa finalice.

   No tiene ningún efecto perjudicial llamar a esto más de una vez; las llamadas 
   repetidas no tienen efecto.

   .. ## pygame.display.quit ##

.. function:: get_init

   | :sl:`Returns True if the display module has been initialized`
   | :sg:`get_init() -> bool`

   Deuvelve True (verdad) si el módulo :mod:`pygame.display` está actualmente inicializado.

   .. ## pygame.display.get_init ##

.. function:: set_mode

   | :sl:`Initialize a window or screen for display`
   | :sg:`set_mode(size=(0, 0), flags=0, depth=0, display=0, vsync=0) -> Surface`

   Esta función creará una superficie de visualización. Los argumentos que 
   se pasan son solicitudes de un tipo de visualización. La visualización 
   creada será la mejor coincidencia posible compatible con el sistema. 
   
   Hay que tener en cuenta que llamar a esta función inicializa implícitamente 
   ``pygame.display``, si no se había inicializado antes.

   El argumento de tamaño es un par de números que representan el ancho y la 
   altura. El argumento de banderas es una colección de opciones adicionales. 
   El argumento de profundidad representa el número de bits que se utilizarán 
   para el color.

   La suprficie que se devuelve se puede dibujar como una superficie regular,
   pero los cambios se verán eventualmente en el monitor.

   Si no se pasa ningún tamaño o se establece como ``(0, 0)`` y pygame utiliza 
   la versión 1.2.10 o superior de ``SDL``, la superficie creada tendrá el 
   mismo tamaño que la resolución de pantalla actual. Si solo el ancho o la 
   altura se establecen en ``0``, la superficie tendrá el mismo ancho o altura 
   que la resolución de pantalla. El uso de una versión de ``SDL`` anterior a 
   1.2.10 generará una excepción.

   Generalmente es mejor no pasar el argumento de profundidad. Se establecerá 
   por defecto en la mejor y más rápida profundidad de color para el sistema. 
   Si tu juego requiere un formato de color específico, puedes controlar la 
   profundidad con este argumento. Pygame emulará una profundidad de color no 
   disponible, lo cual puede ser lento.
   
   Al solicitar modos de visualización de pantalla completa, a veces no se 
   puede encontrar una coincidencia exacta para el tamaño solicitado. En estas 
   situaciones, pygame seleccionará la coincidencia compatible más cercana. 
   La superficie devuelta siempre coincidirá con el tamaño solicitado.

   En pantallas de alta resolución (4k, 1080p) y juegos de gráficos pequeños 
   (640x480), la visualización aparece muy pequeña y no se puede jugar. 
   SCALED escala la ventana para vos. El juego piensa que es una ventana de 
   640x480, pero en realidad puede ser más grande. Los eventos del mouse se 
   escalan automáticamente, por lo que tu juego no necesita hacerlo. Hay que
   tener en cuenta que SCALED se considera una API experimental y puede 
   cambiar en futuras versiones.

   El argumento de banderas (flags argument) controla qué tipo de 
   visualización deseas. Hay varios para elegir, e incluso puedes combinar 
   múltiples tipos usando el operador OR bitwise (el carácter "|"). 
   Aquí están las banderas de visualización entre las que puedes elegir:

   ::

      pygame.FULLSCREEN    crea una visualización a pantalla completa
      pygame.DOUBLEBUF     solo aplicable con OPENGL
      pygame.HWSURFACE     (obsoleta en pygame 2) acelerado por hardware, solo en FULLSCREEN
      pygame.OPENGL        crea una visualización renderizable en OpenGL
      pygame.RESIZABLE     la ventana de visualización debe ser redimensionable
      pygame.NOFRAME       la ventana de visualización no tendrá borde ni controles
      pygame.SCALED        la resolución depende del tamaño del escritorio y escala los gráficos
      pygame.SHOWN         la ventana se abre en modo visible (por defecto)
      pygame.HIDDEN        la ventana se abre en modo oculto


   .. versionadded:: 2.0.0 ``SCALED``, ``SHOWN`` and ``HIDDEN``

   Al establecer el parámetro "vsync" en "1", es posible obtener una 
   visualización con sincronización vertical, pero no se garantiza obtener una. 
   La solicitud solo funciona para llamadas a ``set_mode()`` con las banderas 
   ``pygame.OPENGL`` o ``pygame.SCALED`` establecidas, y aún así no está 
   garantizado incluso con una de ellas establecida. Lo que obtengas depende de 
   la configuración de hardware y controlador del sistema en el que se ejecute 
   pygame. Aquí tienes un ejemplo de uso de una llamada a ``set_mode()`` que puede
   darte una visualización con sincronización vertical:
   ::

     flags = pygame.OPENGL | pygame.FULLSCREEN
     window_surface = pygame.display.set_mode((1920, 1080), flags, vsync=1)

   El comportamiento de Vsync se considera experimental y puede cambiar en futuras versiones.
   

   .. versionadded:: 2.0.0 ``vsync``

   Basic example:

   ::

        # Abrir ventana en la pantalla
        screen_width=700
        screen_height=400
        screen=pygame.display.set_mode([screen_width, screen_height])

   El índice de visualización ``0`` significa que se utiliza la visualización 
   predeterminada. Si no se proporciona un argumento de índice de visualización, 
   la visualización predeterminada puede ser reemplazada con una variable de 
   entorno.

   .. versionchanged:: 1.9.5 ``display`` argument added

   .. versionchanged:: 2.1.3
      pygame now ensures that subsequent calls to this function clears the
      window to black. On older versions, this was an implementation detail
      on the major platforms this function was tested with.

   .. ## pygame.display.set_mode ##

.. function:: get_surface

   | :sl:`Get a reference to the currently set display surface`
   | :sg:`get_surface() -> Surface`

   Return a reference to the currently set display Surface. If no display mode
   has been set this will return None.

   .. ## pygame.display.get_surface ##

.. function:: flip

   | :sl:`Update the full display Surface to the screen`
   | :sg:`flip() -> None`

   This will update the contents of the entire display. If your display mode is
   using the flags ``pygame.HWSURFACE`` and ``pygame.DOUBLEBUF`` on pygame 1,
   this will wait for a vertical retrace and swap the surfaces.

   When using an ``pygame.OPENGL`` display mode this will perform a gl buffer
   swap.

   .. ## pygame.display.flip ##

.. function:: update

   | :sl:`Update portions of the screen for software displays`
   | :sg:`update(rectangle=None) -> None`
   | :sg:`update(rectangle_list) -> None`

   This function is like an optimized version of ``pygame.display.flip()`` for
   software displays. It allows only a portion of the screen to be updated,
   instead of the entire area. If no argument is passed it updates the entire
   Surface area like ``pygame.display.flip()``.

   Note that calling ``display.update(None)`` means no part of the window is
   updated. Whereas ``display.update()`` means the whole window is updated.

   You can pass the function a single rectangle, or a sequence of rectangles.
   It is more efficient to pass many rectangles at once than to call update
   multiple times with single or a partial list of rectangles. If passing a
   sequence of rectangles it is safe to include None values in the list, which
   will be skipped.

   This call cannot be used on ``pygame.OPENGL`` displays and will generate an
   exception.

   .. ## pygame.display.update ##

.. function:: get_driver

   | :sl:`Get the name of the pygame display backend`
   | :sg:`get_driver() -> name`

   Pygame chooses one of many available display backends when it is
   initialized. This returns the internal name used for the display backend.
   This can be used to provide limited information about what display
   capabilities might be accelerated. See the ``SDL_VIDEODRIVER`` flags in
   ``pygame.display.set_mode()`` to see some of the common options.

   .. ## pygame.display.get_driver ##

.. function:: Info

   | :sl:`Create a video display information object`
   | :sg:`Info() -> VideoInfo`

   Creates a simple object containing several attributes to describe the
   current graphics environment. If this is called before
   ``pygame.display.set_mode()`` some platforms can provide information about
   the default display mode. This can also be called after setting the display
   mode to verify specific display options were satisfied. The VidInfo object
   has several attributes:

   ::

     hw:         1 if the display is hardware accelerated
     wm:         1 if windowed display modes can be used
     video_mem:  The megabytes of video memory on the display. This is 0 if
                 unknown
     bitsize:    Number of bits used to store each pixel
     bytesize:   Number of bytes used to store each pixel
     masks:      Four values used to pack RGBA values into pixels
     shifts:     Four values used to pack RGBA values into pixels
     losses:     Four values used to pack RGBA values into pixels
     blit_hw:    1 if hardware Surface blitting is accelerated
     blit_hw_CC: 1 if hardware Surface colorkey blitting is accelerated
     blit_hw_A:  1 if hardware Surface pixel alpha blitting is accelerated
     blit_sw:    1 if software Surface blitting is accelerated
     blit_sw_CC: 1 if software Surface colorkey blitting is accelerated
     blit_sw_A:  1 if software Surface pixel alpha blitting is accelerated
     current_h, current_w:  Height and width of the current video mode, or
                 of the desktop mode if called before the display.set_mode
                 is called. (current_h, current_w are available since
                 SDL 1.2.10, and pygame 1.8.0). They are -1 on error, or if
                 an old SDL is being used.

   .. ## pygame.display.Info ##

.. function:: get_wm_info

   | :sl:`Get information about the current windowing system`
   | :sg:`get_wm_info() -> dict`

   Creates a dictionary filled with string keys. The strings and values are
   arbitrarily created by the system. Some systems may have no information and
   an empty dictionary will be returned. Most platforms will return a "window"
   key with the value set to the system id for the current display.

   .. versionadded:: 1.7.1

   .. ## pygame.display.get_wm_info ##

.. function:: get_desktop_sizes

   | :sl:`Get sizes of active desktops`
   | :sg:`get_desktop_sizes() -> list`

   This function returns the sizes of the currently configured
   virtual desktops as a list of (x, y) tuples of integers.

   The length of the list is not the same as the number of attached monitors,
   as a desktop can be mirrored across multiple monitors. The desktop sizes
   do not indicate the maximum monitor resolutions supported by the hardware,
   but the desktop size configured in the operating system.

   In order to fit windows into the desktop as it is currently configured, and
   to respect the resolution configured by the operating system in fullscreen
   mode, this function *should* be used to replace many use cases of
   ``pygame.display.list_modes()`` whenever applicable.

   .. versionadded:: 2.0.0

.. function:: list_modes

   | :sl:`Get list of available fullscreen modes`
   | :sg:`list_modes(depth=0, flags=pygame.FULLSCREEN, display=0) -> list`

   This function returns a list of possible sizes for a specified color
   depth. The return value will be an empty list if no display modes are
   available with the given arguments. A return value of ``-1`` means that
   any requested size should work (this is likely the case for windowed
   modes). Mode sizes are sorted from biggest to smallest.

   If depth is ``0``, the current/best color depth for the display is used.
   The flags defaults to ``pygame.FULLSCREEN``, but you may need to add
   additional flags for specific fullscreen modes.

   The display index ``0`` means the default display is used.

   Since pygame 2.0, ``pygame.display.get_desktop_sizes()`` has taken over
   some use cases from ``pygame.display.list_modes()``:

   To find a suitable size for non-fullscreen windows, it is preferable to
   use ``pygame.display.get_desktop_sizes()`` to get the size of the *current*
   desktop, and to then choose a smaller window size. This way, the window is
   guaranteed to fit, even when the monitor is configured to a lower resolution
   than the maximum supported by the hardware.

   To avoid changing the physical monitor resolution, it is also preferable to
   use ``pygame.display.get_desktop_sizes()`` to determine the fullscreen
   resolution. Developers are strongly advised to default to the current
   physical monitor resolution unless the user explicitly requests a different
   one (e.g. in an options menu or configuration file).

   .. versionchanged:: 1.9.5 ``display`` argument added

   .. ## pygame.display.list_modes ##

.. function:: mode_ok

   | :sl:`Pick the best color depth for a display mode`
   | :sg:`mode_ok(size, flags=0, depth=0, display=0) -> depth`

   This function uses the same arguments as ``pygame.display.set_mode()``. It
   is used to determine if a requested display mode is available. It will
   return ``0`` if the display mode cannot be set. Otherwise it will return a
   pixel depth that best matches the display asked for.

   Usually the depth argument is not passed, but some platforms can support
   multiple display depths. If passed it will hint to which depth is a better
   match.

   The function will return ``0`` if the passed display flags cannot be set.

   The display index ``0`` means the default display is used.

   .. versionchanged:: 1.9.5 ``display`` argument added

   .. ## pygame.display.mode_ok ##

.. function:: gl_get_attribute

   | :sl:`Get the value for an OpenGL flag for the current display`
   | :sg:`gl_get_attribute(flag) -> value`

   After calling ``pygame.display.set_mode()`` with the ``pygame.OPENGL`` flag,
   it is a good idea to check the value of any requested OpenGL attributes. See
   ``pygame.display.gl_set_attribute()`` for a list of valid flags.

   .. ## pygame.display.gl_get_attribute ##

.. function:: gl_set_attribute

   | :sl:`Request an OpenGL display attribute for the display mode`
   | :sg:`gl_set_attribute(flag, value) -> None`

   When calling ``pygame.display.set_mode()`` with the ``pygame.OPENGL`` flag,
   Pygame automatically handles setting the OpenGL attributes like color and
   double-buffering. OpenGL offers several other attributes you may want control
   over. Pass one of these attributes as the flag, and its appropriate value.
   This must be called before ``pygame.display.set_mode()``.

   Many settings are the requested minimum. Creating a window with an OpenGL context
   will fail if OpenGL cannot provide the requested attribute, but it may for example
   give you a stencil buffer even if you request none, or it may give you a larger
   one than requested.

   The ``OPENGL`` flags are:

   ::

     GL_ALPHA_SIZE, GL_DEPTH_SIZE, GL_STENCIL_SIZE, GL_ACCUM_RED_SIZE,
     GL_ACCUM_GREEN_SIZE,  GL_ACCUM_BLUE_SIZE, GL_ACCUM_ALPHA_SIZE,
     GL_MULTISAMPLEBUFFERS, GL_MULTISAMPLESAMPLES, GL_STEREO

   :const:`GL_MULTISAMPLEBUFFERS`

     Whether to enable multisampling anti-aliasing.
     Defaults to 0 (disabled).

     Set ``GL_MULTISAMPLESAMPLES`` to a value
     above 0 to control the amount of anti-aliasing.
     A typical value is 2 or 3.

   :const:`GL_STENCIL_SIZE`

     Minimum bit size of the stencil buffer. Defaults to 0.

   :const:`GL_DEPTH_SIZE`

     Minimum bit size of the depth buffer. Defaults to 16.

   :const:`GL_STEREO`

     1 enables stereo 3D. Defaults to 0.

   :const:`GL_BUFFER_SIZE`

     Minimum bit size of the frame buffer. Defaults to 0.

   .. versionadded:: 2.0.0 Additional attributes:

   ::

     GL_ACCELERATED_VISUAL,
     GL_CONTEXT_MAJOR_VERSION, GL_CONTEXT_MINOR_VERSION,
     GL_CONTEXT_FLAGS, GL_CONTEXT_PROFILE_MASK,
     GL_SHARE_WITH_CURRENT_CONTEXT,
     GL_CONTEXT_RELEASE_BEHAVIOR,
     GL_FRAMEBUFFER_SRGB_CAPABLE

   :const:`GL_CONTEXT_PROFILE_MASK`

     Sets the OpenGL profile to one of these values:

     ::

       GL_CONTEXT_PROFILE_CORE             disable deprecated features
       GL_CONTEXT_PROFILE_COMPATIBILITY    allow deprecated features
       GL_CONTEXT_PROFILE_ES               allow only the ES feature
                                           subset of OpenGL

   :const:`GL_ACCELERATED_VISUAL`

     Set to 1 to require hardware acceleration, or 0 to force software render.
     By default, both are allowed.

   .. ## pygame.display.gl_set_attribute ##

.. function:: get_active

   | :sl:`Returns True when the display is active on the screen`
   | :sg:`get_active() -> bool`

   Returns True when the display Surface is considered actively
   renderable on the screen and may be visible to the user.  This is
   the default state immediately after ``pygame.display.set_mode()``.
   This method may return True even if the application is fully hidden
   behind another application window.

   This will return False if the display Surface has been iconified or
   minimized (either via ``pygame.display.iconify()`` or via an OS
   specific method such as the minimize-icon available on most
   desktops).

   The method can also return False for other reasons without the
   application being explicitly iconified or minimized by the user.  A
   notable example being if the user has multiple virtual desktops and
   the display Surface is not on the active virtual desktop.

   .. note:: This function returning True is unrelated to whether the
       application has input focus.  Please see
       ``pygame.key.get_focused()`` and ``pygame.mouse.get_focused()``
       for APIs related to input focus.

   .. ## pygame.display.get_active ##

.. function:: iconify

   | :sl:`Iconify the display surface`
   | :sg:`iconify() -> bool`

   Request the window for the display surface be iconified or hidden. Not all
   systems and displays support an iconified display. The function will return
   True if successful.

   When the display is iconified ``pygame.display.get_active()`` will return
   ``False``. The event queue should receive an ``ACTIVEEVENT`` event when the
   window has been iconified. Additionally, the event queue also receives a
   ``WINDOWEVENT_MINIMIZED`` event when the window has been iconified on pygame 2.

   .. ## pygame.display.iconify ##

.. function:: toggle_fullscreen

   | :sl:`Switch between fullscreen and windowed displays`
   | :sg:`toggle_fullscreen() -> int`

   Switches the display window between windowed and fullscreen modes.
   Display driver support is not great when using pygame 1, but with
   pygame 2 it is the most reliable method to switch to and from fullscreen.

   Supported display drivers in pygame 1:

    * x11 (Linux/Unix)
    * wayland (Linux/Unix)

   Supported display drivers in pygame 2:

    * windows (Windows)
    * x11 (Linux/Unix)
    * wayland (Linux/Unix)
    * cocoa (OSX/Mac)

   .. Note:: :func:`toggle_fullscreen` doesn't work on Windows
             unless the window size is in :func:`pygame.display.list_modes()` or
             the window is created with the flag ``pygame.SCALED``.
             See `issue #2380 <https://github.com/pygame/pygame/issues/2380>`_.

   .. ## pygame.display.toggle_fullscreen ##

.. function:: set_gamma

   | :sl:`Change the hardware gamma ramps`
   | :sg:`set_gamma(red, green=None, blue=None) -> bool`

   DEPRECATED: This functionality will go away in SDL3.

   Set the red, green, and blue gamma values on the display hardware. If the
   green and blue arguments are not passed, they will both be the same as red.
   Not all systems and hardware support gamma ramps, if the function succeeds
   it will return ``True``.

   A gamma value of ``1.0`` creates a linear color table. Lower values will
   darken the display and higher values will brighten.

   .. deprecated:: 2.2.0

   .. ## pygame.display.set_gamma ##

.. function:: set_gamma_ramp

   | :sl:`Change the hardware gamma ramps with a custom lookup`
   | :sg:`set_gamma_ramp(red, green, blue) -> bool`

   DEPRECATED: This functionality will go away in SDL3.

   Set the red, green, and blue gamma ramps with an explicit lookup table. Each
   argument should be sequence of 256 integers. The integers should range
   between ``0`` and ``0xffff``. Not all systems and hardware support gamma
   ramps, if the function succeeds it will return ``True``.

   .. deprecated:: 2.2.0

   .. ## pygame.display.set_gamma_ramp ##

.. function:: set_icon

   | :sl:`Change the system image for the display window`
   | :sg:`set_icon(Surface) -> None`

   Sets the runtime icon the system will use to represent the display window.
   All windows default to a simple pygame logo for the window icon.

   Note that calling this function implicitly initializes ``pygame.display``, if
   it was not initialized before.

   You can pass any surface, but most systems want a smaller image around
   32x32. The image can have colorkey transparency which will be passed to the
   system.

   Some systems do not allow the window icon to change after it has been shown.
   This function can be called before ``pygame.display.set_mode()`` to create
   the icon before the display mode is set.

   .. ## pygame.display.set_icon ##

.. function:: set_caption

   | :sl:`Set the current window caption`
   | :sg:`set_caption(title, icontitle=None) -> None`

   If the display has a window title, this function will change the name on the
   window. In pygame 1.x, some systems supported an alternate shorter title to
   be used for minimized displays, but in pygame 2 ``icontitle`` does nothing.

   .. ## pygame.display.set_caption ##

.. function:: get_caption

   | :sl:`Get the current window caption`
   | :sg:`get_caption() -> (title, icontitle)`

   Returns the title and icontitle for the display window. In pygame 2.x
   these will always be the same value.

   .. ## pygame.display.get_caption ##

.. function:: set_palette

   | :sl:`Set the display color palette for indexed displays`
   | :sg:`set_palette(palette=None) -> None`

   This will change the video display color palette for 8-bit displays. This
   does not change the palette for the actual display Surface, only the palette
   that is used to display the Surface. If no palette argument is passed, the
   system default palette will be restored. The palette is a sequence of
   ``RGB`` triplets.

   .. ## pygame.display.set_palette ##

.. function:: get_num_displays

   | :sl:`Return the number of displays`
   | :sg:`get_num_displays() -> int`

   Returns the number of available displays. This is always 1 if
   :func:`pygame.get_sdl_version()` returns a major version number below 2.

   .. versionadded:: 1.9.5

   .. ## pygame.display.get_num_displays ##

.. function:: get_window_size

   | :sl:`Return the size of the window or screen`
   | :sg:`get_window_size() -> tuple`

   Returns the size of the window initialized with :func:`pygame.display.set_mode()`.
   This may differ from the size of the display surface if ``SCALED`` is used.

   .. versionadded:: 2.0.0

   .. ## pygame.display.get_window_size ##

.. function:: get_allow_screensaver

   | :sl:`Return whether the screensaver is allowed to run.`
   | :sg:`get_allow_screensaver() -> bool`

   Return whether screensaver is allowed to run whilst the app is running.
   Default is ``False``.
   By default pygame does not allow the screensaver during game play.

   .. note:: Some platforms do not have a screensaver or support
             disabling the screensaver.  Please see
             :func:`pygame.display.set_allow_screensaver()` for
             caveats with screensaver support.

   .. versionadded:: 2.0.0

   .. ## pygame.display.get_allow_screensaver ##

.. function:: set_allow_screensaver

   | :sl:`Set whether the screensaver may run`
   | :sg:`set_allow_screensaver(bool) -> None`

   Change whether screensavers should be allowed whilst the app is running.
   The default value of the argument to the function is True.
   By default pygame does not allow the screensaver during game play.

   If the screensaver has been disallowed due to this function, it will automatically
   be allowed to run when :func:`pygame.quit()` is called.

   It is possible to influence the default value via the environment variable
   ``SDL_HINT_VIDEO_ALLOW_SCREENSAVER``, which can be set to either ``0`` (disable)
   or ``1`` (enable).

   .. note:: Disabling screensaver is subject to platform support.
             When platform support is absent, this function will
             silently appear to work even though the screensaver state
             is unchanged.  The lack of feedback is due to SDL not
             providing any supported method for determining whether
             it supports changing the screensaver state.
             ``SDL_HINT_VIDEO_ALLOW_SCREENSAVER`` is available in SDL 2.0.2 or later.
             SDL1.2 does not implement this.

   .. versionadded:: 2.0.0


   .. ## pygame.display.set_allow_screensaver ##

.. ## pygame.display ##
