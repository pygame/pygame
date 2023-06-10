.. include:: ../../reST/common.txt

:mod:`pygame.cursors`
=====================

.. module:: pygame.cursors
   :synopsis: módulo de pygame para recursos de cursor

| :sl:`pygame module for cursor resources`

Pygame ofrece control sobre el cursor del hardware del sistema.
Pygame admite cursores en blanco y negro (cursores de mapa de bits), así como
cursores variantes del sistema y cursores de color.
Podés controlar el cursor utilizando funciones dentro del módulo :mod:`pygame.mouse`.

Este módulo de cursores contiene funciones para cargar y decodificar 
varios formatos de cursores. Estas te permiten almacenar fácilmente tus 
cursores en archivos externos o directamente como cadenas de caracteres codificadas 
en Python. 

El módulo incluye varios cursores estándar. La función :func:`pygame.mouse.set_cursor()`
toma varios argumentos. Todos estos argumentos se han almacenado en una única 
tupla que puedes llamar de la siguiente manera: 

::

   >>> pygame.mouse.set_cursor(*pygame.cursors.arrow)
   
Las siguientes variables pueden ser pasadas a ``pygame.mouse.set_cursor`` function:

   * ``pygame.cursors.arrow``

   * ``pygame.cursors.diamond``

   * ``pygame.cursors.broken_x``

   * ``pygame.cursors.tri_left``

   * ``pygame.cursors.tri_right``

Este módulo también contiene algunos cursores como cadenas de caracters formateadas.
Será necesario pasarlos a la función ``pygame.cursors.compile()`` antes de 
poder utilizarlos. 
El ejemplo de llamada se vería así:

::

   >>> cursor = pygame.cursors.compile(pygame.cursors.textmarker_strings)
   >>> pygame.mouse.set_cursor((8, 16), (0, 0), *cursor)

Las siguientes cadenas de caracteres se pueden convertir en mapas de bits de cursor con
``pygame.cursors.compile()`` :

   * ``pygame.cursors.thickarrow_strings``

   * ``pygame.cursors.sizer_x_strings``

   * ``pygame.cursors.sizer_y_strings``

   * ``pygame.cursors.sizer_xy_strings``
   
   * ``pygame.cursor.textmarker_strings``

.. function:: compile

   | :sl:`create binary cursor data from simple strings`
   | :sg:`compile(strings, black='X', white='.', xor='o') -> data, mask`

   Se puede utilizar una secuencia de cadenas para crear datos binarios de 
   cursor para el cursor del sistema. Esto devuelve los datos binarios en 
   forma de dos tuplas. Estas se pueden pasar como tercer y cuarto argumento,
   respectivamente, de la función :func:`pygame.mouse.set_cursor()`.
   
   Si estás creando tus propias cadenas de caracteres, podés usar cualquier valor 
   para representar los píxeles blanco y negro. Algunos sistemas permiten 
   establecer un color especial de alternancia para el color del sistema, 
   también llamado color xor. Si el sistema no admite cursores xor, ese color 
   será simplemente negro.

   La altura debe ser divisible por 8. el ancho de las cadenas debe ser igual 
   y divisible por 8. Si estas dos condiciones no se cumplen, se generará un 
   ``ValueError``.
   Un ejemplo de conjunto de cadenas de caracteres de cursor se ve así:

   ::

       thickarrow_strings = (               #sized 24x24
         "XX                      ",
         "XXX                     ",
         "XXXX                    ",
         "XX.XX                   ",
         "XX..XX                  ",
         "XX...XX                 ",
         "XX....XX                ",
         "XX.....XX               ",
         "XX......XX              ",
         "XX.......XX             ",
         "XX........XX            ",
         "XX........XXX           ",
         "XX......XXXXX           ",
         "XX.XXX..XX              ",
         "XXXX XX..XX             ",
         "XX   XX..XX             ",
         "     XX..XX             ",
         "      XX..XX            ",
         "      XX..XX            ",
         "       XXXX             ",
         "       XX               ",
         "                        ",
         "                        ",
         "                        ")

   .. ## pygame.cursors.compile ##

.. function:: load_xbm

   | :sl:`load cursor data from an XBM file`
   | :sg:`load_xbm(cursorfile) -> cursor_args`
   | :sg:`load_xbm(cursorfile, maskfile) -> cursor_args`

   Esto carga cursores para un subconjunto simple de archivos ``XBM`` . Los 
   archivos ``XBM`` son tradicionalmente utilizados para almacenar cursores 
   en sistemas UNIX, son un formato ASCII utilizado para representar 
   imágenes simples.

   A veces, los valores de color blanco y negro se dividen en dos archivos 
   ``XBM``  separados. Podés pasar un segundo argumento de archivo de máscara 
   (maskfile) para cargar las dos imágenes en un solo cursor.
   
   Los argumentos 'cursorfile' y 'maskfile' pueden ser nombres de archivos 
   u objetos similares a archivos con el método 'readlines'

   El valor de retorno 'cursor_args' puede ser pasado directamente 
   a la función ``pygame.mouse.set_cursor()``.

   .. ## pygame.cursors.load_xbm ##



.. class:: Cursor

   | :sl:`pygame object representing a cursor`
   | :sg:`Cursor(size, hotspot, xormasks, andmasks) -> Cursor`
   | :sg:`Cursor(hotspot, surface) -> Cursor`
   | :sg:`Cursor(constant) -> Cursor`
   | :sg:`Cursor(Cursor) -> Cursor`
   | :sg:`Cursor() -> Cursor`

   En pygame 2, hay 3 tipos de cursores que podés crear para darle un poco 
   de brillo adicional a tu juego. Existen cursores de tipo **bitmap**, que 
   ya existían en Pygame 1.x, y se compilan a partir de una cadena de caracteres 
   o se cargan desde un archivo xbm. Luego, están los cursores de tipo **system**, 
   donde eliges un conjunto predefinido que transmitirá el mismo significado pero 
   se verá nativo en diferentes sistemas operativos. Por último puedes crear un 
   cursor de tipo **color**, que muestra una superficie de Pygame como el cursor.

   **Creando un cursor del sistema**

   Elegí una constante de esta lista, pasala a ``pygame.cursors.Cursor(constant)``, 
   ¡y listo! Tené en cuenta que no todos los sistemas admiten todos los cursores 
   del sistema y es posible que obtengas una sustitución en su lugar. Por ejemplo, 
   en MacOS, WAIT/WAITARROW debería mostrarse como una flecha y 
   SIZENWSE/SIZENESW/SIZEALL debería mostrarse como una mano cerrada. Y en Wayland, 
   cada cursor SIZE debería aparecer como una mano.
   debería mostrarse como una mano cerrada. 

   ::

      Pygame Cursor Constant           Description
      --------------------------------------------
      pygame.SYSTEM_CURSOR_ARROW       arrow (flecha)
      pygame.SYSTEM_CURSOR_IBEAM       i-beam (viga en i, o viga de doble t)
      pygame.SYSTEM_CURSOR_WAIT        wait (espera)
      pygame.SYSTEM_CURSOR_CROSSHAIR   crosshair (cruz de mira)
      pygame.SYSTEM_CURSOR_WAITARROW   small wait cursor (pequeño cursor de espera)
                                       (or wait if not available) (o si no está disponible, espera)
      pygame.SYSTEM_CURSOR_SIZENWSE    double arrow pointing (doble flecha apuntando al noroeste y sudeste)
                                       northwest and southeast
      pygame.SYSTEM_CURSOR_SIZENESW    double arrow pointing (doble flecha apuntando al noreste y sudoeste)
                                       northeast and southwest
      pygame.SYSTEM_CURSOR_SIZEWE      double arrow pointing (doble flecha apuntando al oeste y al este)
                                       west and east
      pygame.SYSTEM_CURSOR_SIZENS      double arrow pointing (doble flecha apuntando al norte y sur)
                                       north and south
      pygame.SYSTEM_CURSOR_SIZEALL     four pointed arrow pointing (flecha de cuatro puntas apuntando al norte, sur, este y oeste)
                                       north, south, east, and west
      pygame.SYSTEM_CURSOR_NO          slashed circle or crossbones (círculo tachado o calaveras cruzadas)
      pygame.SYSTEM_CURSOR_HAND        hand (mano)

   **Creando un cursor sin pasar argumentos**
   
   Además de las constantes del cursor disponibles y descritas anteriormente, 
   también podés llamar a ``pygame.cursors.Cursor()``, y tu cursor está listo 
   (hacer esto es lo mismo que llamar a ``pygame.cursors.Cursor(pygame.SYSTEM_CURSOR_ARROW)``)
   Haciendo una de estas llamadas lo que en realidad se crea es un cursor del sistema 
   utilizando la imagen nativa predeterminada.

   **Creando un curosr de color**

   Para crear un cursor de color, hay que crear un objeto ``Cursor`` a partir de un 
   ``hotspot`` y una ``surface``. Un ``hotspot`` es una coordenada (x,y) que determina 
   donde en el cursor está el punto exacto. La posición debe estar dentro de los límites 
   de la ``surface``.


   **Creando un cursor de mapa de bits (bitmap)**

   Cuando el cursor del mouse está visible, se mostrará como un mapa de bits 
   en blanco y negro utilizando los arrays de máscaras (bitmask) dadas. El 
   ``size`` (tamaño) es una secuencia que contiene el ancho y alto del cursor. 
   El ``hotspot``es una secuencia que contiene la posición del hotspot del 
   cursor.
   
   Un cursor tiene un ancho y alto, pero la posición del mouse está 
   representada mediante un conjunto de coordenadas de punto. Por lo tanto, 
   el valor pasado al ``hotspot`` del cursor ayuda a pygame a determinar 
   exactamente en qué punto se encuentra el cursor.
   
   ``xormasks``es una secuencia de bytes que contiene las máscaras de 
   datos del cursor. Por último, ``andmasks`` es una secuencia de bytes 
   que contiene los datos de máscara de bits del cursor. Para crear estas 
   variable podemos utilizar la función :func:`pygame.cursors.compile()`.

   Ancho y alto deben ser múltiplos de 8, y las arrays de máscara (mask arrays) 
   deben tener el tamaño correcto para el ancho y el alto dados. De lo contrario, 
   se generará una excepción.
   
   .. method:: copy

      | :sl:`copy the current cursor`
      | :sg:`copy() -> Cursor`
      
      Devuelve un nuevo objeto Cursor con los mismos datos y hotspots que el original.
   .. ## pygame.cursors.Cursor.copy ##
   

   .. attribute:: type
   
      | :sl:`Gets the cursor type`
      | :sg:`type -> string`

      El tipo será ``"system"``, ``"bitmap"``, o ``"color"``.

   .. ## pygame.cursors.Cursor.type ##

   .. attribute:: data

      | :sl:`Gets the cursor data`
      | :sg:`data -> tuple`

      Devuelve los datos que se utilizaron para crear este objeto de cursor, envuelto en una tupla.

   .. ## pygame.cursors.Cursor.data ##

   .. versionadded:: 2.0.1

   .. ## pygame.cursors.Cursor ##
   
.. ## pygame.cursors ##

Código de ejemplo para crear y establecer cursores. (Click en el mouse para cambiar el cursor)

.. literalinclude:: ../../reST/ref/code_examples/cursors_module_example.py
