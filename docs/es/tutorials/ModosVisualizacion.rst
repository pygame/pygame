.. TUTORIAL: Choosing and Configuring Display Modes

.. include:: ../../reST/common.txt

********************************************************************
  Tutoriales de Pygame - Configuración de los Modos de Visualización
********************************************************************


Configuración de los Modos de Visualización
===========================================

.. rst-class:: docinfo

:Author: Pete Shinners
:Contact: pete@shinners.org
:Traducción al español: Estefanía Pivaral Serrano

Introducción
------------

Configurar el modo de visualización en *pygame* crea una imagen de *Surface*
visible en el monitor.
Esta *Surface* puede o cubrir la pantalla completa, o si se está usando una 
plataforma que soporta la gestión de ventanas, la imagen puede usarse en ventana.
La *Surface* de visualización no es más que un objeto de *Surface* estándar de *pygame*.
Hay funciones especiales necesarias en el módulo :mod:`pygame.display` para mantener
los contenidos de la imagen de *Surface* actualizada en el monitor.

Configurar el modo de visualización en *pygame* es una tarea más fácil que con 
la mayoría de las bibliotecas gráficas.
La ventaja es que si el modo de visualización no está disponible, *pygame*
va a emular el modo de visualización que fue pedido.
*Pygame* seleccionará la resolución de la visualización y la profundidad 
del color de la visualización que mejor coincida con la configuración solicitada,
luego permitirá tener acceso al formato de visualización requerido.
En realidad, ya que el módulo :mod:`pygame.display` está
enlazado con la librería SDL, es SDL quién realmente hace todo este trabajo.

Esta forma de configurar el modo de visualización presenta ventajas y desventajas.
La ventaja es que si tu juego requiere un modo de visualización específico, 
el juego va a poder ejecutarse aún en plataformas que no soporten los requerimientos.
Esto también va a simplificarles la vida cuando estén comenzando con algo,
ya que siempre es fácil volver luego y hacer la selección de modo un poco más
específicos.
La desventaja es que lo que soliciten no es siempre lo que van a obtener.
Hay un castigo o multa en el rendimineto cuando el modo de visualización 
debe ser emulado. 
Este tutorial les ayudará a entender los métodos diferentes para consultar (querying)
las capacidades de visualización de las plataformas, y configurar el modo de 
visualización para tu juego.


Configuración básica
--------------------

Lo primero a aprender es cómo configurar realmente el modo de visualización actual.
El modo de visualización se puede establecer en cualquier momento luego de 
haber inicializado el módulo :mod:`pygame.display`
Si ya estableciste previamente el modo de visualización, configurarlo nuevamente va 
a cambiar el actual modo. La configuración del modo de visualización se maneja con la 
función :func: `pygame.display.set_mode((width, height), flags, depth)
<pygame.display.set_mode>`.
El único argumento requerido en esta función es la secuencia que contiene
el ancho (width) y el alto (height) del nuevo modo de visualización.
La bandera de profundidad (depth flag) es los bits por píxel solicitados 
para la *Surface*. Si la profundidad dada es 8, *pygame* va a crear la asignación 
de colores de la *Surface*.
En el caso que se le otorgue una mayor profundida de bits, *pygame* usará 
el modo de color empaquetado.

Podrán encontrar mucha más información acerca de profundidades y modo de 
color en la documentación sobre los módulos de visualización y *Surface*.
El valor por default para la profundidad es 0.
Cuando a un argumento se le asigna 0, *pygame* va a seleccionar el mejor bit 
de profunidad para usar, generalmente es el mismo bit de profundidad que el 
sistema actual.
El argumento de banderas permite controlar características extras para el 
modo de visualización.
Nuevamente, en caso de querer más información acerca del tema, se puede encontrar
en los documentos de referencia de *pygame*.


Cómo decidir
------------

Entonces, ¿cómo seleccionar el modo de visualización que va a funcionar mejor
con los recursos gráficos y en la plataforma en la que está corriendo el juego?
Hay varios métodos diferentes para reunir la información sobre la visualización
del dispositivo.
Todos estos métodos deben ser 'llamados' (called) luego de que se haya inicializado 
el módulo de visualización, pero es probable que quieran llamarlos antes de 
configurar el modo de visualización.
Primero, :func:`pygame.display.Info() <pygame.display.Info>`
va a devolver un tipo de objeto VidInfo especial, que les dirá mucho acerca
de las capacidades del controlador gráfico.
La función :func:`pygame.display.list_modes(depth, flags) <pygame.display.list_modes>`
puede ser usada para encontrar los modos gráficos respaldados por el sistema.
:func: `pygame.display.mode_ok((width, height), flags, depth)
<pygame.display.mode_ok>` toma el mismo argumento que 
:func:`set_mode() <pygame.display.set_mode>`,
pero devuelve la coincidencia más próxima al bit de profundidad solicitado.
Por último, :func:`pygame.display.get_driver() <pygame.display.get_driver>`
devuelve el nombre del controlador gráfico seleccionado por *pygame*

Solo hay que recordar la regla de oro: 
*Pygame* va a trabajar con practicamente cualquier modo de visualización solicitado.
A algunos modos de visualización va a ser necesario emularlos, lo cual va lentificar el
juego, ya que *pygame* va a necesitar convertir cada actualziación que se haga, al 
modo de visualización "real". La mejor apuesta es siempre dejar que *pygame* elija
la mejor profundidad de bit, y que convierta todos los recursos gráficos a ese formato 
cuando se carguen.
Al 'llamar' (call) a la función :func:`set_mode() <pygame.display.set_mode>` sin ningún 
argumento o con profundidad 0 dejamos que *pygame* elija por sí mismo la profundidad de bit. 
O sino se puede llamar a :func:`mode_ok() <pygame.display.mode_ok>` para encontrar
la coincidencia más cercana a la profundidad de bit necesaria.

Cuando el modo de visualización es en una ventana, lo que generalmente se debe hacer 
es hacer coincidir el bit de profundidad con el del escritorio.
Cuando se está usando pantalla completa, algunas plataformas pueden cambiar a 
cualquier bit de profundidad que mejor se adecue a las necesidades del usuario.
Pueden encontrar la profundidad del escritorio actual si obtienen un *objeto 
VidInfo* antes de configurar el modo de visualización.

Luego de establecer el modo de visualización,
pueden descubrir información acerca de su configuración al obtener el objeto
VidInfo, o al llamar cualquiera de los métodos Surface.get* en la superficie 
de visualización.

Funciones 
---------

Estas son las rutinas que se pueden usar para determinar el modo de 
visualización más apropiado.
Pueden encontrar más información acerca de estas funciones en la
documentación del modo de visualización.

  :func:`pygame.display.mode_ok(size, flags, depth) <pygame.display.mode_ok>`

    Esta función toma exactamente el mismo argumento que pygame.display.set_mode().
    Y devuelve el mejor bit de profundidad disponible para el modo que hayan descripto.
    Si lo que devuelve es cero, entonces el modo de visualización deseado no está
    disponible sin emulación.

  :func:`pygame.display.list_modes(depth, flags) <pygame.display.list_modes>`

    Deveuelve una lista de modos de visualización respaldados con la profundidad y 
    banderas solicitadas.
    Cuando no hay modos van a obtener como devolución una lista vacía.
    El argumento de las banderas por defecto es 
    :any:`FULLSCREEN <pygame.display.set_mode>`\ .
    Si especifican sus propias banderas sin :any:`FULLSCREEN <pygame.display.set_mode>`\ ,
    probablemente obtengan una devolución con valor -1.
    Esto significa que cualquier tamaño de visualización está bien, ya que la 
    visualización va a ser en ventana.
    Tengan en cuenta que los modos listados están ordenados de mayor a menor.

  :func:`pygame.display.Info() <pygame.display.Info>`

    Esta función devuelve un objeto con muchos miembros que describen
    el dispositivo de visualización.
    Mostrar (printing) el objeto VidInfo mostrará rápidamente todos los
    miembros y valores para ese objeto. ::

      >>> import pygame.display
      >>> pygame.display.init()
      >>> info = pygame.display.Info()
      >>> print(info)
      <VideoInfo(hw = 0, wm = 1,video_mem = 0
              blit_hw = 0, blit_hw_CC = 0, blit_hw_A = 0,
              blit_sw = 0, blit_sw_CC = 0, blit_sw_A = 0,
              bitsize  = 32, bytesize = 4,
              masks =  (16711680, 65280, 255, 0),
              shifts = (16, 8, 0, 0),
              losses =  (0, 0, 0, 8),
              current_w = 1920, current_h = 1080
      >

Pueden probar todas estas banderas (flags) simplemente como miembros del objeto VidInfo.


Ejemplos
--------

Acá hay algunos ejemplos de diferentes métodos para iniciar la 
visualización gráfica.
Estos deberían ayudar a dar una idea de cómo configurar su modo de visualizción ::

  >>> #dame la mejor profundidad con una visualización de ventana en 640 x 480
  >>> pygame.display.set_mode((640, 480))

  >>> #dame la mayor visualización disponible en 16-bit
  >>> modes = pygame.display.list_modes(16)
  >>> if not modes:
  ...     print('16-bit no está soportado')
  ... else:
  ...     print('Resolución encontrada:', modes[0])
  ...     pygame.display.set_mode(modes[0], FULLSCREEN, 16)

  >>> #es necesario una surface de 8-bit, nada más va a funcionar
  >>> if pygame.display.mode_ok((800, 600), 0, 8) != 8:
  ...     print('Solo puede funcionar con una visualización de 8-bit, lo lamento')
  ... else:
  ...     pygame.display.set_mode((800, 600), 0, 8)
