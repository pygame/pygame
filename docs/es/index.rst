Página Principal de Pygame
==========================

.. toctree::
   :maxdepth: 2
   :glob:
   :hidden:

   referencias/*
   tutorials/*
   logos

Documentos
----------

`Readme`_
  Información básica acerca de pygame: qué es, quién está involucrado, y dónde encontrarlo.

`Install`_
  Pasos necesarios para compilar pygame en varias plataformas.
  También ayuda a encontrar e instalar binarios preconstruidos para tu sistema.
  
`File Path Function Arguments`_
  Cómo maneja Pygame las rutas del sistema de archivos.

`Pygame Logos`_
   Los logotipos de Pygame en diferentes resoluciones.

`LGPL License`_
  Esta es la licencia bajo la cual se distribuye pygame.
  Permite que pygame se distribuya como software de código abierto y comercial.
  En general, si pygame no se cambia, se puede utilizar con cualquier programa.

Tutoriales
----------

.. :doc:`Introducción a Pygame <tutorials/IntroduccionAPygame>`
..   Una introducción a los conceptos básicos de Pygame.
..   Esto está escrito por usuarios de Python y aparece en el volúmen dos de la revista Py.

:doc:`Importación e Inicialización <tutorials/IniciarImportar>`
  Los pasos principales para importar e inicializar pygame.
  El paquete pygame está compuesto por varios módulos.
  Algunos de los módulos no están incluidos en todas las plataformas.

:doc:`¿Cómo muevo una imagen? <tutorials/MoverImagen>`
  Un tutorial básico que cubre los conceptos detrás de la animación 2D en computadoras.
  Información acerca de dibujar y borrar objetos para que parezcan animados.

:doc:`Tutorial del Chimpancé, Linea por Linea <tutorials/ChimpanceLineaporLinea>`
  Los ejemplos de pygame inlcuyen un simple programa con un puño interactivo y un chimpancé.
  Esto fue inspirado por un molesto banner flash de principios de los años 2000.
  Este tutorial examina cada línea del código usada en el ejemplo.

:doc:`Introducción al Módulo de Sprites <tutorials/SpriteIntro>`
  Pygame incluye un módulo de spirtes de nivel superior para ayudar a organizar juegos.
  El módulo de sprites incluye varias clases que ayudan a administrar detalles encontrados en casi 
  todos los tipos de juegos.
  Las clases de Sprites son un poco más avanzadas que los módulos regulares de pygame, y necesitan 
  de mayor comprensión para ser usados correctamente.

:doc:`Introducción a Surfarray <tutorials/SurfarrayIntro>`
  Pygame utiliza el módulo NumPy de Python para permitir efectos eficientes por píxel en imágenes.
  El uso de arrays de superficie (surface) es una función avanzada que permite efectos y filtros 
  personalizados. 
  Esto también examina algunos de los efectos simples del ejemplo de pygame, arraydemo.py.
  
:doc:`Introducción al Módulo de Cámara <tutorials/CamaraIntro>`
  Pygame, desde la versión 1.9, tiene un módulo de camara que te permite capturar imágenes,
  mirar transmiciones en vivo y hacer algo básico de visión de computadora.
  Este tutorial cubre esos usos.

:doc:`Guía Newbie <tutorials/GuiaNewbie>`
  Una lista de trece útiles tips para que las personas se sientas cómodas usando pygame.

:doc:`Tutorial para Crear Juegos <tutorials/CrearJuegos>`
  Un largo tutorial que cubre los grandes temas necesarios para crear un juego completo.

:doc:`Modos de Visualización <tutorials/ModosVisualizacion>`
  Obteniendo una superficie de visualización para la pantalla.


Referencias
-----------

:ref:`genindex`
  Una lista de todas las funciones, clases, y métodos en el paquete de pygame.

:doc:`referencias/bufferproxy`
  Una vista del protocolo de arrays de píxeles de superficie.

:doc:`referencias/color`
  Representación de color

:doc:`referencias/cursors`
  Carga y compilación de imágenes de cursores.

.. :doc:`referencias/display`
..   Configuración de la visualización de surface (superficie).

.. :doc:`referencias/draw`
..   Dibujo de formas simples como líneas y elipses en la surface (superficie).

.. :doc:`referencias/event`
..   Administración de eventos entrantes de varios dispositivos de entrada y de la plataforma de ventanas.

.. :doc:`referencias/examples`
..   Varios programas demostrando la utilización de módulos individuales de pygame.

.. :doc:`referencias/font`
..   Carga y representación de fuentes (letras) TrueType.

.. :doc:`referencias/freetype`
..   Módulo de Pygame mejorado para cargar y representar tipos de letras.

.. :doc:`referencias/gfxdraw`
..   Funciones de dibujo con suavizado de bordes (anti-aliasing).

.. :doc:`referencias/image`
..   Carga, guardado y transferencia de superficies (surfaces).

.. :doc:`referencias/joystick`
..   Administración de dispositivos joystick.

.. :doc:`referencias/key`
..   Administración de dispositivos de teclado.

.. :doc:`referencias/locals`
..   Constantes de Pygame.

.. :doc:`referencias/mixer`
..   Carga y reproducción de sonidos.

.. :doc:`referencias/mouse`
..   Administración del dispositivo de mouse y visualización.

.. :doc:`referencias/music`
..   Reproducción de pistas de sonido.

.. :doc:`referencias/pygame`
..   Funciones de nivel superior para manejar pygame.
  
.. :doc:`referencias/pixelarray`
..   Manipulación de datos de píxeles de imagen.

.. :doc:`referencias/rect`
..   Contenedor flexible para un rectángulo.

.. :doc:`referencias/scrap`
..   Acceso nativo al portapapeles.

.. :doc:`referencias/sndarray`
..   Manipulación de datos de muestra de sonidos.

.. :doc:`referencias/sprite`
..   Objetos de nivel superior para representar imágenes de juegos.

.. :doc:`referencias/surface`
..   Objetos para imagenes y la pantalla.

.. :doc:`referencias/surfarray`
..   Manipulación de datos de píxeles de imágenes.

.. :doc:`referencias/tests`
..   Testeo de pygame.

.. :doc:`referencias/time`
..   Administración del tiempo y la frecuencia de cuadros (framerate).

.. :doc:`referencias/transform`
..   Redminesionar y mover imágenes.

.. :doc:`pygame C API <c_api>`
..   La API de C compartida entre los módulos de extensión de Pygame.

:ref:`search`
  Búsqueda de documentos de Pygame por palabra clave.

.. _Readme: ../../wiki/about

.. _Install: ../../wiki/GettingStarted#Pygame%20Installation

.. _File Path Function Arguments: ../filepaths.html

.. _LGPL License: ../LGPL.txt

.. _Pygame Logos: logos.html