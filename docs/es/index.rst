Página Principal de Pygame
==========================

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

:doc:`Introducción a Pygame <tut/es/IntroduccionAPygame>`
  Una introducción a los conceptos básicos de Pygame.
  Esto está escrito por usuarios de Python y aparece en el volúmen dos de la revista Py.

:doc:`Importación e Inicialización <tut/es/ImportInit>`
  Los pasos principales para importar e inicializar pygame.
  El paquete pygame está compuesto por varios módulos.
  Algunos de los módulos no están incluidos en todas las plataformas.

:doc:`¿Cómo muevo una imagen? <tut/es/MoverImagen>`
  Un tutorial básico que cubre los conceptos detrás de la animación 2D en computadoras.
  Información acerca de dibujar y borrar objetos para que parezcan animados.

:doc:`Tutorial del Chimpancé, Linea por Linea <tut/es/ChimpanceLineaporLinea>`
  Los ejemplos de pygame inlcuyen un simple programa con un puño interactivo y un chimpancé.
  Esto fue inspirado por un molesto banner flash de principios de los años 2000.
  Este tutorial examina cada línea del código usada en el ejemplo.

:doc:`Introducción al Módulo de Sprites <tut/es/SpriteIntro>`
  Pygame incluye un módulo de spirtes de nivel superior para ayudar a organizar juegos.
  El módulo de sprites incluye varias clases que ayudan a administrar detalles encontrados en casi 
  todos los tipos de juegos.
  Las clases de Sprites son un poco más avanzadas que los módulos regulares de pygame, y necesitan 
  de mayor comprensión para ser usados correctamente.

:doc:`Introducción a Surfarray <tut/es/SurfarrayIntro>`
  Pygame utiliza el módulo NumPy de Python para permitir efectos eficientes por píxel en imágenes.
  El uso de arrays de superficie (surface) es una función avanzada que permite efectos y filtros 
  personalizados. 
  Esto también examina algunos de los efectos simples del ejemplo de pygame, arraydemo.py.
  
:doc:`Introducción al Módulo de Cámara <tut/es/CamaraIntro>`
  Pygame, desde la versión 1.9, tiene un módulo de camara que te permite capturar imágenes,
  mirar transmiciones en vivo y hacer algo básico de visión de computadora.
  Este tutorial cubre esos usos.

:doc:`Guía Newbie <tut/es/GuiaNewbie>`
  Una lista de trece útiles tips para que las personas se sientas cómodas usando pygame.

:doc:`Tutorial para Crear Juegos <tut/es/CrearJuegos>`
  Un largo tutorial que cubre los grandes temas necesarios para crear un juego completo.

:doc:`Modos de Visualización <tut/ModosVisualización>`
  Obteniendo una superficie de visualización para la pantalla.

:doc:`한국어 튜토리얼 (Korean Tutorial) <tut/ko/빨간블록 검은블록/overview>`
  빨간블록 검은블록

Referencias
-----------

:ref:`genindex`
  Una lista de todas las funciones, clases, y métodos en el paquete de pygame.

:doc:`ref/bufferproxy`
  Una vista del protocolo de arrays de píxeles de superficie.

:doc:`ref/color`
  Representación de color

:doc:`ref/cursors`
  Carga y compilación de imágenes de cursores.

:doc:`ref/display`
  Configuración de la visualización de surface (superficie).

:doc:`ref/draw`
  Dibujo de formas simples como líneas y elipses en la surface (superficie).

:doc:`ref/event`
  Administración de eventos entrantes de varios dispositivos de entrada y de la plataforma de ventanas.

:doc:`ref/examples`
  Varios programas demostrando la utilización de módulos individuales de pygame.

:doc:`ref/font`
  Carga y representación de fuentes (letras) TrueType.

:doc:`ref/freetype`
  Módulo de Pygame mejorado para cargar y representar tipos de letras.

:doc:`ref/gfxdraw`
  Funciones de dibujo con suavizado de bordes (anti-aliasing).

:doc:`ref/image`
  Carga, guardado y transferencia de superficies (surfaces).

:doc:`ref/joystick`
  Administración de dispositivos joystick.

:doc:`ref/key`
  Administración de dispositivos de teclado.

:doc:`ref/locals`
  Constantes de Pygame.

:doc:`ref/mixer`
  Carga y reproducción de sonidos.

:doc:`ref/mouse`
  Administración del dispositivo de mouse y visualización.

:doc:`ref/music`
  Reproducción de pistas de sonido.

:doc:`ref/pygame`
  Funciones de nivel superior para manejar pygame.
  
:doc:`ref/pixelarray`
  Manipulación de datos de píxeles de imagen.

:doc:`ref/rect`
  Contenedor flexible para un rectángulo.

:doc:`ref/scrap`
  Acceso nativo al portapapeles.

:doc:`ref/sndarray`
  Manipulación de datos de muestra de sonidos.

:doc:`ref/sprite`
  Objetos de nivel superior para representar imágenes de juegos.

:doc:`ref/surface`
  Objetos para imagenes y la pantalla.

:doc:`ref/surfarray`
  Manipulación de datos de píxeles de imágenes.

:doc:`ref/tests`
  Testeo de pygame.

:doc:`ref/time`
  Administración del tiempo y la frecuencia de cuadros (framerate).

:doc:`ref/transform`
  Redminesionar y mover imágenes.

:doc:`pygame C API <c_api>`
  La API de C compartida entre los módulos de extensión de Pygame.

:ref:`search`
  Búsqueda de documentos de Pygame por palabra clave.

.. _Readme: ../wiki/about

.. _Install: ../wiki/GettingStarted#Pygame%20Installation

.. _File Path Function Arguments: filepaths.html

.. _LGPL License: LGPL.txt

.. _Pygame Logos: logos.html