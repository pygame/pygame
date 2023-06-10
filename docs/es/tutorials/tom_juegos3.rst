.. include:: ../../reST/common.txt

****************
  Dando Inicio
****************

.. role:: citetitle(emphasis)

.. _hacerjuegos-3:

1. Dando Inicio
================

Las primeras secciones de código son relativamente simples y, una vez escritas, pueden usualmente ser pueden reutilizar en todos los 
juegos que posteriormente hagas. Realizarán todas las tareas aburridas y genéricas como cargar módulos, cargar imágenes, abrir 
conexiones de red, reproducir música y así sucesivamente. También incluirán una gestión de errores simple pero efectiva, y cualquier 
personalización que desees proporcionar además de las funciones proporcionadas por los módulos como ``sys`` y ``pygame``.


.. _hacerjuegos-3-1:

3.1. Primeras líneas y carga de módulos
---------------------------------------

En primer lugar, necesitás iniciar tu juego y cargar tus módulos. Siempre es una buena idea establecer algunas cosas al principio del 
archivo fuente principal, como el nombre del archivo, qué contiene, bajo qué licencia se encuentra, y cualquier otra información útil 
que desees proporcionar a quienes que lo van a estar viendo. Luego puedes cargar módulos, con algunas verificaciones de errores para 
que Python no imprima una traza desagradable que los no programadores no entenderán. El código es bastante simple, por lo que no me 
molestaré en explicarlo.::

  #!/usr/bin/env python
  #
  # Tom's Pong
  # A simple pong game with realistic physics and AI
  # http://www.tomchance.uklinux.net/projects/pong.shtml
  #
  # Released under the GNU General Public License

  VERSION = "0.4"

  try:
      import sys
      import random
      import math
      import os
      import getopt
      import pygame
      from socket import *
      from pygame.locals import *
  except ImportError, err:
      print(f"couldn't load module. {err}")
      sys.exit(2)


.. _hacerjuegos-3-2:

3.2. Funciones de manejo de recursos
------------------------------------

En el ejemplo :doc:`Chimpancé, Línea Por Línea <ChimpanceLineaporLinea>`, el primer código que se escribió fue para cargar imagenes y sonidos. 
Como estos eran totalmente independiente de cualquier lógica de juego u objetos del juego, se los escribió como funciones separadas 
y se escribieron primero para que el código posterior pudiera hacer uso de ellas. Generalmente, coloco todo mi código de esta 
naturaleza primero, en sus propias funciones sin clase; estas serán, en términos generales, funciones de manejo de recursos. Por 
supuesto, también podés crear clases para estas funciones, para que puedas agruparlas y tal vez tener un objeto con el que puedas 
controlar todos los recursos. Como con cualquier buen entorno de programación, depende de vos desarrollar tu propia práctica y estilo 
óptimo.

Siempre es una buena idea escribir tus propias funciones de manejo de recursos, porque aunque Pygame tiene métodos para abrir 
imágenes y sonidos, y otros módulos tendrán sus métodos para abrir otros recursos, esos métodos pueden ocupar más de una línea, 
pueden requerir una modificación constante de tu parte y a menudo no proporcionan un manejo de errores satisfactorios. Escribir 
funciones de manejo de recursos te da código sofisticado y reutilizable, y te da mayor control sobre tus recursos. Tomá este 
ejemplo de una función de carga de imágenes::

  def load_png(name):
      """ Load image and return image object"""
      fullname = os.path.join("data", name)
      try:
          image = pygame.image.load(fullname)
          if image.get_alpha() is None:
              image = image.convert()
          else:
              image = image.convert_alpha()
      except FileNotFoundError:
          print(f"Cannot load image: {fullname}")
          raise SystemExit
      return image, image.get_rect()

Acá creamos una función de carga de imagen más sofisticada que la proporcionada por :func:`pygame.image.load`. Observen que la 
primera línea de la función es una cadena de documentación describiendo qué hace cada función, y qué objeto(s) devuelve. La 
función asume que todas tus imágenes están en el directorio llamado "data", por lo que toma el nombre del archivo y crea la ruta 
completa, por ejemplo ``data/ball.png``, usando el módulo :citetitle:`os` para asegurar la compatibilidad entre plataformas. 
Luego intenta cargar la imagen y convertir cualquier región alfa para que puedas lograr la transparencia, y devuelve un mensaje 
de error más legible si hay algún problema. Finalmente, devuelve el objeto de imagen y su clase :class:`rect <pygame.Rect>`.

Podés crear funciones similares para cargar cualquier otro recurso, como cargar sonidos. También podés crear clases de manejo de 
recursos para darte más flexibilidad con recursos más complejos. Por ejemplo, podrías crear una clase de música, con una función 
``__init__`` que carga la música (quizás tomando prestada de la función ``load_sound()``), una función para pausar la música y 
otra para reiniciarla. Otra clase útil de manejo de recursos es para conexiones de red. Funciones para abrir sockets, pasar datos 
con seguridad y verificación de errores adecuados, cerrar sockets, buscar direcciones, y otras tareas de red, pueden hacer que 
escribir un juego con capacidades de red sea relativamente indoloro.

Recordá que la tarea principal de estas funciones/clases es asegurarse de que para cuando llegues a escribir las clases de objetos 
de juego y el bucle principal, casi no haya nada más que hacer. La herencia de clases puede hacer que estas clases básicas sean 
especialmente útiles. Pero no te excedas, las funciones que solo serán utilizadas por una clase deben ser escritas como parte de 
esa clase, no como una función global.
