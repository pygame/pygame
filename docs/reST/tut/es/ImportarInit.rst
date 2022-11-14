.. TUTORIAL:Import and Initialize

.. incluir:: common.txt

********************************************
  Pygame Tutorials - Import and Initialize
********************************************
 
Import and Initialize
=====================

.. rst-class:: docinfo

:Autor: Pete Shinners
:Contacto: pete@shinners.org


Importar e inicializar pygame es un proceso simple. Tambien es suficientemente 
flexible para darte control sobre lo que está sucediendo. Pygame es una colección
de diferentes módulos en un solo paquete de python. Unos de los módulos están 
escritos en C y otros están escritos en python. Otros módulos también son 
opcionales y es posible que no siempre estén presentes.

Esto es solo una introducción corta sobre lo que sucede cuando importas pygame.
Para una explicación más clara definitivamente ve los ejemplos de pygame.



Importar
------

Primero debemos importar el paquete pygame. Desde la versión 1.4 de pygame, esto se 
ha actualizado para que sea mucho más fácil. La mayoría de los juegos importarán 
pygame de esta manera. ::

  import pygame
  from pygame.locals import *

La primera línea es la única que es necesaria. Esta línea importa todos los módulos de
pygame disponibles al el paquete pygame. La segunda línea es opcional y coloca un grupo
limitado de constantes y funciones en el espacio de nombres global de su secuencia de comandos.

Algo importante que tomar en cuenta es que varios módulos de pygame son opcionales.
Por ejemplo, uno de estos es el módulo de fuentes/fonts. Cuando "importas pygame", pygame 
verificará si el módulo de fuentes está disponible. Si el módulo está disponible, se importará
como "pygame.font". Si el módulo no está disponible, "pygame.font" se establecerá como None. 
Esto hace que sea fácil probar luego si el módulo de fuentes está disponible.


Init
----

Antes de que pueda hacer mucho con pygame, deberás inicializarlo. La forma más común de hacer
esto es simplemente hacer una llamada. ::

  pygame.init()

Esto intentará inicializar todos los módulos de pygame. No es necesario inicializar todos los 
módulos de pygame, pero esto automáticamente inicializará todos los que sean necesarios. 
También puede inicializar cada módulo de pygame a mano. Por ejemplo, para inicializar solamente 
el módulo de fuente necesitas hacer la siguiente llamada. ::

  pygame.font.init()

Teama en cuenta que si hay un error cuando inicializas "pygame.init()", fallará silenciosamente.
Al inicializar manualmente módulos como este, cualquier error generará una excepción. Cualquier 
módulo que deba inicializarse también tiene una función "get_init()", que devolverá True si el 
módulo se ha inicializado.

Es seguro llamar a la función init() para cualquier módulo más de una vez.


Quit
----

Los módulos que se inicializan también suelen tener una función quit() que lo limpiará.
No hay necesidad de llamarlos explícitamente, ya que pygame limpiamente cerrará todos los módulos 
inicializados cuando python finalice.
