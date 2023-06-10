.. TUTORIAL:Import and Initialize

.. include:: ../../reST/common.txt

***********************************************
  Tutoriales de Pygame - Importar e Inicializar
***********************************************
 
Importar e Inicializar
======================

.. rst-class:: docinfo

:Autor: Pete Shinners
:Contacto: pete@shinners.org
:Traducción al español: Estefanía Pivaral Serrano

Importar e inicializar pygame es un proceso muy simple. También es lo
suficientemente flexible para que el usuario tenga el control sobre lo que 
está sucediendo. Pygame es una colección de diferentes módulos en un mismo 
paquete de python. Algunos de los módulos están escritos en C, y algunos otros 
están escritos en python. Algunos módulos también son opcionales y es posible 
que no estén presentes.

Esto es solo una breve introducción sobre lo que sucede cuando se importa pygame.
Para una explicación más clara, definitivamente recomiendo que vean los ejemplos 
de pygame.


Importar
--------

Primero debemos importar el paquete de pygame. Desde la versión 1.4 de pygame
este ha sido actualizado para ser mucho más fácil. La mayoría de los juegos 
importarán todo pygame de esta manera.::

  import pygame
  from pygame.locals import *

La primera línea aquí es la única necesaria. Esta línea importa todos los módulos de 
pygame disponibles en el paquete de pygame. La segunda línea es opcional y plantea un 
conjunto de funciones limitadas en el 'espacio global de nombres' (global namespace) de 
la secuencia de comandos. 

Una cosa importante a tener en cuenta es que muchos de los módulos de pygame son 
opcionales. Por ejemplo, uno de estos es el módulo de fuentes. Cuando se importa pygame 
(import pygame), pygame comprobará si el módulo de fuentes está disponible.

Si el módulo de fuentes está disponible se importará como "pygame.font". Si el módulo
no está disponible, "pygame.font" se establecera como 'None' (ninguno). Esto hace que
sea bastante fácil probar más adelante si el módulo de fuentes está disponible.


Inicializar
-----------

Antes de que pueda hacerse mucho con pygame, será necesario inicializarlo.
La manera más común es hacerlo mediante una 'llamada' (call).::

  pygame.init()

Esto intentará inicializar todos los módulos de pygame automáticamente. No todos los 
módulos necesitan ser inicializados, pero esto inicializará automaticamente los que sí son 
necesarios. Se puede también inicializar fácilmente cada módulo de pygame de forma manual. 
Por ejemplo para inicializar únicamente el módulo de fuentes simplemente habría que hacer
el siguiente 'llamado'. ::

  pygame.font.init()

Tengan en cuenta que si hay un error cuando se inicialzia con "pygame.init()", fallará
silenciosamente. Al inicializar manualmente módulos como éste, cualquier error
generará una excepción. Cualquier módulo que deba ser inicializado también tiene 
una función "get_init()", que devolverá Verdadero (true) si el módulo ha sido inicializado.

Es seguro llamar a la función init() para cualquier módulo más de una vez.


Cerrar (Quit)
-------------

Los módulos que son inicializados por lo general tienen una función quit() (abandonar)
que dejará la configuración de los recursos como se encontraba antes. Las variables utilizadas
son destruidas. No hay necesidad de hacer un llamado explicitamente, ya que *pygame* cerrará 
limpiamente todos los módulos inicializados, una vez que python finaliza.
