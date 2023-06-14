.. TUTORIAL:Tom Chance's Making Games Tutorial

.. include:: ../../reST/common.txt

****************************
  Crear Juegos con Pygame
****************************


Crear Juegos con Pygame
========================

.. rst-class:: docinfo

:Traducción al español: Estefanía Pivaral Serrano

.. .. toctree::
..    :hidden:
..    :glob:

..    tom_games2
..    tom_games3
..    tom_games4
..    tom_games5
..    tom_games6

Tabla de Contenido
-------------------

\1. :ref:`Introducción <crearjuegos-1>`

  \1.1. :ref:`A note on coding styles <crearjuegos-1-1>`

.. \2. :ref:`Revisión: Fundamentos de Pygame <crearjuegos-2>`

..   \2.1. :ref:`El juego básico de pygame <crearjuegos-2-1>`

..   \2.2. :ref:`Objetos básicos de pygame <crearjuegos-2-2>`

..   \2.3. :ref:`Blitting <crearjuegos-2-3>`

..   \2.4. :ref:`El evento en loop (búcle de evento) <crearjuegos-2-4>`

..   \2.5. :ref:`Ta-ra! <crearjuegos-2-5>`

.. \3. :ref:`Dandole inicio <crearjuegos-3>`

..   \3.1. :ref:`Las primeras líneas y carga de módulos <crearjuegos-3-1>`

..   \3.2. :ref:`Funciones de manejo de recursos <crearjuegos-3-2>`

.. \4. :ref:`Clases de objeto de juego <crearjuegos-4>`

..   \4.1. :ref:`Una clase de pelota sencilla <crearjuegos-4-1>`

..     \4.1.1. :ref:`Desvío 1: Sprites <crearjuegos-4-1-1>`

..     \4.1.2. :ref:`Desvío 2: Física vectorial <crearjuegos-4-1-2>`

.. \5. :ref:`Objetos controlable por el usuario <crearjuegos-5>`

..   \5.1. :ref:`Una clase de bate sencillo <crearjuegos-5-1>`

..     \5.1.1. :ref:`Desvío 3: Eventos de Pygame <crearjuegos-5-1-1>`

.. \6. :ref:`Ensamblando los elementos  <crearjuegos-6>`

..   \6.1. :ref:`Deja que la pelota golpée los lados <crearjuegos-6-1>`

..   \6.2. :ref:`Deja que la pelota golpée el bate <crearjuegos-6-2>`

..   \6.3. :ref:`El producto final <crearjuegos-6-3>`


.. _crearjuegos-1:

1. Introducción
---------------

Antes que nada, asumo que han leído el tutorial :doc:`Chimpancé línea por línea <ChimpanceLineaporLinea>`,
que presenta lo básico de Python y pygame. Denle una leida antes de leer este tutorial, ya 
que no voy a repetir lo que ese tutorial dice (o al menos no en tanto detalle.) Este tutorial
apunta a aquellos que entienden cómo hacer un "juego" ridiculamente simple, y a quien le 
gustaría hacer un juego relativamente sencillo como Pong.
Les presenta algunos conceptos de diseño de juegos, algunas nociones matemáticas sencillas 
para trabajar con la física de la pelota, y algunas formas de mantener el juego fácil de 
mantener y expandir. 

Todo el código en este tutorial sirve para implementar `TomPong <http://www.tomchance.uklinux.net/projects/pong.shtml>`_,
un juego que yo he escrito. Hacia el fin del tutorial, no solo deberías tener una idea más firme 
de pygame, sino que también deberías poder entender como funciona TomPong y cómo hacer tu propia versión.

Ahora, un breve resumen de los conceptos básicos de pygame. Un método común para organizar el código 
de un juego es dividirlo en las siguientes seis secciones:

  - **Carga de módulos** que son requeridos por el juego. Cosas estándar, excepto que deberías recordar 
    importar nombres locales de pygame, así como el propio módulo de pygame.

  - **Funciones de manejo de recursos**; define algunas clases para el manejo de los recursos más básicos,
    que estará cargando imágenes y sonidos, como también conectandose y desconectandose de y hacia redes, 
    cargando partidas guardadas y cualquier otro recurso que puedas tener.

  - **Clases de objeto de juego**; define las clases de los objetos del juego. En el ejemplo de Pong, estos 
    serían uno para el bate del jugador (que podrás inicializar varias veces, uno para cada jugador en el juego)
    y otro para la pelota (que también podrá tener múltiples instancias). Si vas a tener un buen menú en el 
    juego, también es una buena idea hacer una clase del menú.
  
  - **Cualquier otra función del juego**; define otras funciones necesarias, como marcadores, manejo de menú, etc.
    Cualquier código que se podría poner en la lógica principal del juego, pero que dificultaría la comprensión de dicha
    lógica, deberá tener su propia función. Algo como trazar un marcador no es lógica del juego, entonces deberá moverse
    a una función.

  - **Inicializar el juego**,  incluyendo los propios objetos de pygame, el fondo, los objetos del juego (inicializando
    instancias de las clases) y cualquier otro pequeño fragmento de código que desee agregar.

  - **El loop (búcle) principal**, en el cual se puede poner cualquier manejo de entrada (es decir, pendiente de usuarios 
    presionando teclas/botones), el código para actualizar los objetos del juego y finalmente para actualizar la pantalla.

Cada juego que hagas tendrá alguna o todas estas secciones, posiblemente con más de las propias. Para los propósitos de este
tutorial, voy a escribir sobre como TomPong está planteado y las ideas sobre las que escribo pueden transferirse a casi 
cualquier tipo de juego que puedas crear. También voy a asumir que deseas mantener todo el código en un único archivo, pero 
si estás creando un juego razonablemente grande, suele ser una buena idea incluir ciertas secciones en los archivos de módulos.
Poner las clases de objeto en un archivo llamado ``objects.py``, por ejemplo, puede ayudarte a mantener la lógica del juego 
separada de los objetos del juego. Si tenés mucho código de manejo de recursos, también puede ser útil poner eso en ``resources.py``
Luego podés usar :code:`from objects,resources import *` para importar todas las clases y funciones.

.. _crearjuegos-1-1:

1.1. Una nota sobre estilos de codificación
-------------------------------------------

Lo primero a tener en cuenta cuando abordamos cualquier proyecto de programación es el decidir el estilo de codificación, y mantenerse
consistente. Python resuelve mucho de los problemas debido a su estricta interpretación de los espacios en blanco y la sangría, pero 
aún así se puede elegir el tamaño de sus sangrías, si coloca cada importación de módulo en una nueva línea, cómo comentas el código, 
etc. Verás cómo hago todas estas cosas en los ejemplos del código; no es necesario que se use mi estilo, pero cualquiera sea el estilo 
que adoptes, usalo en todo el programa. Tratá también de documentar todas tus clases y comentá en cada fragmento de código que parezca 
oscuro, sin caer en comentar lo obvio. He visto mucho gente que hace lo siguiente: :

  player1.score += scoreup        # Add scoreup to player1 score
                                    (Agrega scoreup al score de player 1)

El peor código está mal diseñado, con cambios en el estilo que aparentan ser aleatorios y documentación deficiente. El código deficiente
no solo es molesto para otras personas, pero también hace que sea difícil de mantener para uno mismo.
