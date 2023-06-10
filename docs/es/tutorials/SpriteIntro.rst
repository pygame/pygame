.. TUTORIAL: Sprite Module Introduction

.. include:: ../../reST/common.txt

********************************************************
Tutoriales de Pygame - Introducción al Módulo de Sprites
********************************************************


Introducción al Módulo de Sprites
=================================

.. rst-class:: docinfo

:Author: Pete Shinners
:Contact: pete@shinners.org
:Traducción al español: Estefanía Pivaral Serrano

Comentario: una forma simple de entender los Sprites, es pensarlos como 
elementos visuales utilizados para representar objetos y personajes en 
juegos, y se pueden crear y manipular utilizando la biblioteca de Pygame. 
Si bien se podría traducir el término "sprite" por "imagen en movimiento" 
o "personaje animado", en el contexto de programación se ha adoptado 
ampliamente y es comúnmente utilizado en español, sin traducción.

La versión de pygame 1.3 viene con un nuevo módulo, ``pygame.sprite``. Este 
módulo está escrito en Python e incluye algunas clases de nivel superior para 
administrar los objetos del juego. Al usar este módulo en todo su potencial, 
se puede fácilmente administrar y dibujar los objetos del juego. Las clases de 
sprites están muy optimizadas, por lo que es probable que tu juego funcione más 
rápido con el módulo de sprites que sin él.

El módulo de sprites también pretende ser genérico, resulta que lo podés 
usar con casi cualquier tipo de juego. Toda esta flexibilidad viene con una 
pequeña penalización, es necesario entenderlo para usarlo correctamente. El 
:mod:`reference documentation <pygame.sprite>` para el módulo de sprites 
puede mantenerte andando, pero probablemente necesites un poco más de 
explicaicón sobre cómo usar ``pygame.sprite`` en tu propio juego.

Varios de los ejemplos de pygame (como "chimp" y "aliens") han sido actualizados 
para usar el módulo de sprites. Es posible que quieras verificarlos para ver de 
qué se trata este módulo de sprites. El módulo de chimp incluso tiene su propio 
tutorial línea por línea, que puede ayudar a comprender mejor la programación 
con python y pygame.

Tengan en cuenta que esta introducción asumirá que tienen un poco de experiencia 
programando con python y que están familiarizados con diferentes partes de la 
creación de un simple juego. En este tutorial la palabra "referencia" es usada 
ocasionalmente. Esta representa una variable de python. Las variables en python 
son referencias, por lo que pueden haber varias variables apuntando al mismo 
objeto. 

Lección de Historia
-------------------

El término "sprite" es un vestigio de las computadoras y máquinas de juego 
más antiguas. Estas cajas antiguas no eran capaces de dibujar y borrar 
gráficos normales lo suficientemente rápido como para que funcionara como 
juego. Estas máquinas tenían un hardware especial para manejar juegos como 
objetos que necesitaban animarse rápidamente. Estos objetos eran llamados 
"sprites" y tenían limitaciones especiales, pero podían dibujarse y 
actualizarse muy rápido. Por lo general, existían en buffers especiales 
superpuestos en el video. Hoy en día las computadores se han vuelto lo 
suficientemente rápidas para manejar objetos similares a sprites sin un 
hardware dedicado. El término sprite es todavía usado para representar 
casi cualquier cosa en un juego 2D animado.  

Las Clases
----------

El módulo de sprites viene con dos clases principales. La primera es 
:class:`Sprite <pygame.sprite.Sprite>`, que debe usarse como clse base para 
todos los objetos de tu juego. Esta clase realmente no hace nada por sí sola, 
sólo incluye varias funciones para ayudar a administrar el objeto del juego. 
El otro tipo de clase es :class:`Group <pygame.sprite.Group>`. La clase 
``Group`` es un contenedor para diferentes objetos ``Sprite``. De hecho, hay 
varios tipos diferentes de clases de Group. Algunos de los ``Groups`` pueden 
dibujar todos los elementos que contienen, por ejemplo.

Esto es todo lo que hay, realmente. Comenzaremos con una descriçión de lo que 
hace cada tipo de clase y luego discutiremos las formas adecuadas de usar las 
dos clases.

La Clase Sprite
---------------

Como se mencionó anteriormente, la clase Sprite está diseñada para ser una clase 
base para todos los objetos del juego. Realmente no podés usarla por sí sola, ya 
que sólo tiene varios métodos para ayudarlo a trabajar con diferentes clases 
``Grupo``. El sprite realiza un seguimiento de a qué grupo pertenece.
El constructor de clases (método ``__init__``) toma un argumento de un ``Grupo`` 
(o listas de ``Grupos``) al que debería pertencer la instancia ``Sprite``. 
También se puede cambiar la pertenencia del ``Sprite`` con los métodos 
:meth:`add() <pygame.sprite.Sprite.add>` y 
:meth:`remove() <pygame.sprite.Sprite.remove>`. 
Hay también un método :meth:`groups() <pygame.sprite.Sprite.groups>`, que devuelve 
una lista de los grupos actuales que contiene el sprite.

Cuando se usen las clases de Sprite, es mejor pensarlas como "válidas" o "vivas", 
cuando pertenecen a uno o más ``Grupos``. Cuando se eliminen las instancias de todos 
los grupos, pygame limpiará el objeto. (A menos que tengas tus propias referencias 
a la instancia en otro lugar.) El método :meth:`kill() <pygame.sprite.Sprite.kill>` 
elimina los sprites de todos los grupos a los que pertenece. Esto eliminará 
limpiamente el objeto sprite. Si ya has armado algún juego, sabés que a veces 
eliminar limpiamente un objeto del juego puede ser complicado. El sprite también 
viene con un método :meth:`alive() <pygame.sprite.Sprite.alive>` que devuelve "true"
(verdadero) si todavía es miembro de algún grupo.


La Clase Grupo
--------------

La clase ``Group`` es solo un simple contenedor. Similar a un sprite, tiene 
un método :meth:`add() <pygame.sprite.Group.add>` y otro método
:meth:`remove()<pygame.sprite.Group.remove>` que puede cambiar qué sprites 
pertenecen a el grupo. También podés pasar un sprite o una lista de sprites 
al constructor (``__init__()`` method) para crear una instancia ``Group`` 
que contiene algunos sprites iniciales.

El ``Group`` tiene algunos otros métodos como 
:meth:`empty()<pygame.sprite.Group.empty>` para eliminar todos los sprites 
de el grupo y :meth:`copy() <pygame.sprite.Group.copy>` que devolverá una 
copia del grupo con todos los mismos miembros. Además, el método 
:meth:`has() <pygame.sprite.Group.has>` verificará rápidamente si el 
``Group`` contiene un sprite o lista de sprites.

La otra función que usarás frecuentemente es el método 
:meth:`sprites()<pygame.sprite.Group.sprites>`. Esto devuelve un objeto 
que se puede enlazar para acceder a todos los sprites que contiene el grupo.
Actualmente, esta es solo una lista de sprites, pero en una versión posterior 
de python es probable que use iteradores para un mejor rendimiento. 

Como atajo, el ``Group`` también tiene un método 
:meth:`update()<pygame.sprite.Group.update>`, que llamará a un método 
``update()`` para cada sprite en el grupo, pasando los argumentos a cada uno. 
Generalmente, en un juego se necesita alguna función que actualice el estado de 
los objetos del juego. Es muy fácil llamar a tu propio método usando el método 
``Group.sprites()``, pero este es un atajo que se usa lo suficiente como para 
ser incluido. También, tengan en cuenta que la clase base ``Sprite`` tiene un 
método ficticio, tipo "dummy", ``update()`` que toma cualquier tipo de 
argumento y no hace nada.

Por último, el Group tiene un par de otros métodos que permiten usarlo como 
funición interna ``len()``, obteniendo el número de sprites que contiene, y 
el operador "truth" (verdad), que te permite hacer "if mygroup:" para verificar 
si el grupo tiene sprites.


Mezclándolos Juntos
-------------------

A esta altura, las dos clases parecen bastante básicas. No hacen mucho más de 
lo que podés hacer con una simple lista y tu propia clase de objetos de juego.
Pero hay algunas ventajas grandes al usar ``Sprite`` y ``Group`` juntos. Un 
sprite puede pertenecer a tantos grupos como quieras, recordá que tan pronto 
como pertenezca a ningún grupo, generalmente se borrará (a menos que tengas otra 
referencia "no-grupales" para ese objeto) 

Lo primero es una forma rápida y sencilla de categorizar sprites. Por ejemplo, 
digamos que tenemos un juego tipo Pacman. Podríamos hacer grupos separados por 
diferentes tipos de objetos en el juego. Fantasmas, Pac y Pellets (pastilla de 
poder). Cuando Pac come una pastilla de poder, podemos cambiar el estado de todos 
los objetos fantasma afectando a todo el grupo Fantasma. Esta manera es más rápida 
y sencilla que recorrer en loop la lista de todos los objetos del juego y comrpobar
cuáles son fantasmas.

Agregar y eliminar grupos y sprites entre sí es una operación muy rápida, más 
rápida que usar listas para almacenar todo. Por lo tanto, podés cambiar de manera 
muy eficiente la pertenencia de los grupos. Los grupos se pueden usar para funcionar 
como atributos simples para cada objeto del juego. En lugar de rastrear algún atributo 
como "close_to_player" para un montón de objetos enemigos, podrías agregarlos a un 
grupo separado. Luego, cuando necesites acceder a todos los enemigos que están cerca 
del jugador, ya tenés una lista de ellos, en vez de examinar una lista de todos los 
enemigos, buscando el indicador "close_to_player". Más adelante, tu juego podría 
agregar múltiples jugadores, y en lugar de agregar más atributos "close_to_player2",
"close_to_player3", podés fácilmente agregarlos a diferentes grupos o a cada jugador.

Otro beneficio importante de usar ``Sprites`` y ``Groups`` es que los grupos 
manejan limpiamente el borrado (o eliminación) de los objetos del juego. En un juego 
en el que muchos objetos hacen referencia a otros objetos, a veces eliminar un objeto 
puede ser la parte más difícil, ya que no puede desaparecer hasta que nadie haga 
referencia a él. Digamos que tenemos un objeto que está "persiguiendo" a otro objeto.
El perseguidor puede mantener un Group simple que hace referencia al objeto (u 
objetos) que está persiguiendo. Si el objeto perseguido es destruido, no necesitamos 
preocuparnos por notificar al perseguidor que deje de perseguir. El perseguidor puede 
verlo por sí mismo que su grupo está ahora vacío y quizás encuentre un nuevo objetivo.

Una vez más, lo que hay que recordar es que agregar y eliminar sprites de grupos es 
una operación muy barata/rápida. Puede que te vaya mejor agregando muchos grupos 
para contener y organizar los objetos de tu juego. Algunos podrían incluso estar 
vacíos durante gran parte del juego, no hay penalizaciones por administrar tu juego 
de esta manera.


Los Muchos Tipos de Grupos
--------------------------

Los ejemplos anteriores y las razones para usar ``Sprites`` y ``Groups`` son solo 
la punta del iceberg. Otra ventaja es que el módulo viene con varios tipos 
diferentes de ``Groups``. Todos estos grupos funcionan como un ``Group`` normal 
y corrientes, pero también tienen funcionalidades añadidas (o ligeramente 
diferentes). Acá hay una lista de las clases ``Group`` incluidas con el módulo 
de sprites.

  :class:`Group <pygame.sprite.Group>`

    Este es el grupo estándar, "sin lujos", explicado principalmente 
    anteriormente. La mayoría de los otros ``Groups`` se derivan de este,
    pero no todos.

  :class:`GroupSingle <pygame.sprite.GroupSingle>`

    Esto funciona exactamente como la clase regular ``Group``, pero solo contiene 
    el sprite agregado más recientemente. Por lo tanto, cuando agregues un sprite 
    a este grupo, se "olvida" de los sprites que tenía anteriormente. Por lo tanto, 
    siempre contiene solo uno o cero sprites.

  :class:`RenderPlain <pygame.sprite.RenderPlain>`

    Este es un grupo estándar derivado de ``Group``. Tiene un método draw() 
    que dibuja en la pantalla (o en cualquier ``Surface``) todos los sprites 
    que contiene. Para que esto funcione, requiere que todos los sprites 
    contenidos tengan los atributos "imagen" y "rect". Estos son utilizados 
    para saber qué blittear y donde blittear.

  :class:`RenderClear <pygame.sprite.RenderClear>`

    Esto se deriva del grupo ``RenderPlain`` y agrega además un método 
    llamado ``clear()``. Esto borrará las posiciónes previas de todos los 
    sprites dibujados. Utiliza la imagen de fondo para rellenar las áreas 
    donde estaban los sprites. Es lo suficientemente inteligente como para 
    manejar los sprites eliminados y borrarlos adecuadamente de la pantalla 
    cuando se llama al método ``clear()``.

  :class:`RenderUpdates <pygame.sprite.RenderUpdates>`

    Este es el Cádilac de renderizado de ``Groups``. Es heredado de 
    ``RenderClear``, pero cambia el método ``draw()`` para también 
    devolver una lista de ``Rects`` de pygame, que representan todas las 
    áreas de la pantalla que han sido modificadas.

Esa es la lista de los diferentes grupos disponibles. Hablaremos más acerca 
de estos grupos de rendering en la próxima sección. No hay nada que te impida 
crear tus propias clases de grupos tampoco. Son solo código de python, asi que 
podés heredar de uno de estos y agregar/cambiar lo que quieras. En el futuro, 
espero que podamos agregar un par más de ``Groups`` a la lista. Un ``GroupMulti``
que es como el ``GroupSingle``, pero que puede contener hasta un número 
determinado de sprites (¿en algún tipo de búfer circular?). También un grupo 
súper renderizador que puede borrar la posición de los sprites sin necesitar 
una imagen de fondo para hacerlo (al tomar una copia de la pantalla antes de 
blittear). Quién sabe realmente, pero en el futuro podemos agregar más clases 
útiles a esta lista.

Nota de traducción: "rendering" se puede entender como el proceso de producir 
una imagen o animación a partir de datos digitales utilizando software de 
gráficos. La traducción puede ser "renderizado" o "procesamiento de imágenes".

Los Grupos de Renderizado
-------------------------

De lo analizado anteriormente, podemos ver que hay tres grupos diferentes de 
renderizado. Con ``RenderUpdates`` podríamos salirnos con la nuestra, pero 
agrega una sobrecarga que no es realmente necesaria para algo como un juego de 
desplazamiento. Así que acá tenemos un par de herramientas, elegí la adecuada 
para cada trabajo.

Para un juego del tipo de desplazamiento, donde el fondo cambia completamente 
en cada cuadro, obviamente necesitamos no necesitamos preocuparnos por los 
rectángulos de actualización de python en la llamada ``display.update()``. 
Definitvamente deberías ir con el grupo ``RenderPlain`` para administrar tu 
renderizado.

Para juegos donde el fondo es más estático, definitivamente no vas a querer 
que Pygame actualice la pantalla completa (ya que no es necesario). Este tipo 
de juegos generalmente implica borrar la posición anterior de cada objeto y 
luego dibujarlo en el lugar nuevo de cada cuadro. De esta manera solo estamos 
cambiando lo necesario. La mayoría de las veces solo querrás usar la clase 
``RenderUpdates`` acá. Dado que también querrás pasar la lista de cambios a 
la función ``display.update()``.

La clase ``RenderUpdates`` también hace un buen trabajo al minimizar las 
áreas superpuestas en la lista de rectángulos actualizados. Si la posición 
anterior y la actual de un objeto se superponen, las fusionará en un solo 
rectángulo. Combinado con el hecho de que maneja los objetos eliminados, 
esta es una poderosa clase ``Group``. Si has escrito un juego que administra 
los rectángulos modificados para los objetos en el juego, sabés que ésta es 
la causa de la gran cantidad de código desordenado en el juego. Especialmente, 
una vez que empiezas a agregar objetos que puedan ser eliminados en cualquier 
momento. Todo este trabajo se reduce a los monstruosos métodos 
``clear()`` y ``draw()``. Además, con la verificación de superposición, es 
probable que sea más rápido que cuando lo hacías manualmente.

También hay que tener en cuenta que no hay nada que impida mezclar y combinar 
estos grupos de renderizado en tu juego. Definitivamente deberías usar 
múltiples grupos de renderizado cuando quieras hacer capas con tus sprites. 
Además, si la pantalla se divide en varias secciones, ¿quizás cada sección 
de la pantalla debería usar un grupo de representación adecuado?


Detección de Colisiones
-----------------------

El módulo de sprites también viene con dos funciones de detección de 
colisiones muy genéricas. Para juegos más complejos, estos realmente 
no funcionarán adecuadamente, pero fácilmente se puede obtener el código 
fuente y modificarlos según sea necesario. 

Acá hay un resumen de lo que son y lo que hacen.

  :func:`spritecollide(sprite, group, dokill) -> list <pygame.sprite.spritecollide>`

    Esto verifica las colisiones entre un solo sprite y los sprites en un grupo. 
    Requiere un atributo "rect" para todos los sprites usados. Devuelve una lista 
    de todos los sprites que se superponen con el primer sprite. El argumento 
    "dokill" es un argumento booleano. Si es verdadero, la funcion llamará al 
    método ``kill()`` para todos los sprites. Esto significa que la última 
    referencia para cada sprite esté probablemente en la lista devuelta. Una vez 
    que la lista desaparece, también lo hacen los sprites. Un ejemplo rápido del 
    uso de este bucle ::

      >>> for bomb in sprite.spritecollide(player, bombs, 1):
      ...     boom_sound.play()
      ...     Explosion(bomb, 0)

    Esto encuentra todos los sprites en el grupo "bomb" que chocan con el jugador.
    Debido al argumento "dokill", elimina todas las bombas estrelladas. Por cada 
    bomba que chocó, se reproduce el sonido "boom" y crea un nuevo ``Explosion``
    donde estaba la bomba. (Tengan en cuenta que la clase ``Explosion`` acá sabe 
    agregar cada instancia de la clase apropiada, por lo que no necesitamos 
    almacenarla en una variable, esa última línea puede sonar un poco rara para 
    los programadores python.)

  :func:`groupcollide(group1, group2, dokill1, dokill2) -> dictionary <pygame.sprite.groupcollide>`

    Esto es similar a la función ``spritecollide``, pero un poco más compleja. 
    Comprueba las colisiones de todos los sprites de un grupo con los sprites de 
    otro grupo. Hay un argumento ``dokill`` para los sprites en cada lista. Cuando 
    ``dokill1`` es verdadero, los sprites que colisionan en ``group1`` serán 
    ``kill()`` (matados). Cuando ``dokill2`` es verdaero, vamos a tener el mismo 
    resultado para el ``group2``. El diccionario que devuelve funciona así; cada 
    clave (keys) en el diccionario es un sprite de ``group1`` que tuvo una colisión. 
    El valor de esa clave es una lista de los sprites con los que chocó. Quizás otra 
    muestra de código lo explique mejor. ::

      >>> for alien in sprite.groupcollide(aliens, shots, 1, 1).keys()
      ...     boom_sound.play()
      ...     Explosion(alien, 0)
      ...     kills += 1

    Este código comprueba las colisiones entre las balas de los jugadores y todos 
    los aliens con los que podrían cruzarse. En este caso, solo iteramos las 
    claves (keys) del diccionario, pero podríamos recorrer también los ``values()`` 
    o ``items()`` si quisiéramos hacer algo con los disparos específicos que 
    chocaron con extraterrestres. Si recorrieramos ``values()`` estaríamos 
    iterando listas que contienen sprites. El mismo sprite podría 
    aparecer más de una vez en estas iteraciones diferentes, ya que el mismo 
    'disparo' pudo haber chocado con múltiples aliens.

Estas son las funciones básicas de colisión que vienen con pygame. Debería 
ser fácil crear uno propio que quizás use algo diferente al atributo "rect".
¿O tal vez intentar ajustar un poco más tu código afectando directamente el 
objeto de colisión en lugar de construir una lista de colisiones? El código 
en las funciones de colisión de sprites está muy optimizado, pero podrías 
acelerarlo ligeramente eliminando algunas funcionalidaded que no necesitas.


Problemas Comunes
-----------------

Actualmente hay un problema principal que atrapa a los nuevos usuarios. Cuando 
derivas tus nueva clase de sprites con la base de Sprite, TENÉS que llamar al 
método ``Sprite._init_()`` desde el método ``_init_()`` de tu propia clase. Si 
te olvidás de llamar al método  ``Sprite.__init__()``, vas a obtener un error 
críptico, como este ::

    AttributeError: 'mysprite' instance has no attribute '_Sprite__g'


Extendiendo tus Propias Clases *(Avanzado)*
-------------------------------------------

Debido a problemas de velocidad, las clases de ``Group`` actuales intentan solo 
hacer exactamente lo que necesitan, y no manejar muchas situaciones generales. 
Si decidís que necesitás funciones adicionales, es posible que desees crear tu 
propia clase ``Group``.

Las clases ``Sprite`` y ``Gorup`` fueron diseñadas para ser extendidas, así que 
sentite libre de crear tus propias clases ``Group`` para hacer cosas 
especializadas. El mejor lugar para empezar es probablemente el código fuente 
real de python para el módulo de sprite. Mirar el actual grupo ``Sprite`` 
debería ser ejemplo suficiente de cómo crear el tuyo propio.

Por ejemplo, aquí está el código fuente para un ``Group`` de renderización que 
llama a un método ``render()`` para cada sprite, en lugar de simplemente blittear 
una variable de "imagen" de él. Como queremos que también maneje áreas 
actualizadas, empezaremos con una copia del grupo ``RenderUpdates`` original,
acá está el código ::

    class RenderUpdatesDraw(RenderClear):
        """call sprite.draw(screen) to render sprites"""
        def draw(self, surface):
            dirty = self.lostsprites
            self.lostsprites = []
            for s, r in self.spritedict.items():
                newrect = s.draw(screen) #Here's the big change
                if r is 0:
                    dirty.append(newrect)
                else:
                    dirty.append(newrect.union(r))
                self.spritedict[s] = newrect
            return dirty

A continuación hay más información acerca de cómo podés crear tus propios 
objetos ``Sprite`` y ``Group`` de cero.

Los objetos ``Sprite`` solo "requieren" dos métodos: "add_internal()" y 
"remove_internal()". Estos son llamados por la clase ``Group`` cuando están 
eliminando un sprite de sí mismos. Los métodos ``add_internal()`` y 
``remove_internal()`` tienen un único argumento que es un grupo. Tu ``Sprite`` 
necesitará alguna forma de realizar un seguimiento de los ``Groups`` a los que 
pertenece. Es probable que quieras intentar hacer coincidir los otros métodos 
y argumentos con la clase real de ``Sprites``, pero si no vas a usar esos 
métodos, seguro que no los necesitás.

Son casi los mismos requerimientos para crear tu propio ``Group``. De hecho, 
si observas la fuente, verás que el ``GroupSingle`` no está derivado de la 
clase ``Group``, simplemente implementa los mismos métodos, por lo que 
realmente no se puede notar la diferencia. De nuevo, necesitás un método 
"add_internal()" y "remove_internal()" para que los sprites llamen cuando 
quieren pertenecer o eliminarse a sí mismos del grupo. Tanto ``add_internal()`` 
como ``remove_internal()`` tienen un único argumento que es un sprite. El único 
requisito adicional para las clases ``Group`` es que tengan un atributo ficticio 
llamado "_spritegroup". No importa cuál sea el valor, en tanto el atributo esté 
presente. Las clases Sprite pueden buscar este atributo para determinar la 
diferencia entre un "grupo" y cualquier contenedor ordinario de python. (Esto 
es importante porque varios métodos de sprites pueden tomar un argumento de 
un solo grupo o una secuencia de grupos. Dado que ambos se ven similares, esta 
es la forma más flexible de "ver" la diferencia.)

Deberías pasar por el código para el módulo de sprite. Si bien el código está 
un poco "afinado", tiene suficientes comentarios para ayudarte a seguirlo. Hay 
incluso una sección de tareas para hacer en la fuente si tenés ganas de 
contribuir.
