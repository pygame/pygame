.. TUTORIAL:¡Ayuda! ¿Cómo Muevo Una Imagen?

.. include:: ../../reST/common.txt

*********************************************************
Tutoriales de Pygame - ¡Ayuda! ¿Cómo Muevo Una Imagen?
*********************************************************

¡Ayuda! ¿Cómo Muevo Una Imagen?
===============================

.. rst-class:: docinfo

:Autor: Pete Shinners
:Contacto: pete@shinners.org
:Traducción al español: Estefanía Pivaral Serrano

Muchas personas nueva en programación y gráficos tienen dificultades 
para descubrir cómo hacer que una imagen se mueva por la pantalla. Sin 
entender todos los conceptos puede resultar muy confuso. No sos la primera 
persona atrapada ahí, haré todo lo posible para que vayamos paso por paso.
Incluso, intentaremos terminar con métodos para mantener la eficiencia de 
tus animaciones. 

Tengan en cuenta que en este articulo no vamos a enseñar cómo programar en 
python, solo presentaremos algunos conceptos básicos de pygame.



Solo Píxeles en la Pantalla
---------------------------

Pygame tiene una Surface de visualización. Básicamente, esto es una 
imagen que está visible en la pantalla y la imagen está compuesta por 
píxeles. La forma principal de cambiar estos píxeles es llamando a la 
función blit(). Esto copia los píxeles de una imagen a otra.

Esto es lo primero que hay que entender. Cuando proyectás (blit) una 
imagen en la pantalla, lo que estás haciendo es simplemente cambiar el 
color de los píxeles. Los píxeles no se agregan ni se mueven, simplemente 
cambiamos el color de los píxeles que ya se encuentran en la pantalla.
Las imágenes que uno proyecta (blit) a la pantalla son también surfaces 
(superficies) en pygame pero no están conectadas de ninguna manera a la 
Surface de visualización. Cuando se proyectan en la pantalla, se copian 
en la visualización, pero aún mantenes una copia única del original. 

Luego de esta breve descripción, quizás ya puedas entender lo que se 
necesita para "mover" una imagen. En realidad, no movemos nada en 
absoluto. Lo que hacemos es simplemente proyectar (blit) la imagen 
en una nueva posición, pero antes de dibujar la imagen en la nueva 
posición, necesitamos "borrar" la anterior. De lo contrario, la 
imagen será visible en dos lugares de la pantalla. Al borrar 
rápidamente la imagen y volverla a dibujar en un nuevo lugar en la 
pantalla, logramos la "ilusión" de movimiento. 

A lo largo del tutorial, vamos a dividir este proceso en pasos más 
simples. Incluso explicaremos la mejor manera de tener múltiples imagenes 
moviendose por la pantalla. Probablemente ya tengas preguntas; por 
ejemplo, ¿cómo "borramos" la imagen antes de dibujarla en una nueva 
posición? Quizás todavía estás completamente perdido. Bueno, espero que 
el resto de este tutorial pueda aclarar las cosas.


Damos Un Paso Hacia Atrás
-------------------------

Es posible que el concepto de píxeles e imagenes sea aún un poco extraño. 
¡Buenas noticias! En las próximas secciones vamos a usar código que hace 
todo lo que queremos, solo que no usa píxeles. Vamos a crear una pequeña 
lista de python de 6 números, y vamos a imaginar que representa unos 
gráficos fantásticos que podemos ver en la pantalla. Podría ser de hecho 
sorprendente lo cerca que esto representa lo que haremos después con 
gráficos reales.

Entonces, comencemos creando nuestra lista de pantalla y completandola 
con un paisaje hermoso de 1s y 2s. ::

  >>> screen = [1, 1, 2, 2, 2, 1]
  >>> print(screen)
  [1, 1, 2, 2, 2, 1]


Ahora hemos creado nuestro fondo. No va a ser muy emocionante a menos que 
también dibujemos un jugador en la pantalla. Vamos a crear un héroe 
poderoso que se parezca al número 8. Vamos a ponerlo cerca de la mitad 
del mapa y veamos cómo se ve. ::

  >>> screen[3] = 8
  >>> print(screen)
  [1, 1, 2, 8, 2, 1]


Puede que esto haya sido tan lejos como hayas llegado si saltaste a hacer 
algo de programación gráfica con pygame. Tenés algunas cosas bonitas en 
la pantalla, pero no pueden moverse a ningun lado. Quizás ahora que 
nuestra pantalla es una lista de números, es más fácil ver cómo moverlo.


Hacer Mover al Héroe
--------------------

Antes de empezar a mover el personaje, necesitamos hacer el seguimiento 
de algún tipo de posición para él. En la última sección, cuando lo 
dibujamos, simplemente elegimos una posición al arbitraria. Esta vez 
hagámoslo de forma más oficial. ::

  >>> playerpos = 3
  >>> screen[playerpos] = 8
  >>> print(screen)
  [1, 1, 2, 8, 2, 1]


Ahora es bastante fácil moverlo en una nueva posición. Podemos 
simplemente cambiar el valor de playerpos (posición del player)
y dibujarlo en la pantalla nuevamente. ::

  >>> playerpos = playerpos - 1
  >>> screen[playerpos] = 8
  >>> print(screen)
  [1, 1, 8, 8, 2, 1]


Whoops. Ahora podemos ver dos héroes. Uno en la vieja posición, y otro en 
la nueva posición. Esta es exactamente la razón por la que necesitamos 
"borrar" al héroe en la posición anterior antes de dibujarlo en la nueva 
posición. Para borrarlo, necesitamos cambiar ese valor en la lista de nuevo
al valor que tenía antes de que el héroe lo reemplazara. Eso significa que 
debemos hacer un seguimiento de los valores en la pantalla antes que el 
héroe estuviera allí. Hay varias formas de hacerlo, pero la más fácil suele 
ser mantener una copia separada del fondo de la pantalla. Esto significa 
que tenemos que hacer cambios en nuestro pequeño juego.


Crear un Mapa
-------------

Lo que queremos hacer es crear una lista separada que llamaremos nuestro 
fondo (background). Vamos a crear el fondo para que se vea como lo hacía 
nuestra pantalla original, con 1s y 2s. Luego, vamos a copiar cada item 
del fondo a la pantalla. Después de eso, podemos finalmente dibujar nuestro 
héroe en la pantalla.  ::

  >>> background = [1, 1, 2, 2, 2, 1]
  >>> screen = [0]*6                         #una nueva pantalla en blanco
  >>> for i in range(6):
  ...     screen[i] = background[i]
  >>> print(screen)
  [1, 1, 2, 2, 2, 1]
  >>> playerpos = 3
  >>> screen[playerpos] = 8
  >>> print(screen)
  [1, 1, 2, 8, 2, 1]


Puede parecer mucho trabajo extra. No estamos muy lejos de donde estabamos 
la última vez que tratamos de hacer que se moviera. Pero esta vez tenemos 
la información extra que necesitamos para moverlo correctamente.


Hacer Mover al Héroe (Toma 2)
-----------------------------

Esta vez va a ser fácil mover al héroe. Primero borramos el héroe de su 
antigua posición. Esto lo podemos hacer copiando el valor correcto del 
fondo a la pantalla. Luego, dibujamos el personaje en la nueva posición 
en la pantalla.


  >>> print(screen)
  [1, 1, 2, 8, 2, 1]
  >>> screen[playerpos] = background[playerpos]
  >>> playerpos = playerpos - 1
  >>> screen[playerpos] = 8
  >>> print(screen)
  [1, 1, 8, 2, 2, 1]


Ahí está. El héroe se ha movido un lugar hacia la izquierda.
Podemos usar este mismo código para moverlo una vez más hacia la izqueirda.
::

  >>> screen[playerpos] = background[playerpos]
  >>> playerpos = playerpos - 1
  >>> screen[playerpos] = 8
  >>> print(screen)
  [1, 8, 2, 2, 2, 1]


Excelente! Esto no es exactamente lo que llamarías una animación fluida, 
pero con unos pequeños cambios, haremos que esto funcione directamente 
con gráficos en la pantalla.


Definición: "blit"
------------------

En las próximas secciónes transformaremos nuestro programa, de usar 
listas pasará a usar gráficos reales en la pantalla. Al mostrar los 
gráficos vamos a usar el término **blit** frecuentemente. Si sos nuevo 
en el trabajo gráfico, probablemente no estés familiarizado con este 
término común. 

BLIT: Basicamente, blit significa copiar gráficos de una imagen a otra.
Una definición más formal es copiar una matriz de datos a un mapa de 
bits. 'Blit' se puede pensar como *asignar* píxeles. Es similar a 
establecer valores en nuestra lista de pantalla más arriba, blitear
asigna el color de los píxeles en nuestra imagen. 

Otras bibliotecas gráficas usarán la palabra *bitblt*, o solo *blt*, 
pero están hablando de lo mismo. Es básicamente copiar memoria de un 
lugar a otro. En realidad, es un poco más avanzado que simpleente 
copiar la memoria, ya que necesita manejar cosas como formatos de 
píxeles, recortes y separaciones de líneas de exploración. Los 
mezcladores (blitters) avanzados también pueden manejar cosas como 
la transparecia y otros efectos especiales.


Pasar de la Lista a la Pantalla
-------------------------------

Tomar el código que vemos en los ejemplos anteriores y hacerlo funcionar con 
pygame es muy sencillo. Simulemos que tenemos cargados algunos gráficos 
bonitos y los llamamos "terrain1", "terrain2" y "hero". Donde antes 
asignamos números a una lista, ahora mostramos (blit) gráficos en la pantalla.
Otro gran cambio, en vez de usar posiciones como un solo índice (0 through 5), 
ahora necesitamos una coordenada bidimensional. Fingiremos que uno de los 
gráficos de nuestro juego tiene 10 píxeles de ancho. ::

  >>> background = [terrain1, terrain1, terrain2, terrain2, terrain2, terrain1]
  >>> screen = create_graphics_screen()
  >>> for i in range(6):
  ...     screen.blit(background[i], (i*10, 0))
  >>> playerpos = 3
  >>> screen.blit(playerimage, (playerpos*10, 0))


Hmm, ese código debería parecerte muy familiar, y con suerte, más importante;
el código anterior debería tener un poco de sentido. Con suerte, mi 
ilustración de configurar valores simples en una lista muestra la similitud 
de establecer píxeles en la pantalla (con blit). La única parte que es 
realmente trabajo extra es convertir la posición del jugador en coordenadas 
en la pantalla. Por ahora, solo usamos un :code:`(playerpos*10, 0)` crudo, 
pero ciertamente podemos hacer algo mejor que eso. Ahora, movamos la imagen 
del jugador sobre un espacio. Este código no debería tener sorpresas. ::

  >>> screen.blit(background[playerpos], (playerpos*10, 0))
  >>> playerpos = playerpos - 1
  >>> screen.blit(playerimage, (playerpos*10, 0))


Ahí está. Con este código, hemos mostrado cómo visualizar un fondo simple 
con la imagen de un héroe. Luego, hemos movido correctamente a ese héroe 
un espacio hacia la izquierda. Entonces, ¿dónde vamos desde aquí? Bueno, 
para empezar, el código es todavía un poco extraño. Lo primero que queremos 
hacer es encontrar una forma más límpia de representar el fondo y la posición 
del jugador. Luego, quizás una animación un poco más real y fluida.

Coordenadas de Pantalla
-----------------------

Para posicionar un objeto en la pantalla, necesitamos decirle a la función 
blit () dónde poner la imagen. En pygame siempre pasamos las posiciones como 
una coordenada (X,Y). Esto reprenseta el número de píxeles a la derecha y el 
número de pixeles hacia abajo, para colocar la imagen. La esquina superior 
izquierda de la Surface es la coordenada (0,0). Moverse un poco hacia la 
derecha sería (10, 0), y luego moverse hacia abajo en la misma proporción 
sería (10,10). Al hacer blit, el argumento de posición representa dónde se 
debe colocar la esquina superior izquierda de la fuente en el destino. 

Pygame viene con un conveniente container para estas coordenadas, este es 
un Rect. El Rect básicamente representa un área rectangular en estas 
coordenadas. Tiene una esquina superior izquierda y un tamaño. El Rect 
viene con muchos métodos convenientes que ayudan a moverlo y posicionarlo.
En nuestros próximos ejemplos representaremos las posiciones de nuestros 
objetos con Rects.

También, hay que tener en cuenta que muchas funciones en pygame esperan 
argumentos Rect. Todas estas funciones pueden también aceptar una simple 
tupla de 4 elementos (izquierda, arriba, ancho, alto). No siempre es 
necesario usar estos objetos Rect, pero mayormente querrás hacerlo. 
Además la función blit () puede aceptar un Rect como su argumento de 
posición, simplemente usa la esquina superior izquierda del Rect como 
su posición real. 


Cambiando el Fondo
------------------

En todas nuestras secciones anteriores, hemos estado almacenando el fondo 
como una lista de diferentes tipos de terrenos. Esa es una buena forma de 
crear un juego basado en mosaicos, pero queremos un desplazamiento fluido.
Para hacerlo un poco más fácil, vamos a cambiar el fondo a una imagen única 
que cubra toda la pantalla. De esta forma, cuando queremos "borrar" nuestros 
objetos (antes de volver a dibujarlos) solo necesitamos blitear la sección 
del fondo borrado en la pantalla.

Al pasar a blit un tercer argumento Rect de manera opcional, le decimos que 
use esa subsección de la imagen de origen. Lo verás en uso a continuación 
mientras borramos la imagen del jugador.

Nótese que ahora, cuando terminamos de dibujar en la pantalla, llamamos 
pygame.display.update() que mostrará todo lo que hemos dibujado en la 
pantalla. 

Movimiento Fluido
-----------------

Para hacer que algo parezca moverse suavemente, vamos a querer moverlo 
únicamente un par de píxeles a la vez. Acá está el código para hacer que 
un objeto se mueva suavemente a través de la pantalla. Según lo que 
ya sabemos, esto debería parecer bastante simple. ::

  >>> screen = create_screen()
  >>> player = load_player_image()
  >>> background = load_background_image()
  >>> screen.blit(background, (0, 0))        #dibujar el fondo
  >>> position = player.get_rect()
  >>> screen.blit(player, position)          #dibujar el jugador
  >>> pygame.display.update()                #y mostrarlo todo
  >>> for x in range(100):                   #animar 100 cuadros
  ...     screen.blit(background, position, position) #borrar
  ...     position = position.move(2, 0)     #mover el jugador
  ...     screen.blit(player, position)      #dibujar nuevo jugador
  ...     pygame.display.update()            #y mostrarlo todo
  ...     pygame.time.delay(100)             #detener el programa por 1/10 segundos


Ahí está. Este es todo el código que es necesario para animar suavemente un 
objeto a través de la pantalla. Incluso podemos usar un bonito paisaje de 
fondo. Otro beneficio de hacer el fondo de esta manera es que la imagen 
para el jugador puede tener transaprencias o secciones recortadas y aún 
así se dibujará de correctamente sobre el fondo (un bonus gratis).

También hicimos una llamada a pygame.time.delay() al final de nuestro bucle 
(loop) anterior. Esto ralentiza un poco nuestro programa; de lo contrario, 
podría ejecutarse tan rápido que sería posible no verlo.


Entonces, ¿Qué Sigue?
---------------------

Bueno, aquí lo tenemos. Esperemos que este artículo haya cumplido con lo 
prometido. Aún así, en este punto, el código no está realmente listo para ser 
el próximo juego más vendido. ¿Cómo hacer para tener múltiples objetos 
moviendose fácilmente? ¿Qué son exactamente esas misteriosas funciones como 
load_player_image()? También necesitamos una forma de obtener una entrada simple 
de usuario y un bucle de más de 100 cuadros. Tomaremos el ejemplo que 
tenemos acá, y lo convertiremos en una creación orientada a objetos que podría 
enorgullecería a mamá.


Primero, Funciones Misteriosas
------------------------------

Se puede encontrar información completa de este tipo de funciones en otros 
tutoriales y referencia. El módulo pygame.image tiene una función load() 
que hará lo que queramos. Las líneas para cargar las imágenes deberían 
llegar a ser así. ::

  >>> player = pygame.image.load('player.bmp').convert()
  >>> background = pygame.image.load('liquid.bmp').convert()


Podemos ver que es bastante simple, la función load() solo toma un 
nombre de archivo y devuelve una nueva Surface con la imagen cargada. 
Después de cargar, hacemos una llamada al método de Surface, conver().
'Convert' nos devuelve una nueva Surface de la imagen, pero ahora 
convertida al mismo formato de píxel que nuestra pantalla. Dado que 
las imagenes serán del mismo formato que la pantalla, van a blittear 
muy rápidamente. 
Si no usaramos 'convert', la función blit() es más lenta, ya que tiene 
que convertir de un tipo de píxel a otro a medida que avanza.

Es posible que hayas notado que ambas load() y convert() devuelven una 
nueva Surface. Esto significa que estamos realmente creando dos Surfaces 
en cada una de estas líneas. En otros lenguajes de programación, esto da 
como resultado una fuga de memoria (no es algo bueno). Afortunadamente, 
Python es lo suficientemente inteligente como manejar esto, y pygame 
limpiará adecuadamente la Surface que terminamos sin usar.

La otra función misteriosa que vimos en el ejemplo anterior fue 
create_screen(). En pygame es simple de crear una nueva ventana para 
gráficos. El código para crear una surface de 640x480 está a 
continuación. Al no pasar otros argumentos, pygame solo eligirá la mejor 
profundidad de color y formato de píxel para nosotros. ::

  >>> screen = pygame.display.set_mode((640, 480))


Manejo de Algunas Entradas
--------------------------

Necesitamos desesperadamente cambiar el bucle principal para que buscar 
cualquier entrada de usuario (como cuando el usuario cierra la ventana). 
Necesitamos agregar "manejo de eventos" a nuestro programa. Todos los 
programas gráficos usan este diseño basado en eventos. El programa 
obtiene eventos como "tecla presionada" o "mouse movido" de la computadora. 
Entonces el programa responde a los diferentes eventos. Así es como debería 
ser el código. En lugar de un bucle de 100 cuadros, seguiremos en el bucle 
hasta que el usuario nos pida que nos detengamos.::

  >>> while True:
  ...     for event in pygame.event.get():
  ...         if event.type in (QUIT, KEYDOWN):
  ...             sys.exit()
  ...     move_and_draw_all_game_objects()


Lo que simplemente hace este código es, en primer lugar ejecuta el bucle 
para siempre, luego verifica si hay algún evento del usuario. Salimos del 
programa si el usuario presiona el teclado o el botón de cerrar en la 
ventana. Después de revisar todos los eventos, movemos y dibujamos nuestros 
objetos del juego. (También los borraremos antes de moverlos.)


Mover Imágenes Múltiples
------------------------

Esta es la parte en que realmente vamos a cambiar las cosas. Digamos que 
queremos 10 imágenes diferentes moviéndose en la pantalla. Una buena forma 
de manejar esto es usando las CLASES de python. Crearemos una CLASE que 
represente nuestro objeto de juego. Este objeto tendrá una función para 
moverse solo y luego podemos crear tantos como queramos. Las funciones 
para dibujar y mover el objeto necesitan funcionar de una manera en que 
muevan solo un cuadro (o un paso) a la vez. Acá está el código de python 
para crear nuestra clase. ::

  >>> class GameObject:
  ...     def __init__(self, image, height, speed):
  ...         self.speed = speed
  ...         self.image = image
  ...         self.pos = image.get_rect().move(0, height)
  ...     def move(self):
  ...         self.pos = self.pos.move(0, self.speed)
  ...         if self.pos.right > 600:
  ...             self.pos.left = 0


Entonces, tenemos dos funciones en nuestra clase. La función init (inicializar)
construye nuestro objeto, posiciona el objeto y establece su velocidad. El 
método move (mover) mueve el objeto un paso. Si se va demasiado lejos, mueve el 
objeto de nuevo hacia la izquierda.


Ensamblando Todo
----------------

Ahora con nuestra nueva clase objeto, podemos montar el juego completo. 
Así es como se verá la función principal para nuestro programa. ::

  >>> screen = pygame.display.set_mode((640, 480))
  >>> player = pygame.image.load('player.bmp').convert()
  >>> background = pygame.image.load('background.bmp').convert()
  >>> screen.blit(background, (0, 0))
  >>> objects = []
  >>> for x in range(10):                    #crear 10 objetos</i>
  ...     o = GameObject(player, x*40, x)
  ...     objects.append(o)
  >>> while True:
  ...     for event in pygame.event.get():
  ...         if event.type in (QUIT, KEYDOWN):
  ...             sys.exit()
  ...     for o in objects:
  ...         screen.blit(background, o.pos, o.pos)
  ...     for o in objects:
  ...         o.move()
  ...         screen.blit(o.image, o.pos)
  ...     pygame.display.update()
  ...     pygame.time.delay(100)


Y ahí está. Este es el código que necesitamos para animar 10 objetos en la 
pantalla. El único punto que podría necesitar explicación son los dos bucles 
(loops) que usamos para borrar todos los objetos y dibujar todos los objetos. 
Para hacer las cosas correctamente, necestamos borrar todos los objetos antes 
de dibujar alguno de ellos. En nuestro ejemplo puede que no importe pero 
cuando los objetos se superponen, el uso de dos bucles (loops) como estos se 
vuelve muy importante.


De Ahora En Más, Estás Por Tu Cuenta
------------------------------------

Entonces, ¿qué será lo siguiente en tu camino de aprendizaje? Bueno, 
primero jugar un poco con este ejemplo. La versión ejecutable completa 
de este ejemplo está disponible en los directorios de ejemplos de pygame. 
Está en el ejemplo llamado :func:`moveit.py <pygame.examples.moveit.main>` .
Dale una mirada al código y jugá con él, correlo, aprendelo.

Algunas cosas en las que quizás quieras trabajar es en tener más de un tipo 
de objeto. Encontrar una manera de "eliminar" objetos limpiamente cuando ya 
no quieras mostrarlos. También, actualizar el llamado (call) display.update()
para pasar una lista de las áreas en pantalla que han cambiado. 

En pygame hay otros tutoriales y ejemplos que cubren estos temas. 
Así que cuando estés listo para seguir aprendiendo, seguí leyendo. :-)

Por último, podés unirte a la lista de correos de pygame o al chatroom 
con total libertad para consultar dudas al respecto. Siempre hay personas 
disponibles que están dispuestas a ayudar con estos temas.

Finalmente, divertite, para eso son los juegos!
