.. TUTORIAL:Descripciones línea por línea del ejemplo del chimpancé
.. include:: common.txt

***************************************************************
  Tutorial de Pygame - Ejemplo del Chimpancé, Línea Por Línea
***************************************************************


Chimpancé, Línea Por Línea
==========================
.. rst-class:: docinfo

:Autor: Pete Shinners. Traducción al español: Estefania Pivaral Serrano
:Contacto: pete@shinners.org

.. toctree::
   :hidden:

   ../chimp.py


Introducción
------------

Entre los ejemplos de *pygame* hay un ejemplo simple llamado "chimp" (chimpancé).
Este ejemplo simula un mono golpeable que se mueve alrededor de la pantalla 
con promesas de riquezas y recomepensas. El ejemplo en sí es muy simple y acarrea
poco código de comprobación de error. Como modelo de programa, Chimp demuestra muchas 
de las bondades de pygame, como por ejemplo crear una ventana, cargar imágenes
y sonidos, representar texto, y manejo de eventos básicos y del mouse.

El programa y las imagenes se pueden encontrar dentro de la fuente estándar
de distribución de pygame. Se puede ejecutar al correr `python -m pygame.examples.chimp` 
en la terminal.

Este tutorial atravesará el código bloque a bloque, explicando cómo 
funciona el mismo. Además, se hará mención de cómo se puede 
mejorar el código y qué errores de comprobación podrían ser de ayuda.

Este tutorial es excelente para aquellas personas que están buscando
una primera aproximación a códigos de *pygame*. Una vez que *pygame* 
esté completamente instalado, podrás encontrar y ejecutar la demostración
del chimpancé para ti mismo en el directorio de ejemplos.

.. container:: fullwidth leading trailing

   .. rst-class:: small-heading

   (no, este no es un anuncio, es una captura de pantalla)

   .. image:: chimpshot.gif
      :alt: chimp game banner

   :doc:`Full Source <../chimp.py>`


Importación de Módulos
----------------------

Este es el código que importa todos los módulos necesarios del programa.
Este código también comprueba la disponibilidad de algunos de los módulos opcionales 
de pygame. ::

    # Import Modules
    import os
    import pygame as pg

    if not pg.font:
        print("Warning, fonts disabled")
    if not pg.mixer:
        print("Warning, sound disabled")

    main_dir = os.path.split(os.path.abspath(__file__))[0]
    data_dir = os.path.join(main_dir, "data")


Primero, se importa el módulo estándar de python "os" (sistema operativo).
Esto permite hacer cosas como crear rutas de archivos independientes de la 
platforma.

En la sigueinte línea, se importa el paquete de pygame. En nuestro caso,
importamos pygame como ``pg``, para que todas las funciones de pygame puedan
ser referenciadas desde el espacio de nombres ``pg``.

Algunos de los módulos de pygame son opcionales, y si no fueran encontrados,
la evaluación será ``False``. Es por eso que decidimos mostrar (print) un agradable
mensaje de advertencia si los módulos :mod:`font<pygame.font>` o 
:mod:`mixer <pygame.mixer>` no están disponibles.
(Aunque estos solo podrían no estar disponibles en situaciones poco comunes).

Finalmente, se preparan dos rutas que serán usadas para el resto del código.
Una de ellas es ``main_dir``, que usa el módulo `os.path` y la variable  `__file__` 
asignada por Python para localizar el archivo de juegos de python, y extraer la carpeta 
desde esa ruta. Luego, ésta prepara la ruta ``data_dir`` para indicarle a las 
funciones de carga exactamente dónde buscar.


Carga de Recursos
-----------------

A continuación, se presentan dos funciones que se pueden usar para cargar imágenes y sonidos.
En esta sección examinaremos cada función individualmente. ::

    def load_image(name, colorkey=None, scale=1):
        fullname = os.path.join(data_dir, name)
        image = pg.image.load(fullname)

        size = image.get_size()
        size = (size[0] * scale, size[1] * scale)
        image = pg.transform.scale(image, size)

        image = image.convert()
        if colorkey is not None:
            if colorkey == -1:
                colorkey = image.get_at((0, 0))
            image.set_colorkey(colorkey, pg.RLEACCEL)
        return image, image.get_rect()


Esta función toma el nombre de la imagen a cargar. Opcionalmente, también
toma un argumento que puede usar para definir la clave de color (colorkey) de 
la imagen, y un argumento para determinar la escala de la imagen.
La clave de color se usa en la gráfica para representar un color en la imagen
que es transparente.

Lo que esta función hace en primera instancia es crearle al archivo un nombre
de ruta completo.
En este ejemplo, todos los recursos están en el subdirectorio "data". Al usar
la función `os.path.join`, se creará el nombre de ruta para cualquier plataforma
en que se ejecute el juego.

El paso siguiente es cargar la imagen usando la función :func:`pygame.image.load`.
Luego de que la imagen se cargue, llamamos a la función
`convert()`. Al hacer esto se crea una nueva copia del Surface y convierte
su formato de color y la profundidad, de tal forma que coincida con el mostrado.
Esto significa que el dibujo (blitting) de la imagen a la pantalla sucederá 
lo más rápido posible.

Luego, usando la función :func:`pygame.transform.scale` se definirá el tamaño de 
la imagen. Esta función toma una Surface y el tamaño al cual se debería adecuar.
Para darle tamaño con números escalares, se puede tomar la medida y determinar las
dimensiones *x* e *y* con número escalar.

Finalmente, definimos la clave de color para la imagen. Si el usuario suministró
un valor para el parametro de la clave de color, usamos ese valor como la clave
de color de la imagen. Usualmente, éste sería un valor de color RGB 
(red-green-blue = rojo-verde-azul), como (255, 255, 255) para el color blanco. 
También es posible pasar el valor -1 como la clave de color. En este caso, la 
función buscará el color en el píxel de arriba a la izquierda de la imagen, 
y lo usará para la clave de color. ::

    def load_sound(name):
        class NoneSound:
            def play(self):
                pass

        if not pg.mixer or not pg.mixer.get_init():
            return NoneSound()

        fullname = os.path.join(data_dir, name)
        sound = pg.mixer.Sound(fullname)

        return sound


La anterior, es la función para cargar un archivo de sonido. Lo primero que hace 
esta función es verificar si el módulo :mod:`pygame.mixer` se importó correctamente.
En caso de no ser así, la función va a devolver una instancia de reproducción de un
sonido de error. Esto obrará como un objeto de Sonido normal para que el juego se 
ejecute sin ningún error de comprobación extra.

Esta funcion es similar a la función de carga de imagen, pero maneja diferentes problemas. 
Primero, creamos una ruta completa al sonido de la imagen y cargamos el archivo 
de sonido. Luego, simplemente devolvemos el objeto de Sonido cargado.



Clases de Objetos para Juegos
-----------------------------

En este caso creamos dos clases (classes) que representan los objetos en nuestro juego.
Casi toda la logica del juego se organiza en estas dos clases. A continuación 
las revisaremos de a una. ::

    class Fist(pg.sprite.Sprite):
        """moves a clenched fist on the screen, following the mouse"""

        def __init__(self):
            pg.sprite.Sprite.__init__(self)  # call Sprite initializer
            self.image, self.rect = load_image("fist.png", -1)
            self.fist_offset = (-235, -80)
            self.punching = False

        def update(self):
            """move the fist based on the mouse position"""
            pos = pg.mouse.get_pos()
            self.rect.topleft = pos
            self.rect.move_ip(self.fist_offset)
            if self.punching:
                self.rect.move_ip(15, 25)

        def punch(self, target):
            """returns true if the fist collides with the target"""
            if not self.punching:
                self.punching = True
                hitbox = self.rect.inflate(-5, -5)
                return hitbox.colliderect(target.rect)

        def unpunch(self):
            """called to pull the fist back"""
            self.punching = False


En este caso, creamos una clase (class) que representa el puño del jugador. Esta se 
deriva de la clase `Sprite` incluida en el módulo :mod:`pygame.sprite`. La 
función `__init__` es llamada cuando se crean nuevas instancias de este clase.
Esto le permite a la función `__init__` del Sprite preparar nuestro objeto para
ser usado como una imagen (sprite).
Este juego usa uno de los dibujos de sprite de la clase de Grupo. Estas clases
pueden dibujar sprites que tienen un atributo "imagen" y uno "rect". Al cambiar
simplemente estos dos atributos, el compilador (renderer) dibujará la imagen actual 
en la posición actual.

Todos los sprites tienen un método `update()`. Esta función es tipicamente 
llamada una vez por cuadro. Es en esta función donde se debería colocar el código 
que mueva y actualice las variables para el sprite. El método de `update()` para el 
movimiento del puño, mueve el puño al lugar donde se encuentre el puntero del mouse. 
Asímismo, compensa sutilmente la posición del puño sobre el objeto, si el puño está 
en condición de golpear.


Las siguientes dos funciones `punch()` y `unpunch()` cambian la condición de 
golpeado del puño. El método `punch()` también devuelve un valor verdadero si
el puño está chocando con el sprite objetivo. ::

    class Chimp(pg.sprite.Sprite):
        """moves a monkey critter across the screen. it can spin the
        monkey when it is punched."""

        def __init__(self):
            pg.sprite.Sprite.__init__(self)  # call Sprite intializer
            self.image, self.rect = load_image("chimp.png", -1, 4)
            screen = pg.display.get_surface()
            self.area = screen.get_rect()
            self.rect.topleft = 10, 90
            self.move = 18
            self.dizzy = False

        def update(self):
            """walk or spin, depending on the monkeys state"""
            if self.dizzy:
                self._spin()
            else:
                self._walk()

        def _walk(self):
            """move the monkey across the screen, and turn at the ends"""
            newpos = self.rect.move((self.move, 0))
            if not self.area.contains(newpos):
                if self.rect.left < self.area.left or self.rect.right > self.area.right:
                    self.move = -self.move
                    newpos = self.rect.move((self.move, 0))
                    self.image = pg.transform.flip(self.image, True, False)
            self.rect = newpos

        def _spin(self):
            """spin the monkey image"""
            center = self.rect.center
            self.dizzy = self.dizzy + 12
            if self.dizzy >= 360:
                self.dizzy = False
                self.image = self.original
            else:
                rotate = pg.transform.rotate
                self.image = rotate(self.original, self.dizzy)
            self.rect = self.image.get_rect(center=center)

        def punched(self):
            """this will cause the monkey to start spinning"""
            if not self.dizzy:
                self.dizzy = True
                self.original = self.image


Si bien la clase (class) `Chimp` está haciendo un poco más de trabajo que el 
puño, no resulta mucho más complejo. Esta clase moverá al chimpancé hacia adelante
y hacia atrás, por la pantalla. Cuando el mono es golpeado, él girará 
con un efecto de emoción. Esta clase también es derivada de la base de clases 
:class:`Sprite <pygame.sprite.Sprite>` y es iniciada de igual manera que el puño. 
Mientras se inicia, la clase también establece el atributo "area" para que sea
del tamaño de la pantalla de visualización.

La función `update` para el chimpancé simplemente se fija en el estado actual
del mono. Esta puede ser "dizzy" (mareado), la cual sería verdadera si el mono 
está girando a causa del golpe. La función llama al método `_spin` o `_walk`.
Estas funciones son prefijadas con un guión bajo, lo cual en el idioma estándar
de python sugiere que estos métodos deberían ser solo usados por la clase `Chimp`.
Podríamos incluso hasta escribirlas con un doble guión bajo, lo cual indicaría a 
python que realmente intente hacerlas un método privado, pero no necesitamos tal
protección. :)

El método `_walk` crea una nueva posición para el mono al mover el 'rect' actual
al centro del puño. Si la nueva posición se cruza hacia afuera del área
de visualización de la pantalla, el movimiento del puño da marcha atrás.
También imita la imagen usando la función :func:`pygame.transform.flip`. Este es
un efecto crudo que hace que el mono se vea como si estuviera cambiando de
dirección.

El método `_spin` es llamado cuando el mono está actualmente en estado "dizzy"
(mareado). El atributo 'dizzy' es usado para guardar el monto de rotación. 
Cuando el mono ha rotado por completo en su eje (360 grados) se resetea la
imagen a la versión original no rotada. Antes de llamar a la función
:func:`pygame.transform.rotate`, verás que el código hace una referencia local
a la función simplemente llamanda "rotate". No hay ncesidad de hacer eso en 
este ejemplo, aquí fue realizada para mantener la siguiente línea un poco 
más corta. Notese que al llamar a la función `rotate`, se está siempre rotando 
la imagen original del mono. Cuando rotamos, hay un pequeña pérdida de calidad.
Rotar repetidamente la misma imagen genera que la calidad se deteriore cada vez
más. Esto se debe a que las esquinas de la imagen van a haber sido rotadas de más,
causando que la imagen se haga más grande. Nos aseguramos que la nueva imagen
coincida con el centro de la vieja imagen, para que de esta forma se rote sin
moverse.

El último método es `punched()` el cual indica al sprite que entre en un estado de 
mareo. Esto causará que la imagen empice a girar. Además, también crea una copia de 
la actual imagen llamada "original".


Inicializar Todo
----------------

Antes de poder hacer algo con pygame, necesitamos asegurarnos que los módulos
estén inicializados. En este caso, vamos a abrir también una simple ventana de
gráficos.
Ahora estamos en la función `main()` del programa, la cual ejecuta todo. ::


    pg.init()
    screen = pg.display.set_mode((1280, 480), pg.SCALED)
    pg.display.set_caption("Monkey Fever")
    pg.mouse.set_visible(False)

La primera línea para inicializar *pygame* realiza algo de trabajo por nosotros.
Verifica a través del módulo importado *pygame* e intenta inicializar cada uno de
ellos. Es posible volver y verificar que los módulos que fallaron al iniciar, pero
no vamos a molestarnos acá con eso. También es posible tomar mucho más control
e inicializar cada módulo en especifico, uno a uno. Ese tipo de control no es 
necesario generalmente, pero está disponible en caso de ser deseado.

Luego, se configura el modo de visualización de gráficos. Notse que el módulo 
:mod:`pygame.display` es usado para controlar todas las configuraciones de
visualización. En este caso nosotros estamos buscando una ventana 1280x480,
con ``SCALED``, que es la señal de visualización (display flag)
Esto aumenta proporcionalmente la ventana de visualización (display) más grande que la 
ventana. (window)

Por último, establecemos el título de la ventana y apagamos el cursor del mouse
para nuestra ventana. Es una acción básica y ahora tenemos una pequeña ventana negra
que está lista para nuestras instrucciones (bidding) u ofertas. Generalmente, el cursor 
se mantiene visible por default, asi que no hay mucha necesidad de realmente
establecer este estado a menos que querramos esconderlo.


Crear el Fondo
--------------

Nuestro programa va a tener un mensaje de texto en el fondo. Sería bueno
crear un único surface que represente el fondo y lo use repetidas veces.
El primer paso es crear el Surface. ::

    background = pg.Surface(screen.get_size())
    background = background.convert()
    background.fill((170, 238, 187))

Esto crea el nuevo surface, que en nuestro caso, es del mismo tamaño que
la ventana de visualización. Notese el llamado extra a `convert()` luego 
de crear la Surface. La función `convert()` sin argumentos es para asegurarnos
que nuestro fondo sea del mismo formato que la ventana de visualización,
lo cual nos va a brindar resultados más rápidos.

Lo que nosotros hicimos también, fue rellenar el fondo con un color verduzco. 
La función `fill()` suele tomar como argumento tres instancias de colores RGB, 
pero soporta muchos formatos de entrada. Para ver todos los formatos de color 
veasé :mod:`pygame.Color`.


Centrar Texto en el Fondo
-------------------------

Ahora que tenemos el surface del fondo, vamos a representar el texto en él. 
Nosotros solo haremos esto si vemos que el módulo :mod:`pygame.font` se importó
correctamente. De no ser así, hay que saltear esta sección. ::

    if pg.font:
        font = pg.font.Font(None, 64)
        text = font.render("Pummel The Chimp, And Win $$$", True, (10, 10, 10))
        textpos = text.get_rect(centerx=background.get_width() / 2, y=10)
        background.blit(text, textpos)

Tal como pueden ver, hay un par de pasos para realizar esto. Primero, debemos
crear la fuente del objeto y renderizarlo (representarlo) en una nueva Surface.
Luego, buscamos el centro de esa nueva surface y lo pegamos (blit) al fondo.

La fuente es creada con el constructor `Font()` del módulo `font`. Generalmente,
uno va a poner el nombre de la fuente TrueType en esta función, pero también se 
puede poner `None`, como hicimos en este caso, y entonces se usará la fuente
por predeterminada. El constructor `Font` también necesita la información
del tamaño de la fuente que se quiere crear.

Luego vamos a represetar (renderizar) la fuente en la nueva surface. La función
`render` crea una nueva surface que es del tamaño apropiado para nuestro texto.
En este caso, también le estamos pidiendo al render que cree un texto suavizado
(para un lindo efecto de suavidad en la apariencia) y que use un color gris oscuro.

Lo siguiente que necesitamos es encontrar la posición la posición central, para 
colocar el texto en el centro de la pantalla. Creamos un objeto "Rect" de las 
dimensiones del texto, lo cual nos permite asignarlo fácilmente al centro de la
pantalla.

Finalmente, blitteamos (pegamos o copiamos) el texto en la imagen de fondo.


Mostrar el Fondo mientras Termina el Setup
------------------------------------------
Todavía tenemos una ventana negra en la pantalla. Mostremos el fondo
mientras esperamos que se carguen los otros recursos. ::

  screen.blit(background, (0, 0))
  pygame.display.flip()

Esto va a blittear (pegar o copiar) nuestro fondo en la ventana de visualización.
El blit se explica por sí mismo, pero ¿qué está haciendo esa rutina flip?

En pygame, los cambios en la surface de visualización (display) no se hacen visibles
inmediatamente. Normalmente, la pantalla debe ser actualizacada para que el usuario
pueda ver los cambios realizados. En este caso la función `flip()` es perfecta para eso
porque se encarga de toda el área de la pantalla. 


Preparar Objetos del Juego
--------------------------

En este caso crearemos todos los objetos que el juego va a necesitar.

::
    
    whiff_sound = load_sound("whiff.wav")
    punch_sound = load_sound("punch.wav")
    chimp = Chimp()
    fist = Fist()
    allsprites = pg.sprite.RenderPlain((chimp, fist))
    clock = pg.time.Clock()

Primero cargamos dos efectos de sonido usando la función `load_sound`, que se
encuentra definida en código arriba. Luego, creamos una instancia para cada 
uno de los sprites de la clase. Por último, creamos el sprite 
:class:`Group <pygame.sprite.Group>` que va a contener todos nuestros sprites.

En realidad, nosotros usamos un grupo especial de sprites llamado 
:class:`RenderPlain<pygame.sprite.RenderPlain>`. 
Este grupo de sprites puede dibujar en la pantalla todos los sprites que contiene.
Es llamado `RenderPlain` porque en realidad hay grupos Render más avanzados, pero
para nuestro juego nosotros solo necesitamos un dibujo simple. Nosotros creamos
el grupo llamado "allsprites" al pasar una lista con todos los sprites que deberían
pertenecer al grupo. Exise la posibilidad, si más adelante quisieramos, de agregar 
o sacar sprites de este grupo, pero para este juego no sería necesario.

El objeto `clock` que creamos será usado para ayudar a controlar la frequencia de
cuadros de nuestro juego. Vamos a usarlo en el bucle (loop) principal de nuestro
juego para asegurarnos que no se ejecute demasiado rápido.


Bucle principal (Main Loop)
---------------------------

No hay mucho por acá, solo un loop infinito. ::

    going = True
    while going:
        clock.tick(60)

Todos los juegos se ejecutan sobre una especie de loop. El orden usual de las cosas
es verificar el estado de la computadora y la entrada de usuario, mover y actualizar
el estado de todos los objetos, y luego dibujarlos en la pantalla. Verás que este 
ejemplo no es diferente.

También haremos un llamado a nuestro objeto `clock`, que asegurará que nuestro juego
no se ejecute pasando los 60 cuadros por segundo.

Manejar los Eventos de Entrada
------------------------------

Este es un caso extremandamente simple para trabajar la cola de eventos. ::

    for event in pg.event.get():
        if event.type == pg.QUIT:
            going = False
        elif event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
            going = False
        elif event.type == pg.MOUSEBUTTONDOWN:
            if fist.punch(chimp):
                punch_sound.play()  # punch
                chimp.punched()
            else:
                whiff_sound.play()  # miss
        elif event.type == pg.MOUSEBUTTONUP:
            fist.unpunch()

Primero obtenemos todos los Eventos (events) disponibles en pygame y los recorremos en loop. 
Las primeras dos pruebas es para ver si el usuario dejó nuestro juego, o si 
presionó la tecla de escape. En estos casos, configuramos ``going`` en ``False``,
permitiendonos salir del loop infinito.

A continuación, verificamos si se presionó o si se soltó el botón del mouse. En el 
caso de que el botón se haya presionado, preguntamos al primer objeto si chocó con
el mono. Se reproduce el sonido apropiado, y si el mono fue golpeado, le decimos 
que empiece a girar (al hacer un llamado a su método `punched()` )


Actualizar los Sprites
----------------------

::

  allsprites.update()

Los grupos de Sprite tienen un método `update()`, que simplemente llama al
método de actualización para todos los sprites que contiene. Cada uno de los
objetos se va a mover, dependiendo de cuál sea el estado en el que estén. Acá
es donde el mono se va a mover de un lado a otro, o va a girar un poco más 
lejos si fue recientemente golpeado.



Dibujar la Escena Completa
--------------------------

Ahora que todos los objetos están en el lugar indicado, es el momento para 
dibujarlos. ::

  screen.blit(background, (0, 0))
  allsprites.draw(screen)
  pygame.display.flip()

La primera llamada de blit dibujará el fondo en toda la pantalla. Esto borra 
todo lo que vimos en el cuadro anterior (ligeramente ineficiente, pero 
suficientemnte bueno para este juego). A continuación, llamamos al método 
`draw()` del contenedor de sprites. Ya que este contenedor de sprites es 
en realidad una instancia del grupo de sprites "DrawPlain", sabe como dibujar
nuestros sprites. Por último, usamos el método `flip()` para voltear los contenidos
del software de pygame. Se realiza el flip a través del cargado de la imagen en segundo
plano. Esto hace que todo lo que dibujamos luego se visibilice de una vez.


Fin del Juego
-------------

El usuario ha salido del juego, hora de limpiar (clean up) ::

    pg.quit()

Hacer la limpieza, el cleanup, de la ejecución del juego en *pygame* es extremandamente
simple. Ya que todas las variables son automáticamente destruidas, nosotros no 
tenemos que hacer realmnete nada, únicamente llamar a `pg.quit()` que explicitamente
hace la limpieza de las partes internas del pygame.