.. TUTORIAL: David Clark's Newbie Guide To Pygame

.. include:: ../../reST/common.txt

***********************************
  Guía de Pygame para Principiantes
***********************************

.. title:: Guía de Pygame para Newbies

.. rst-class:: docinfo

:Traducción al español: Estefanía Pivaral Serrano


Guía de Pygame para Principiantes
=================================

o **Cosas que aprendí mediante prueba y error para que vos no tengas que pasar por eso**

o **Cómo aprendí a dejar de preocuparme y amar el blit.**

Pygame_ es un contenedor de Python para SDL_, escrito por Pete Shinners. Lo cual 
significa que al usar pygame, podés escribir juegos u otras aplicaciones 
multimedia en Python que se ejecutarán sin alteraciones en cualquier 
plataforma compatible con SDL (Windows, Unix, Mac, BeOS y otras).

Pygame puede ser fácil de aprender, pero el mundo de la programación de gráficos 
pueden ser bastante confusos para el recién llegado. Escribí esto para tratar de 
destilar el conocimiento práctico que obtuve durante el último año trabajando
con pygame y su predecesor, PySDL. He tratado de clasificar las sugerencias en 
orden de importancia, pero cuán relevante es cada consejo dependerá de tu propio
antecedente y los detalles de tu proyecto.


Ponte cómodo trabajando con Python
----------------------------------

Lo más importante es sentirse confiado usando python. Aprender algo 
tan potencialmente complicado como programación de gráficos será un verdadero
fastidio si tampoco se está familiarizado con el lenguaje que se está usando.
Escrí una cantidad de programas considerables en Python -- analizá (parse)
algunos archivos de texto, escribí un juego de adivinanzas o un programa 
de entradas de diario, o algo. Ponte cómodo con las secuencias de caracteres
que representan textos (strings) y la manipulación de listas.
Sepa cómo funciona ``import`` (importar) -- intenta escribir un programa que
se extienda a varios archivos fuente. Escribe tus propias funciones, y practica
manipular números y caracteres; sepa cómo convertir de una a otra.
Llega al punto en que la sintaxis para usar listas y diccionarios es algo instintivo--
no querés tener que ejecutar la documentación cada vez que necesitas dividir una lista 
u ordernar un juego de llaves. Resiste la tentación de correr a una lista de emails,
comp.lang.python, o IRC cuando te encuentres en un problema. En lugar de eso, enciende
el interperte y juega con el problema por unas horas. Imprime el `Python
2.0 Quick Reference`_ y conservalo junto a la computadora.

Esto puede sonar increiblemente aburrido, pero la confianza que vas a ganar al 
familiarizarte con python hará maravillas cuando se trate de escribir tu propio 
juego. El tiempo que dediques a escribir código python de forma instintiva no 
será nada en comparación con el tiempo que ahorrarás al escribir código real.


Reconoce qué partes de pygame necesitás realmente.
--------------------------------------------------

Ver el revoltijo de clases en la parte superior de la documentación del índice
de documentación de pygame puede ser confuso. Lo más importante es darse cuenta 
de que se puede hacer mucho con tan solo un pequeño subconjunto de funciones. 
Existen muchas clases que probablemente nunca uses -- en un año, yo no he tocado 
las funciones  ``Channel``, ``Joystick``, ``cursors``, ``Userrect``, ``surfarray`` 
o ``version``.



Sepa qué es una Surface (superficie)
------------------------------------

La parte más importante de pygame es la Surface (superficie). La Surface puede 
pensarse como una hoja de papel en blanco. Se pueden hacer muchas cosas con la 
Surface -- se pueden dibujar líneas, colorear partes de ella con colores, copiar 
imágenes hacia y desde ella, y establecer o leer píxeles indivduales de colores 
en ella. Una Surface puede ser de cualquier tamaño (dentro de lo lógico) y puede 
haber tantas como quieras (de nuevo, dentro de lo razonable). 
Una Surface es especial -- la que vayas a crear con ``pygame.display.set_mode()``.
Esta 'display surface' (surface de visualización) representa la pantalla; 
lo que sea que hagas en ella aparecerá en la pantalla del usuario. Solo puedes 
tener una de esas -- esa es una limitación de SDL, no de pygame.

Entonces, ¿cómo crear Surfaces? Como mencioné arriba, la Surface especial se 
crea con ``pygame.display.set_mode()``.  Se puede crear una surface que 
contenga una imagen usando ``image.load()``, o podés crear una surface que 
contenga texto con ``font.render()``. Incluso se puede crear una surface que 
no contenga nada en absoluto con ``Surface()``.

La mayoría de las funciones de Surface no son críticas. Sólo es necesario 
aprender ``blit()``, ``fill()``, ``set_at()`` y ``get_at()``,  y vas a estar
bien. 

Usa surface.convert().
----------------------

Cuando yo leí por primera vez la documentación para ``surface.convert()``, no pensé
que fuera algo de lo que tuviera que preocuparme. 'Sólo voy a usar PNGs, por lo 
tanto todo o que haga será en ese formato. Entonces no necesito ``convert()``';. 
Resultó ser que estaba muy, muy equivocado.

El 'format' (formato) al que ``convert()`` se refiere no es el formato del archivo 
(por ejemplo, PNG, JPEG, GIF), es lo que se llama el 'píxel format' (formato pixel). 
Esto se refiere a la forma particular en la que una Surface registra colores 
individuales en un píxel especifico.  Si el formato de la Surface (Surface format) 
no es el mismo que el formato de visualización (display format), SDL tendrá que 
convertirlo sobre la marcha para cada blit -- un proceso que consume bastante 
tiempo. No te preocupes demasiado por la explicación; solo ten en cuenta que 
``convert()`` es necesario si querés que haya velocidad en tus blits.

¿Cómo se usa convert? Sólo hay que hacer una call (llamada) creando la Surface 
con la función ``image.load()``. En vez de hacer únicamente::

    surface = pygame.image.load('foo.png')

Haz::

    surface = pygame.image.load('foo.png').convert()

Es así de fácil. Lo único que se necesita es hacer una de esas calls (llamadas) 
por Surface, cuando cargues una imagen del disco. 
It's that easy. You just need to call it once per surface, when you load an
image off the disk.  Estará satisfecho con los resultados; veo al rededor de 
un 6x aumento de la velocidad de blitting llamando (haciendo la call) ``convert()``.

La única vez que no vas a querer usar ``convert()`` es cuando realmente necesitas 
tener el control absoluto sobre al formato interno de una imagen -- digamos que 
estás escribiendo un programa de conversión de imagen o algo así, y 
necesitás asegurarte que el archivo de salida tenga el mismo formato píxeles 
que el archivo de entrada. Si estás escribiendo un juego, necesitás velocidad. 
Usa ``convert()``.


Animación rect "sucia".
-----------------------

La causa más común de frecuencias de cuadros inadecuadas en los programas Pygame 
resulta de malinterpretar la función ``pygame.display.update()``. Con pygame, con 
simplemente dibujar algo en la Surface de visualización no hace que aparezca en la 
pantalla -- necesitas hacer un llamado a ``pygame.display.update()``.  Hay tres 
formas de llamar a esta función:


 * ``pygame.display.update()`` -- Esto actualiza toda la ventana (o toda la pantalla para visualizaciones en pantalla completa).
 * ``pygame.display.flip()`` -- Esto hace lo mismo, y también hará lo correcto si estás usando ``double-buffered`` aceleración de hardware, que no es así, entonces sigamos ...
 * ``pygame.display.update(a rectangle or some list of rectangles)`` -- Esto actualiza solo las áreas rectangulares de la pantalla que especifiques.
  

La mayoría de la gente nueva en programación gráfica usa la primera opción -- 
ellos actualizan la pantalla completa en cada cuadro. El problema es que esto 
es inaceptablemente lento para la mayoría de la gente. Hacer una call a  
``update()`` toma 35 milisegundos en mi máquina, lo cual no parece mucho, hasta 
que te das cuenta que 1000 / 35 = 28 cuadros por segundo *máximo*. Y eso es 
sin la lóagica del juego, sin blits, sin entrada (input) , sin IA, nada. Estoy 
aquí sentado actualizando la pantalla, y 28 fps (frames per second - cuadros
por segundo) es mi máximo de cuadros por segundo. Ugh.

La solucion es llamada 'dirty rect animation' o 'animación de rect sucia'.
En vez de actualizar la pantalla completa en cada cuadro, solo se actualizan 
las partes que cambiaron desde el último cuadros. Yo hago esto al hacer un 
seeguimiento de esos rectángulos en una lista, luego llamando a ``update(the_dirty_rectangles)`` 
al final del cuadro. En detalle para un sprite en movimiento, yo:

 * Blit una parte del fondo sobre la ubicación actual del sprite, borrándolo.
 * Añado el rectángulo de la ubicación actual a la lista llamada dirty_rects.
 * Muevo el sprite.
 * Dibujo (Draw) el sprite en su nueva ubicación.
 * Agrego la nueva ubicación del sprite a mi lista de dirty_rects.
 * Llamo a ``display.update(dirty_rects)``

La diferenci aen velocidad es asombrosa. Tengan en consideración que SolarWolf_ 
tiene docenas de sprites en constante movimiento que se actualizan sin problemas, 
y aún así le queda suficiente tiempo para mostrar un campo estelar de paralaje 
en el fondo, y también actualizarlo.

Hay dos casos en que esta técnica no funciona. El primero es cuando toda la ventana 
o la pantalla es siendo actualizada realmente en cada cuadro -- pensá en un motor 
de desplazamiento como un juego de estrategia en tiempo real o un desplazamiento 
lateral. Entonces, ¿qué hacés en ese caso? Bueno, la respuesta corta es -- no 
escribas este tipo de juegos en pygame. La respuesta larga es desplazarse en pasos 
de varios píxeles a la vez; no intentes hacer del desplazamiento algo 
perfectamente suave. El jugador apreciará un juego que se desplaza rápidamente y 
no notará demasiado el fondo saltando.

Una nota final -- no todo juego requeire altas frecuencias de cuadros. Un 
juego de guerra estratégico podría funcionar fácilmente con solo unas pocas 
actualizaciones por segundo -- en este caso, la complejidad agregada de la 
animación de rect sucio (dirty rect animation) puede no ser necesaria. 



NO hay regla seis.
------------------

Los surfaces de hardware son más problemáticos de lo que valen.
---------------------------------------------------------------

**Especialmente en pygame 2, porque HWSURFACE ahora no hace nada**

Si estuviste mirando las distintas flags (banderas) que se pueden 
usar con ``pygame.display.set_mode()``, puede que hayas pensado 
lo siguiente: `Hey, HWSURFACE! Bueno, quiero eso -- a quién no le 
gusta la acelación de hardware. Ooo... DOUBLEBUF; bueno, eso suena 
rápido, ¡supongo que yo también quiero eso!`.  No es tu culpa; 
hemos sido entrenados por años en juegos 3D como para creer que 
la aceleración de hardware es buena, y el rendering (representación)
del software es lento.

Desafortunadamente, el rendering de hardware viene con una larga lista 
de inconvenientes:

 * Solo funciona en algunas plataformas. Las máquinas con Windows generalmente pueden obtener surfaces (superficies) si se les solicita. La mayoría de otras plataformas no pueden. Linux, por ejemplo, puede proporcionar una surface de hardware si X4 está isntalado, si DGA2 está funcionando correctamente, y si las lunas están alineadas correctamente. Si la surface de hardware no está disponible, SDL va a proporcionar silenciosamente una surface de software en su lugar.

 * Solo funciona en pantalla completa.

 * Complica el acceso por píxel. Si tenés una surface de hardware, necesitas bloquear la superficie antes de escribir o leer valores de píxel en ella. Si no lo haces, Cosas Malas Suceden. Luego vas a necesitar desbloquear rápidamente la superficie nuevamente antes de que el SO se confunda y comience a entrar en pánico. La mayor parte de los procesos en pygame están automatizados, pero es algo más a tener en cuenta.

 * Pierdes el puntero del mouse. Si especificás ``HWSURFACE`` (y de hecho lo obtienes) tu puntero, por lo general, simplemente desaparecerá (o peor, se quedará en un estado parpadeante por ahí). Deberás crear un sprite para que actúe como puntero manual, y deberás preocuparte por la aceleración y la sensibilidad del puntero. ¡Qué molestia!

 * Podría ser más lento de todos modos. Muchos controladores no están acelerados para los tipos de dibujos que hacemos, y dado que todo tiene que ser blitteado por el bus de video (a menos que también puedas meter la la surface de origen en la memoria de video), puede que termine siendo más lento que el acceso al software de todos modos.

El rendering (representación) de hardware tiene su lugar. Funciona de manera 
bastante confiable en Windows, por lo que si no estás interesado en el 
rendimiento de multiplataformas, puede proporcionarte un aumento sustancial 
de la velocidad. Sin embargo, tiene un costo -- mayor dolor de cabeza y 
complejidad. Es mejor apegarse al viejo y confiable ``SWSURFACE`` hasta 
que estés seguro de lo que estás haciendo. 

No te distraigas con problemas secundarios.
-------------------------------------------

A veces, los nuevos programadores dedican mucho tiempo preocupandose sobre 
problemas que no son realmente críticos para el éxito de su juego. El deseo 
de arreglar los problemas secundarios es entendible, pero al principio en el 
proceso de creación de un juego, ni siquiera puedes saber cuáles son las 
preguntas importantes, mucho menos qué respuestas deberías elegir. El 
resultado puede ser un montón de prevariaciones innecesarias.

Por ejemplo, consideren la pregunta de cómo organizar los archivos gráficos.
¿Debería cada cuadro tener su propio archivo gráfico, o cada sprite? ¿Quizás 
todos los gráficos se deberían comprimir en un archivo? Se ha perdido una 
gran cantidad de tiempo en muchos proyectos, preguntándose estas preguntas 
en lista de correo, debatiendo las respuestas, haciendo perfiles, etc, etc.
Este es un tema secundario; cualquier cantidad de tiempo invertido en 
discutir eso, debería haber sido usado en escribir el código del juego real.

El idea es que es mucho mejor tener una solución 'bastante buena' que haya 
sido implementada, que una solucion perfecta que nunca se haya llegado a 
escribir.


Los rects son tus amigos.
-------------------------

El envoltorio de Pete Shinners puede tener efectos alfa geniales y 
velocidades rápidas de blitting, pero tengo que admitir que mi parte 
favorita de pygame es la humilde clase ``Rect``. Un rect es simplemente 
un rectángulo -- definido solo por la posición de su esquina superior 
izquierda, su ancho y su altura. Muchas funciones de pygame toman rects 
como argumentos, y ellas solo hacen 'rectstyles', una secuencia que tiene 
los mismos valores que un rect. Entonces si necesito un rectángulo que 
defina el área entre 10, 20 y 40, 50, puedo hacer cualquier de las 
siguientes::

    rect = pygame.Rect(10, 20, 30, 30)
    rect = pygame.Rect((10, 20, 30, 30))
    rect = pygame.Rect((10, 20), (30, 30))
    rect = (10, 20, 30, 30)
    rect = ((10, 20, 30, 30))

Sin embargo, si usas cualquiera de las primeras tres versiones, obtendrás 
accesso a las funciones de utilidad del rect. Estas incluyen funciones para 
mover, encoger e inflar los rects, encontrar la union de dos rects, y una 
variedad de funciones de detección de colisión. 

Por ejemplo, supongamos que yo quiero obtener una lista de todos los sprites 
que contiene un punto (x,y) -- quizás el jugador clickeó ahí, o quizás esa es 
la ubicación actual de una bala. Es simple si cada sprite tiene un miembro 
.rect -- solo hay que hacer:

    sprites_clicked = [sprite for sprite in all_my_sprites_list if sprite.rect.collidepoint(x, y)]

Los rects no tienen otra relación con los surfaces o con las funciones gráficas, 
aparte del hecho de que puedes usarlos como argumentos. También se pueden usar en 
lugares que no tienen nada que ver con gráficos, pero aún así deben ser definidos 
como rectángulos. En cada proyecto descrubro algunos lugares nuevos donde usar 
rects donde nunca pensé que los necesitaría.


No te molestes con la detección de colisión de píxel perfecto.
--------------------------------------------------------------

Así que, tenés tus sprites moviendose y necesitás saber si se están chocando entre sí. Es tentador escribir algo como lo siguiente:ite something like the following:

 * Checkear si los rects están en colisión. Si no lo están, ignorarlos.
 * Para cada píxel en el área de superposición, ver si los píxeles correspondientes de ambos sprites son opacos. Si es así, hay una colisión.

Hay otras formas de hacer esto, con ???????? coordinando máscaras de sprite y 
así sucesivamente, pero de cualquier forma en que se haga en pygame, 
probablemente sea demasiado lento. Para la mayoría de los juego probablemente 
sea mejor hacer solo un "sub-rect de colisión" -- esto es, crear un rect por 
cada sprite que es un poco más pequeño que la imagen real, y usar eso para 
colisiones. Esto va a resultar más rápido y, en la mayoría de los casos, el 
jugador no va a notar la imprecisión.
There are other ways to do this, with ANDing sprite masks and so on, but any
way you do it in pygame, it's probably going to be too slow. For most games,
it's probably better just to do 'sub-rect collision' -- create a rect for each
sprite that's a little smaller than the actual image, and use that for
collisions instead. It will be much faster, and in most cases the player won't
notice the imprecision.


Gestión del subsistema de eventos.
----------------------------------

El sistema de eventos de Pygame es un poco truculento. Hay en realidad dos formas 
diferntes de saber qué está haciendo un dispositivo de entrada (teclado, mouse, 
o joystick).

La primera es directamente comprobar el estado del dispositivo. Esto se hace 
mediante la llamada, digamos, ``pygame.mouse.get_pos()`` o
``pygame.key.get_pressed()``.
Esto te indicará el estado de tu dispositivo *en el momento en que llames a 
la función*

El segundo método usa la cola de eventos de SDL. Esta cola es una lista de 
eventos -- eventos se agregan a la lista al ser detectados, y se eliminan 
de la cola mientras se leen.

Hay ventajas y desventajas para cada sistema. Comprobación de estado (sistema 1) 
(state-checking) aporta precisión -- sabés exactamente cuándo se realizó la 
entrada -- si ``mouse.get_pressed([0])`` (mouse fue presionado) es 1, eso significa 
que el botón izquierdo del mpuse está abajo *justo en este momento*. La cola de 
eventos meramente reporta que el mouse estuvo abajo en algún momento del pasado; 
si revisas la cola con bastante frecuencia, eso puede estar bien, pero si te 
demorás en verificarlo con otro código, latencia de entrada puede incrementar. 
Otra ventaja del sistema de comprobación de estado es que detecta "acordes" 
fácilmente; es decir, varios estados al mismo tiempo. Si querés saber si las 
teclas ``t`` y la ``f`` están ambas presionadas al mismo tiempo, sólo hay que
checkear::

    if (key.get_pressed[K_t] and key.get_pressed[K_f]):
        print("Sip!")

Sin embargo, en el sistema de colas, cada pulsación de tecla llega a la cola 
como un evento completamente separado, entonces será necesario recordar que 
la tecla ``t`` estuvo presionada y que aún no había sido soltada mientras la 
tecla ``f`` fue presionada. Un poco más complicado.

Sin embargo, el sistema de estados tiene una gran desventaja. Solo informa 
el estado del dispositivo al momento en que es llamado; si el usuario clickea 
el botón del mouse y lo suelta justo antes del llamado a ``mouse.get_pressed()``, 
el botón del mouse va a devolver un 0 -- ``get_pressed()`` falló completamente 
en detectar la pulsación del botón del mouse. Dos events, ``MOUSEBUTTONDOWN`` 
y ``MOUSEBUTTONUP``, seguirán esperando en la cola de eventos a ser 
recuperados y procesados.

La lección es la siguiente: elegí el sistema que cumpla con tus requisitos. Si 
no hay mucho sucediendo en tu loop -- supongamos, estás sentado en un bucle de 
``while True``, esperando una entrada, usa ``get_pressed()`` u otra función de 
estado; la latencia será menor. Por otro lado, si cada pulsación de tecla es 
crucial, pero la latencia no es tan importante -- por ejemplo, el usuario está 
escribiendo algo en un cuadro de edición, usá la cola de eventos. Algunas 
pulsaciones de tecla pueden retrasarse un poco, pero al menos van a aparecer 
todas.

Una nota sobre ``event.poll()`` vs. ``wait()`` -- ``poll()`` puede parecer mejor
ya que no impide al programa de hacer otra cosa mientras está esperando la 
entrada --  ``wait()`` suspende el programa hasta que reciba el evento. 
Sin embargo, ``poll()`` consumirá el 100% del tiempo disponible del CPU mientras 
se esté ejecutando y llenará la cola de eventos con ``NOEVENTS``.  Para 
seleccionar solo los tipo de eventos que resultan de interés usa ``set_blocked()``,
la cola será mucho más manejable.


Colorkey vs. Alpha.
-------------------

Hay mucha confusión en torno a estas dos técnicas, y gran parte de esto proviene 
de la terminología usada. 

'Colorkey blitting' (blitting de la clave de color) implica decirle a pygame que 
todos lso píxeles de cierto color de una determinada imagen son transparentes en 
vez del color que realmente sean. Estos píxeles transparentes no son blitteados 
cuando el resto de la imagen es blitteada y entonces no oscurecen el fondo. Así 
es como hacemos los sprites que no son de forma rectangular. Simplemente llamamos 
a ``surface.set_colorkey(color)``, donde el color es una tupla RGB, supongamos
(0,0,0). Esto haría que cada píxel en la imagen de origen transparente en vez de 
negro.

'Alpha' es diferente, y como en dos sabores. 'Image alpha' (imagen alfa) que 
aplica a la imagen completa, y es probablemente lo que quieras. Propiamente 
conocido como 'translucidez', alpha causa que cada píxel en la imagen de origen 
sea solo *parcialmente* opaco. Por ejemplo, si configuras el alfa de una surface 
en 192 y después lo blitteas (convertis) en un fondo, 3/4 del color de cada 
píxel provendrá de la imagen de origan, y 1/4 del fondo. Alfa se mide de 255 a 0, 
donde 0 es completamente transparente, y 255 es completamente opaco. Nótese que 
el blitting con colorkey y alfa (colorkey and alfa blitting) pueden combinarse -- 
esto produce una imagen completamente transparete en algunos lugares y 
semi-transparente en otros.

'Per-pixel alpha' ('Alfa por pixel') es el otro tipo de alfa, y es más complicado
Básicamente, cada píxel de la imagen de origen tiene su propio valor alfa, de 0 
a 1. Cada píxel, por lo tanto, puede tener una opacidad diferente cuando se 
blittea (proyecta) sobre el fondo. Este tipo de alfa no se puede mezclar con 
la proyección (el blitting) de la clave de color, y anula el 'per-image' alfa.
El alfa por píxel (per-pixel alfa) es raramente usado en juego, y para usarlo 
tenes que guardar la imagen de origen en un editor gráfico con un *canal alpha* 
especial. Es complicado -- no lo usen todavía.

Haz cosas a la manera de pythony.
---------------------------------

Una nota final (no es la menos importante, simplemente viene al final) Pygame 
es un envoltorio bastante liviano alrededor de SDL, que a su vez es un ligero 
envoltorio alrededor de las calls (llamadas) de gráficos del sistema operativo 
nativo. Las posibilidades son muy buenas de que si tu código sigue lento, 
habiendo seguido las indicaciones que mencioné arriba, entonces el problema 
yace en la forma en que estás direccionando tus datos en python.
Algunos modismos simplemente van a ser lentos en python sin importar lo que hagas.
Afortunadamente, python es un lenguaje muy claro -- si un fragmento del código se 
ve extraño o difícil de manejar, es probable que su velocidad también se pueda 
mejorar. Lée `Python Performance Tips`_ para obtener excelentes consejos sobre 
cómo puede mejorar la velocidad del código. Dicho esto, la optimización prematura
es la razí de todos los males; si simplemente no es lo suficientemente rápido 
no tortures el código intentando hacerlo más rápido. Algunas cosas simplemente 
no están destinadas a ser. :)


¡Ya está! Ahora sabés prácticamente todo lo que yo sé sobre el uso de pygame.
Ahora, ¡ve a escribir ese juego!

----

*David Clark es un ávido usuario de pygame y es editor de Pygame Code
Repository, una vidriera del códigos de juegos en python suministrado por la 
comunidad. Él es también el autor de Twitch, un juego de arcade completamente 
promedio de pygame.*

.. _Pygame: https://www.pygame.org/
.. _SDL: http://libsdl.org
.. _Python 2.0 Quick Reference: http://www.brunningonline.net/simon/python/quick-ref2_0.html
.. _SolarWolf: https://www.pygame.org/shredwheat/solarwolf/index.shtml
.. _Python Performance Tips: http://www-rohan.sdsu.edu/~gawron/compling/course_core/python_intro/intro_lecture_files/fastpython.html
