.. TUTORIAL:Introduction to the surfarray module

.. include:: ../../reST/common.txt

*************************************************
  Tutoriales de Pygame - Introducción a Surfarray
*************************************************

.. currentmodule:: surfarray

Introducción a Surfarray
========================

.. rst-class:: docinfo

:Autor: Pete Shinners
:Contacto: pete@shinners.org
:Traducción al español: Estefanía Pivaral Serrano


Introducción
------------

Este tutorial intentará presentar tanto Numpy como el módulo de surfarray
de pygame a los usuarios. Para principiantes, el código que utiliza surfarray 
puede ser bastante intimidante. Pero en realidad, hay sólo unos pocos conceptos 
que entender y estarás listo para empezar. Con el uso del módulo de surfarray 
es posible realizar operaciones a nivel de píxeles desde el código Python 
sencillo. El rendimiento puede llegar a ser bastante cercano al nivel de hacer 
el código en C.

Puede que solo desees ir directamente a la sección *"Examples"* para tener 
una idea de lo que es posible con este módulo, y luego comenzar desde el 
principio aquí para ir avanzando.

Ahora bien, no voy a engañarte para que pienses que todo va a ser muy sencillo. 
Lograr efectos avanzados modificando los valores de píxeles puede ser complicado.
Solo dominar Numeric Python, NumPy, (el paquete de matrices original de 
SciPy era Numeric, el predecesor de NumPy) requiere aprendizaje. En este tutorial 
me centraré en lo básico y utilizaré muchos ejemplos en un intento de sembrar las 
semillas de la sabiduría. Después de haber terminado el tutorial, deberías tener 
una comprensión básica de cómo funciona el surfarray.


Numeric Python
--------------

Si no tenés instalado el paquete NumPy de python, 
necesitarás hacerlo. Podés descargar el paquete dede la página de 
descargas de NumPy en
`NumPy Downloads Page <http://www.scipy.org/scipylib/download.html>`_
Para asegurarte que Numpy esté funcionando correctamente,
deberías obtener algo como esto desde prompt (inteprete) interactivo de Python.::


  >>> from numpy import *                    #importar numeric
  >>> a = array((1,2,3,4,5))                 #crear un array
  >>> a                                      #mostrar array
  array([1, 2, 3, 4, 5])
  >>> a[2]                                   #index al array
  3
  >>> a*2                                    #nuevo array con valores dobles
  array([ 2,  4,  6,  8, 10])

Como se puede ver, el módulo NumPy nos proporciona un nuevo tipo de data, el *array*.
Este objeto mantiene un array de tamaño fijo, y todos los valores que contiene en su
interior son del mismo tipo. Los arrays (matrices) también pueden ser multidimensionales,
que es como las usaremos con imágenes. 
Hay un poco más de información, pero es suficiente para empezar.

Si mirás al último comando de arriba, verás que las operaciones matemáticas en 
los array de NumPy se aplican para todos los valores del array. Esto se llama 
"element-wise operations" (operaciones elemento a elemento). Estos arrays 
también pueden dividirse en listas normales. La sintaxis de la división es la 
misma que se usa en objetos Python estándar.
*(así que estudia si es necesario)*.

Aquí hay algunos ejemplos más de arrays que funcionan. ::

  >>> len(a)                                 #obtener el tamaño del array
  5
  >>> a[2:]                                  #elementos a partir del 2
  array([3, 4, 5])
  >>> a[:-2]                                 #todos excepto los últimos 2
  array([1, 2, 3])
  >>> a[2:] + a[:-2]                         #agregar el primero y último
  array([4, 6, 8])
  >>> array((1,2,3)) + array((3,4))          #agregar arrays de tamaños incorrectos
  Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
  ValueError: operands could not be broadcast together with shapes (3,) (2,)

Obtenemos un error con el último comando, porque intentamos sumar dos arrays 
que tienen tamaños diferentes. Para que dos arrays operen entre sí, incluyendo 
operaciones comparaciones y asignaciones, deben tener las mismas dismensiones. 
Es muy importante saber que los nuevos arrays creados a partir de cortar el 
original hacen referencia a los mismos valores. Por lo tanto, cambiar los valores 
en una porción de la división también cambia los valores originales. Es importante 
cómo se hace esto. ::

  >>> a                                      #mostrar nuestro array inicial
  array([1, 2, 3, 4, 5])
  >>> aa = a[1:3]                            #dividir al medio 2 elementos
  >>> aa                                     #mostrar la división
  array([2, 3])
  >>> aa[1] = 13                             #cambiar el valor en la división
  >>> a                                      #mostrar cambio en el original
  array([ 1, 2, 13,  4,  5])
  >>> aaa = array(a)                         #copiar el array
  >>> aaa                                    #mostrar copia
  array([ 1, 2, 13,  4,  5])
  >>> aaa[1:4] = 0                           #configurar los valores medios a 0
  >>> aaa                                    #mostrar copia
  array([1, 0, 0, 0, 5])
  >>> a                                      #mostrar nuevamente el original
  array([ 1, 2, 13,  4,  5])

Ahora vamos a ver pequeños arrays con dos dimensiones.
No te preocupes demasiado, comenzar es lo mismo que tener una tupla de dos dimensiones 
*(una tupla dentro de otra tupla)*. 
Empecemos con los arrays de dos dimensiones. ::


  >>> row1 = (1,2,3)                         #crear una tupla de valores
  >>> row2 = (3,4,5)                         #otra tupla
  >>> (row1,row2)                            #mostrar como una tupla de dos dimensiones
  ((1, 2, 3), (3, 4, 5))
  >>> b = array((row1, row2))                #crear un array en 2D
  >>> b                                      #mostrar el array
  array([[1, 2, 3],
         [3, 4, 5]])
  >>> array(((1,2),(3,4),(5,6)))             #mostrar el nuevo array en 2D
  array([[1, 2],
         [3, 4],
         [5, 6]])

Ahora, con estos arrays bidimensionales *(de ahora en más "2D")* podemos 
indexar valores específicos y hacer cortes ambas dimensiones. Simplemente 
usando una coma para separar los índices, nos permite buscar/cortar 
en múltiple dimensiones. Simplemente usando "``:``" como un índex 
*(o no proporcionando suficiente índices)* nos devuelve todos los valores 
en esa dimensión. Veamos cómo funciona esto. ::

  >>> b                                      #mostrar nuestro array desde arriba
  array([[1, 2, 3],
         [3, 4, 5]])
  >>> b[0,1]                                 #indexar un único valor
  2
  >>> b[1,:]                                 #dividir la segunda fila
  array([3, 4, 5])
  >>> b[1]                                   #dividir la segunda fila (igual que arriba)
  array([3, 4, 5])
  >>> b[:,2]                                 #dividir la última columna
  array([3, 5])
  >>> b[:,:2]                                #dividir en un array de 2x2
  array([[1, 2],
         [3, 4]])

De acuerdo, mantente conmigo acá, esto es lo más díficil que puede ponerse. 
Al usar NumPy hay una característica más para la división. La división de arrays 
también permite especificar un *incremento de divsión*. La sintaxis para una 
división con incremento es ``start_index : end_index : increment``. ::

  >>> c = arange(10)                         #como el rango, pero crea un array
  >>> c                                      #muestra el array
  array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
  >>> c[1:6:2]                               #divide valores impares desde el 1 al 6
  array([1, 3, 5])
  >>> c[4::4]                                #divide cada 4to valor, empezando por el 4
  array([4, 8])
  >>> c[8:1:-1]                              #divide 1 al 8, de atrás para adelante /// invertido
  array([8, 7, 6, 5, 4, 3, 2])

Bien, eso es todo. Hay suficiente información acá para que puedas empezar 
a usar Numpy con el módulo surfarray. Ciertamente hay mucho más en 
NumPy, pero esto es solo una introducción. Además, ¿queremos pasar a cosas 
divertidas, no?


Importar Surfarray
------------------

Para usar el módulo surfarray necesitamos importarlo. Dado que ambos, tanto 
surfarray y NumPy, son componentes opcionales para pygame es bueno asegurarse 
de que se importen correctamente antes de usarlos. En estos ejemplos voy a 
importar NumPy en una variable llamada *N*. Esto permitirá saber qué funciones 
estoy usando son del paquete de NumPy. 
*(y es mucho más corto que escribir NumPy antes de cada función)* :: 

  probá:
      import numpy as N
      import pygame.surfarray as surfarray
  except ImportError:
      raise ImportError, "NumPy and Surfarray are required."


Introducción a Surfarray
------------------------

Hay dos tipos principales de funciones en surfarray. Un conjunto de funciones
para crear un array que es una copia de los datos de píxeles de la superficie
(surface). Las otras funciones crean una copia referenciada de los datos de 
píxeles del array, de modo que los cambios en el array afectan directamente a 
la surface original. Hay otras funciones que permiten acceder a cualquier valor 
alfa por pixel, como arrays junto con algunas otras funciones útiles. 
Veremos estas otras funciones más adelante.

Al trabajar con estos arrays de surface, existen dos formas de representar 
los valores de píxeles. En primar lugar, pueden representarse como enteros 
mapeados. Este tipo de array es un array simple en 2D con un solo entero que 
representa el valor de color mapeado de la superficie. Este tipo de array es 
últil para mover partes de una imagen al rededor de la pantalla. 
El otro tipo de array utiliza tres valores RGB para representar el color de 
cada píxel. Este tipo de array hace que sea extremadamente sencillo realizar 
efectos que cambian el color de cada píxel. Este tipo de array es también un 
poco más complicado de manejar, ya que es esencialmente un array numérico 3D. 
Aún así, una vez que ajustas tu mente en el modo adecuado, no es mucho más 
difícil que usar un array 2D normal.

El módulo NumPy utiliza los tipos de números naturales de la máquina para 
representar los valores de los datos, por lo que un array de NumPy puede consistir 
de enteros de 8-bits, 16-bits y 32-bits.
*(los array también pueden usar otro tipos como flotantes y dobles, pero para la 
manipulación de imágenes principalmente necesitamos preocuparnos por los tipos 
de enteros)*.
Debido a esta limitación de tamaños de los enteros, debes tener un poco más de cuidado 
para asegurarte de que el tipo de arrays que hacen referencia a los datos de píxeles se 
pueda mapear correctamente con un tipo adecuado de datos. Las funciones que crean estos 
arrays a partir de las superficies son: 

.. function:: pixels2d(surface)
   :noindex:

   Crea una matriz 2D *(valores de píxeles enteros)* que hace referencia a los datos 
   originales de la superficie. 
   Esto funcionará para todos los formatos de surface excepto el de 24-bit.

.. function:: array2d(surface)
   :noindex:

   Crea un array 2D *(valores de píxeles enteros)* que es copiada desde cualquier 
   tipo de superficie.

.. function:: pixels3d(surface)
   :noindex:

   Crea un array 3D *(valores de píxeles RGB)* que hacen referencia a los datos originales
   de la superficie.
   Esto solo funcionará en superficies de 24-bit y 32-bit que tengan el formato RGB o BGR.

.. function:: array3d(surface)
   :noindex:

   Crea un array 3D *(valores de píxeles RGB)* que se copia desde cualquier tipo 
   de surface.

Aquí hay una pequeña tabla que podría ilustrar mejor qué tipos de funciones 
se deben usar en cada surface. Como se puede observar, ambas funciones 
de array funcionarán con cualquier tipo de surface.

.. csv-table::
   :class: matrix
   :header: , "32-bit", "24-bit", "16-bit", "8-bit(c-map)"
   :widths: 15, 15, 15, 15, 15
   :stub-columns: 1

   "pixel2d", "yes",      , "yes", "yes"
   "array2d", "yes", "yes", "yes", "yes"
   "pixel3d", "yes", "yes",      ,
   "array3d", "yes", "yes", "yes", "yes"


Ejemplos
--------

Con esta información, estamos preparados para comenzar a probar cosas con los 
arrays de surface. A continuación encontrarán pequeñas demostraciones que 
crean un array de NumPy y los muestran en pygame. Estas diferentes pruebas 
se encuentran en el ejemplo arraydemo.py. Hay una función simple llamada 
*surfdemo_show* que muestra un array en la pantalla. 

.. container:: examples

   .. container:: example

      .. image:: ../../reST/tut/surfarray_allblack.png
         :alt: allblack

      ::

        allblack = N.zeros((128, 128))
        surfdemo_show(allblack, 'allblack')

      Nuestro primer ejemplo crea un array completamente negro. Siempre 
      que se necesite crear una nueva matriz numérica de un tamaño específico, 
      es mejor usar la función ``zeros``. Aquí creamos un array 2D de todos 
      ceros y lo mostramos.

      .. container:: break

         ..

   .. container:: example

      .. image:: ../../reST/tut/surfarray_striped.png
         :alt: striped

      ::

        striped = N.zeros((128, 128, 3))
        striped[:] = (255, 0, 0)
        striped[:,::3] = (0, 255, 255)
        surfdemo_show(striped, 'striped')

      Aquí estamos tratando con un array 3D. Empezamos creando una imagen 
      completamente roja. Luego cortamos cada tercera fila y le asignamos a un 
      color azul/verde. Como pueden ver, podemos tratar los arrays 3D casi 
      exactamente de la misma manera que los arrays 2D, solo asegúrense de 
      asignarles 3 valores en lugar de un único entero mapeado.

      .. container:: break

         ..

   .. container:: example

      .. image:: ../../reST/tut/surfarray_rgbarray.png
         :alt: rgbarray

      ::

        imgsurface = pygame.image.load('surfarray.png')
        rgbarray = surfarray.array3d(imgsurface)
        surfdemo_show(rgbarray, 'rgbarray')

      Aquí cargamos una imagen con el módulo de imagen, luego lo convertimos 
      en un array 3D de elementos de color RGB enteros. Una copia RGB 
      de una surface siempre tiene los colores dispuestos como a[r,c,0] para 
      el componente rojo, a[r,c,1] para el componente verde, y a[r,c,2] para 
      el azul. Esto se puede usar sin importar cómo se configuren los píxeles 
      del surface original, a diferencia de un array 2D que es una copia de 
      los píxeles de la surface :meth:`mapped <pygame.Surface.map_rgb>` (raw).
      Usaremos esta imagen en el resto de los ejemplos.

      .. container:: break

         ..

   .. container:: example

      .. image:: ../../reST/tut/surfarray_flipped.png
         :alt: flipped

      ::

        flipped = rgbarray[:,::-1]
        surfdemo_show(flipped, 'flipped')

      Aquí volteamos la imagen verticalmente. Todo lo que necesitamos para esto 
      es tomar el array de la imagen original y cortarlo usando un incremento 
      negativo.

      .. container:: break

         ..

   .. container:: example

      .. image:: ../../reST/tut/surfarray_scaledown.png
         :alt: scaledown

      ::

        scaledown = rgbarray[::2,::2]
        surfdemo_show(scaledown, 'scaledown')

      Basado en el último ejemplo, reducir una imagen escalar es bastante lógico. 
      Simplemente cortamos todos los píxeles usando un incremento de 2 vertical 
      y horizontalmente.

      .. container:: break

         ..


   .. container:: example

      .. image:: ../../reST/tut/surfarray_scaleup.png
         :alt: scaleup

      ::

        shape = rgbarray.shape
        scaleup = N.zeros((shape[0]*2, shape[1]*2, shape[2]))
        scaleup[::2,::2,:] = rgbarray
        scaleup[1::2,::2,:] = rgbarray
        scaleup[:,1::2] = scaleup[:,::2]
        surfdemo_show(scaleup, 'scaleup')

      Aumentar la escala de la imagen requiere un poco más de trabajo, pero es 
      similar al escalado previo hacia abajo, lo hacemos todo con cortes. Primero, 
      creamos un array que tiene el doble del tamaño de nuestro original. Primero 
      copiamos el array original en cada otro píxel del nuevo array. Luego lo 
      hacemos de nuevo para cada otro píxel, haciendo las columnas impares. En 
      este punto, tenemos la imagen escalada correctamente en sentido horizontal, 
      pero las otras filas son negras, por lo que simplemente debemos copiar cada 
      fila a la que está debajo. Entonces tenemos una imagen duplicada en tamaño.

      .. container:: break

         ..


   .. container:: example

      .. image:: ../../reST/tut/surfarray_redimg.png
         :alt: redimg

      ::

        redimg = N.array(rgbarray)
        redimg[:,:,1:] = 0
        surfdemo_show(redimg, 'redimg')

      Ahora estamos usando arrays 3D para cambiar los colores. 
      Acá establecemos todos los valores en verde y azul en cero.
      Esto nos deja solo con el canal rojo.

      .. container:: break

         ..


   .. container:: example

      .. image:: ../../reST/tut/surfarray_soften.png
         :alt: soften

      ::

        factor = N.array((8,), N.int32)
        soften = N.array(rgbarray, N.int32)
        soften[1:,:]  += rgbarray[:-1,:] * factor
        soften[:-1,:] += rgbarray[1:,:] * factor
        soften[:,1:]  += rgbarray[:,:-1] * factor
        soften[:,:-1] += rgbarray[:,1:] * factor
        soften //= 33
        surfdemo_show(soften, 'soften')

      Aquí realizamos un filtro de convulción 3x3 que suavizará nuestra  
      imagen. Parece que hay muchos pasos aquí, pero lo que estamos haciendo 
      es desplazar la imagen 1 píxel en cada dirección y sumarlos todos juntos 
      (con algunas multiplicaciones por ponderación). Luego se promedian todos  
      los valores. No es Gaussiano, pero es rápido. Un punto con los arrays 
      NumPy, la precisión de las operaciones aritméticas está determinada por 
      el array con el tipo de datos más grande. Entonces, si el factor no se 
      declarara como un array de 1 elemento de tipo numpy.int32, las 
      multiplicaciones se realizarían utilizando numpy.int8, el entero de 
      8 bits de cada elemento rgbarray. Esto causará una truncación de 
      valores. El array de suavizado también debe declararse con un tamaño 
      de entero más grande que rgbarray para evitar la truncación. 

      .. container:: break

         ..


   .. container:: example

      .. image:: ../../reST/tut/surfarray_xfade.png
         :alt: xfade

      ::

        src = N.array(rgbarray)
        dest = N.zeros(rgbarray.shape)
        dest[:] = 20, 50, 100
        diff = (dest - src) * 0.50
        xfade = src + diff.astype(N.uint)
        surfdemo_show(xfade, 'xfade')

      Por último, estamos realizando una transición gradual entre la imagen original y 
      una imagen de color azul sólido. No es emocionante, pero la imagen de destino 
      podría ser cualquier cosa, y cambiar el multiplicador 0.50 permitirá elegir 
      cualquier paso en una transición lineal entre dos imágenes.

      .. container:: break

         ..

Con suerte, a estas alturas estás empezando a ver cómo surfarray puede ser 
utilizado para realizar efectos especiales y transformaciones que sólo son 
posibles a nivel de píxeles. Como mínimo, se puede utilizar surfarray para 
realizar muchas operaciones del tipo Surface.set_at() y Surface.set_at() 
rápidamente. Pero no creas que esto ha terminado, todavía queda mucho 
por aprender.


Bloqueo de Superficie (Surface)
-------------------------------

Al igual que el resto de pygame, surfarray bloqueará cualquier Surface que 
necesite para acceder a los datos de píxeles. Sin embargo, hay un elemento 
más a tener en cuenta; al crear los array de *pixeles*, la surface 
original quedará bloqueada durante la vida útil de ese array de píxeles. 
Es importante recordarlo. Asegurate de *"eliminar"* el array de píxeles o 
de dejarlo fuera del alcance *(es decir, cuando la funcion vuelve, etc.)*.

También hay que tener en cuenta que realmente no querés hacer muchos 
*(si es que alguno)* accesos directos a píxeles en la surface del hardware 
*(HWSURFACE)*. Esto se debe a que los datos de la surface se encuentra en  
la tarjeta gráfica, y transferir cambios de píxeles a través del bus 
PCI/AGP no es rápido.


Transparencia
-------------

El módulo surfarray tiene varios métodos para acceder a los valores alpha/colorclave
de una Surface. Ninguna de las funciones alpha se ve afectada por la transparencia 
general de una Surface, solo por los vaores de los píxeles. Aquí está la lista de 
esas funciones.

.. function:: pixels_alpha(surface)
   :noindex:

   Crea un array 2D *(valores enteros de píxeles)* que hace referencia a 
   los datos alpha de la surface original.
   Esto solo funcionará en imágenes de 32-bit con un componente alfa de 
   8-bit.

.. function:: array_alpha(surface)
   :noindex:

   Crea un array 2D *(valores enteros de píxeles)* que se copia desde 
   cualquier tipo de surface.
   Si la surface no tiene valores alfa, 
   el array tendrá valores completamten opacos *(255)*.

.. function:: array_colorkey(surface)
   :noindex:

   Crea un array 2D *(valores enteros de píxeles)* que está establecida 
   como transparente *(0)* donde el color de ese píxel coincide con el 
   color clave de la Surface.


Otras Funciones de Surfarray
----------------------------

Solo hay algunas otras funciones disponibles en surfarray. Podés obtener una lista
mejor con mayor documentación en :mod:`surfarray reference page <pygame.surfarray>`.
Sin embargo, hay una función muy útil.

.. function:: surfarray.blit_array(surface, array)
   :noindex:
   
   Esto transferirá cualquier tipo de array de surface 2D o 3D a una 
   Surface con las mismas dimensiones.
   Este blit de surfarray generalmente será más rápido que asignar un 
   array a la de pixeles referenciado.
   Sin embargo, no debería ser tan rápido como el blitting normal de 
   surface, ya que esos están muy optimizados.


NumPy más Avanzado
------------------
Hay un par más de cosas que deberías saber sobre los arrays Numpy. Cuando se 
trata de arrays muy grandes, como los que son de 640x480, hay algunas cosas 
adicionales sobre las que debes tener cuidado. Principalmente, mientras que 
usar los operadores como + y * en los arrays los hace fáciles de usar, también 
es muy costoso en arrays grandes. Estos operadores deben hacer nuevas copias 
temporales del array, que luego generalmente se copian en otro array. Esto 
puede requerir mucho tiempo. Afortunadamente, todos los operadores de Numpy 
vienen con funciones especiales que pueden realizar la operación *"in place*" 
(en su lugar). Por ejemplo, en lugar de usar ``screen[:] = screen + brightmap`` 
podrías querer usar ``add(screen, brightmap, screen)`` que es más rápido.
De todos modos, debes leer la documentación UFunc de Numpy para obtener más 
información sobre esto. Es importante cuando se trata de los arrays.

Otra cosa a tener en cuenta al trabajar con arrays NumPy es el tipo de datos 
del array. Algunos de los arrays (especialmente el tipo de píxeles mapeado) 
a menudo devuelven arrays con un valor sin signo en 8-bits. Estos arrays se 
desbordarán fácilmente si no tienes cuidado. NumPy usará la misma coerción 
que se encuentra en los programas en C, por lo que mezclar una operación con 
números de 8 bits y 32 bits dará como resultado números de 32 bits. Puedes 
convertir el tipo de datos del array, pero definitivamente debes ser consciente 
de qué tipos de arrays tienes, si NumPy se encuentra en una situación en la que 
se arruinaría la precisión, lanzará una excepción.

Por último, debes tenér en cuenta que al asignar valores en los arrays 3D, 
estos deben estar entre 0 y 255, de lo contrario se producirá alguna 
truncación indefinida.


Graduación
----------

Bueno, ahí está. Mi breve introducción a Numeric Python y surfarray.
Espero que ahora veas lo que es posible, y aunque nunca los uses por 
ti mismo, no tengas miedo cuando veas código que los use. Echale un 
vistazo al ejemplo vgrade para ver más sobre los arrays numéricos. También, 
hay algunos demos *"flame"* que usan surfarray para crear un efectp de 
fuego en tiempo real.

Lo mejor que podés hacer es probar alguna cosas por tu cuenta. Ve despacio 
al principio y ve construyendo poco a poco, ya he visto algunas cosas geniales 
con surfarray, como gradientes radiales y más.
Buena suerte.
