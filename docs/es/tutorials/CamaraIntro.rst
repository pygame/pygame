.. TUTORIAL: Introducción al Módulo de Cámara

.. include:: ../../reST/common.txt

******************************************************
  Tutoriales Pygame - Introducción al Módulo de Cámara
******************************************************


Introducción al Módulo de Cámara 
================================

.. rst-class:: docinfo

:Autor: Nirav Patel
:Contacto: nrp@eclecti.cc
:Traducción al español: Estefanía Pivaral Serrano

Pygame 1.9 viene con soporte para cámaras interconectadas, lo que nos permite 
capturar imagenes quietas, ver transmiciones en vivo y hacer visión computarizada.
Este tutorial cubrirá todos estos casos de uso, proporcionando ejemplos de código 
que pueden usar para basar sus propias apps o juegos. Pueden consultar la 
documentación de referencia por una API completa: 
:mod:`reference documentation <pygame.camera>`

.. note::

  A partir de Pygame 1.9 el módulo de cámara ofrece soporte nativo para 
  cámaras que usan v4l2 en Linux. Existe soporte para otras plataformas via 
  Videocapture o OpenCV, pero esta guía se enfocará en en módulo nativo.
  La mayor parte del código será válido para otras plataformas, pero 
  ciertas cosas como los controles no funcionarán. El módulo está también 
  marcado como **EXPERIMENTAL**, lo que significa que la API podría cambiar  
  las versiones posteriores.


Importación e Inicialización
----------------------------

::

  import pygame
  import pygame.camera
  from pygame.locals import *

  pygame.init()
  pygame.camera.init()

Dado que el módulo de cámara es opcional, necesita ser importado e inicializado
manualmente como se muestra arriba.


Captura de una sola imagen
--------------------------

Ahora repasaremos el caso más simple en el que abrimos una cámara y capturamos 
un cuadro como Surface. En el siguiente ejemplo, asumimos que hay una cámara 
en /dev/video0 en la computadora, y la inicializamos con un tamaño de 640 por 480.
La Surface llamada 'image' es lo que sea que la cámara estaba viendo cuando 
get_image() fue llamada. ::

    cam = pygame.camera.Camera("/dev/video0",(640,480))
    cam.start()
    image = cam.get_image()


Listado de Cámaras Conectadas
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Puede que se estén preguntando, ¿y si no sabemos la ruta exacta de la 
cámara? Podemos pedirle al módulo que nos proporcione la lista de 
cámaras conectadas a la computadora y que inicialice la primera 
cámara en la lista. ::

    camlist = pygame.camera.list_cameras()
    if camlist:
        cam = pygame.camera.Camera(camlist[0],(640,480))


Uso los Controles de la Cámara
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

La mayoría de las cámaras admiten controles como voltear la imagen y cambiar 
el brillo. set_controls() y get_controls() pueden ser usados en cualquier 
momento después de usar start(). ::

    cam.set_controls(hflip = True, vflip = False)
    print camera.get_controls()


Captura de una Transmisión en Vivo
----------------------------------

El resto de este tutorial se basará en capturar un flujo en vivo de
imagenes. Para ello usaremos la clase que se muestra a continuación.
Como se describe, simplemente mostrará (blit) en la pantalla una 
corriente constante de cuadros a la pantalla, mostrando efecivamente 
un video en vivo. Básicamente es lo que se espera, hacer un bucle con 
get_image(), se aplica a la pantalla de Surface, y lo voltea. Por 
razones de rendimiento, suministraremos a la cámara la misma Surface 
para utilizar en cada ocasión. ::

  class Capture:
      def __init__(self):
          self.size = (640,480)
          # crear una visualización de surface. cosas estándar de pygame
          self.display = pygame.display.set_mode(self.size, 0)
  
          # esto es lo mismo que vimos antes
          self.clist = pygame.camera.list_cameras()
          if not self.clist:
              raise ValueError("Sorry, no se detectaron cámaras.")
          self.cam = pygame.camera.Camera(self.clist[0], self.size)
          self.cam.start()
  
          # crear una surface para capturar, con fines de rendimiento
          # profundidad de bit es la misma que la de la Surface de visualización.
          self.snapshot = pygame.surface.Surface(self.size, 0, self.display)
  
      def get_and_flip(self):
          # si no querés vincular la velocidad de cuadros a la cámara, podés verificar
          # si la cámara tiene una imagen lista. Tené en cuenta que mientras esto funciona
          # en la mayoría de las cámaras, algunas nunca van a devolver un 'true'.
          if self.cam.query_image():
              self.snapshot = self.cam.get_image(self.snapshot)
  
          # pasalo (blit) a la Surface de visualización. ¡Simple!
          self.display.blit(self.snapshot, (0,0))
          pygame.display.flip()
  
      def main(self):
          going = True
          while going:
              events = pygame.event.get()
              for e in events:
                  if e.type == QUIT or (e.type == KEYDOWN and e.key == K_ESCAPE):
                      # cerrar la cámara de forma segura
                      self.cam.stop()
                      going = False
  
              self.get_and_flip()


Dado que get_image() es una llamada de bloqueo que podría tomar bastante tiempo 
en una cámara lenta, este ejemplo usa query_image() para ver si la cámara está 
lista. Esto permite separar la velocidad de fotogramas de tu juego de la 
de tu cámara. También es posible hacer que la cámara capture imágenes en un 
subproceso separado obteniendo aproximadamente la misma ganancia de rendimiento, 
si encontrás que tu cámara no es compatible con la función query_image().


Visión Básica por Computadora 
-----------------------------

Al usar los módulos de la cámara, transormación y máscara, pygame puede 
hacer algo de visión por computadora básica.


Modelos de Color
^^^^^^^^^^^^^^^^^

When initializing a camera, colorspace is an optional parameter, with 'RGB',
'YUV', and 'HSV' as the possible choices.  YUV and HSV are both generally more
useful for computer vision than RGB, and allow you to more easily threshold by
color, something we will look at later in the tutorial.

::

  self.cam = pygame.camera.Camera(self.clist[0], self.size, "RGB")

.. image:: ../../reST/tut/camera_rgb.jpg
   :class: trailing

::

  self.cam = pygame.camera.Camera(self.clist[0], self.size, "YUV")

.. image:: ../../reST/tut/camera_yuv.jpg
   :class: trailing

::

  self.cam = pygame.camera.Camera(self.clist[0], self.size, "HSV")

.. image:: ../../reST/tut/camera_hsv.jpg
   :class: trailing


Thresholding (Umbralización)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Usando la función threshold() del módulo de transformación, uno puede hacer 
simple efectos del estilo de pantalla verde o asilar objetos de colores 
especificos en una escena. En el siguiente ejemplo, usamos umbralización 
para separar el árbol verde y hacemos que el resto de la imagen sea negra. 
Consultá la documentación de referencia para más detalles de la función: 
:func:`threshold function <pygame.transform.threshold>`\ .

::

  self.thresholded = pygame.surface.Surface(self.size, 0, self.display)
  self.snapshot = self.cam.get_image(self.snapshot)
  pygame.transform.threshold(self.thresholded,self.snapshot,(0,255,0),(90,170,170),(0,0,0),2)

.. image:: ../../reST/tut/camera_thresholded.jpg
   :class: trailing


Por supuesto, esto solo es útil si ya conocés el color exacto del objeto que estás 
buscando. Para evitar esto y hacer la umbralización utilizable en el mundo real, 
necesitamos agregar una etapa de calibración en la que identifiquemos el color de 
un objeto y usarlo para umbralizarlo. Nosotros usaremos la función average_color() 
del módulo de transformación para hacer esto. A continuación, se muestra un 
ejemplo de la función calibración que se podría repetir hasta que se produzca 
un evento como apretar una tecla y [obtener] una imagen de cómo se vería. 
El color adentro del cuadro será el que se use para la umbralización. Hay que 
tener en cuenta que estamos usando el modelo de color HSV en las imágenes 
a continuación.

::

  def calibrate(self):
      # capturar la imagen
      self.snapshot = self.cam.get_image(self.snapshot)
      # aplicarlo a la Surface de visualización
      self.display.blit(self.snapshot, (0,0))
      # crear un rectángulo (rect) en el medio de la pantalla
      crect = pygame.draw.rect(self.display, (255,0,0), (145,105,30,30), 4)
      # obtener el color promedio del área dentro del rect
      self.ccolor = pygame.transform.average_color(self.snapshot, crect)
      # rellenar la esquina superior izquierda con ese color
      self.display.fill(self.ccolor, (0,0,50,50))
      pygame.display.flip()

.. image:: ../../reST/tut/camera_average.jpg
   :class: trailing

::

  pygame.transform.threshold(self.thresholded,self.snapshot,self.ccolor,(30,30,30),(0,0,0),2)

.. image:: ../../reST/tut/camera_thresh.jpg
   :class: trailing


Pueden usar la misma idea para hacer una simple pantalla verde/azul, al obtener 
primero una imagen del fondo y después umbralizar contrastando con ella. El 
ejemplo a continuación solo tiene la cámara apuntando a una pared blanca en 
modelo de color HSV.

::

  def calibrate(self):
      # captura un montón de imagenes de fondo.
      bg = []
      for i in range(0,5):
        bg.append(self.cam.get_image(self.background))
      # promedia el color de las imágenes para llegar a uno solo y deshacerse de posibles perturbaciones
      pygame.transform.average_surfaces(bg,self.background)
      # aplicarlo a la Surface de visualización
      self.display.blit(self.background, (0,0))
      pygame.display.flip()

.. image:: ../../reST/tut/camera_background.jpg
   :class: trailing

::

  pygame.transform.threshold(self.thresholded,self.snapshot,(0,255,0),(30,30,30),(0,0,0),1,self.background)

.. image:: ../../reST/tut/camera_green.jpg
   :class: trailing


Uso del Módulo de Máscara
^^^^^^^^^^^^^^^^^^^^^^^^^

Lo anterior es genial si solo querés mostrar imágenes, pero con el módulo 
:mod:`mask module <pygame.mask>`, también podés usar la cámara como 
dispositivo de entrada para un juego. Por ejemplo, volviendo al ejemplo 
de la umbralización de un objeto específico, podemos encontrar la posición 
de ese objeto y usarlo para controlar un objeto en la pantalla.

::

  def get_and_flip(self):
      self.snapshot = self.cam.get_image(self.snapshot)
      # umbralizar contra el color que obtuvimos antes
      mask = pygame.mask.from_threshold(self.snapshot, self.ccolor, (30, 30, 30))
      self.display.blit(self.snapshot,(0,0))
      # mantener solo el manchón más grande de ese color
      connected = mask.connected_component()
      # asegurarse que el manchón sea lo suficientemente grande, que no sea solo perturbaciones
      if mask.count() > 100:
          # encontrar el centro del manchónfind the center of the blob
          coord = mask.centroid()
          # dibujar un círculo con un tamaño variable en el tamaño del manchón
          pygame.draw.circle(self.display, (0,255,0), coord, max(min(50,mask.count()/400),5))
      pygame.display.flip()

.. image:: ../../reST/tut/camera_mask.jpg
   :class: trailing


Este es solo el ejemplo más básico. Podés rastrear múltiples manchas de diferentes 
colores, encontrar los contornos de los objetos, tener detección de colisiones 
entre objetos de la vida real y del juego, obtener el ángulo de un objetos 
para permitir su control, y más. ¡A divertirse!

