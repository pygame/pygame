.. include:: ../../reST/common.txt

:mod:`pygame.camera`
====================

.. module:: pygame.camera
   :synopsis: módulo de pygame para el uso de la cámara

| :sl:`pygame module for camera use`

Actualmente, Pygame soporta cámaras nativas de Linux (V4L2) y Windows (MSMF),
con un soporte de plataforma más amplio disponible a través de un backend (controlador) 
integrado en OpenCV.

.. versionadded:: 2.0.2 Windows native camera support
.. versionadded:: 2.0.3 New OpenCV backends

¡EXPERIMENTAL!: Este API puede cambiar o desaparecer en lanzamientos posteriores de pygame.
Si lo utilizas, es muy probable que tu código se rompa en la próxima versión de pygame.

La función de Bayer a ``RGB`` se basa en:

::

 Sonix SN9C101 based webcam basic I/F routines
 Copyright (C) 2004 Takafumi Mizuno <taka-qce@ls-a.jp>
 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions
 are met:
 1. Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
 THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 SUCH DAMAGE.

Nuevo en pygame 1.9.0.

.. function:: init

   | :sl:`Module init`
   | :sg:`init(backend = None) -> None`

   Esta función inicia el módulo de la cámara, seleccionando el mejor controlador 
   (backend) de la cámara web que pueda encontrar en tu sistema. No se garantiza 
   que tenga éxito e incluso puede intentar importar módulos de teceros, como 
   `OpenCV`. Si deseas anular la elección de controlador (backend), podés hacer 
   un llamado para pasar el nombre del controlador que deseas a esta función. 
   Podés obtener más información sobre los controladores (backends) en la función
   :func:`get_backends()`.

   .. versionchanged:: 2.0.3 Option to explicitly select backend

   .. ## pygame.camera.init ##

.. function:: get_backends

   | :sl:`Get the backends supported on this system`
   | :sg:`get_backends() -> [str]`

   Este función devuelve cada controlador (backend) que considera que tienen 
   posibilidad de funcionar en tu sistema, en orden de prioridad.

   pygame.camera Backends:
   ::

      Backend           OS        Description
      ---------------------------------------------------------------------------------
      _camera (MSMF)    Windows   Builtin, works on Windows 8+ Python3
      _camera (V4L2)    Linux     Builtin
      OpenCV            Any       Uses `opencv-python` module, can't enumerate cameras
      OpenCV-Mac        Mac       Same as OpenCV, but has camera enumeration
      VideoCapture      Windows   Uses abandoned `VideoCapture` module, can't enumerate
                                  cameras, may be removed in the future

   Hay dos diferencias princiales entre los controladores (backends).

   Los controladores (backends) _camera están integrados en el mismo pygame 
   y no requieren importaciones de terceros. Todos los demás controlaores sí lo requieren.
   Para los controladores OpenCV y VideoCapture, esos módulos deben estar 
   instalados en tu sistema.

   La otra gran diferencia es "enumeración de cámaras". Algunos constroladores no 
   tienen una forma de enumerar los nombres de las cámaras o incluso la 
   cantidad de cámaras en el sistema. En estos casos, la función 
   :func:`list_cameras()` devolverá algo como ``[0]``. Si sabés que tenés 
   varias cámaras en el sistema, estos puertos de controladores pasarán un 
   "número de índice de cámara" si lo utilizas como el parámetro ``device``.

   .. versionadded:: 2.0.3

   .. ## pygame.camera.get_backends ##

.. function:: colorspace

   | :sl:`Surface colorspace conversion`
   | :sg:`colorspace(Surface, format, DestSurface = None) -> Surface`

   Permite la conversión "RGB" a un espacio de color destino de "HSV" o "YUV".
   Las surfaces (superficies) de origen y destino deben tener el mismo tamaño 
   y profundidad de píxel. Esto es útil para la visión por computadora de 
   dispositivos con capacidad de procesamiento limitada. Captura una imagen 
   lo más pequeña posible, la redimensiona con ``transform.scale()`` haciendola
   aún más pequeña, y luego convierte el espacio de color a "YUV" o "HSV" antes 
   de realizar cualquier procesamiento en ella.

   .. ## pygame.camera.colorspace ##

.. function:: list_cameras

   | :sl:`returns a list of available cameras`
   | :sg:`list_cameras() -> [cameras]`

   Verifica la disponibilidad de cámaras y devuelve una lista de cadenas de 
   nombres de cámaras, listas para ser utilizados por 
   :class:`pygame.camera.Camera`.

   Si el controlador (backend) de la cámara no soporta la enuemración de 
   webcams, esto devolverá algo como ``[0]``. Ver :func:`get_backends()`  
   para obtener mucha más información.


   .. ## pygame.camera.list_cameras ##

.. class:: Camera

   | :sl:`load a camera`
   | :sg:`Camera(device, (width, height), format) -> Camera`

   Carga una cámara. En Linux, el dispositivo suele ser algo como 
   "/dev/video0". El ancho y alto predeterminados son 640x480.
   El formato es el espacio de color deseado para la salida.
   Esto es útil para fines de visión por computadora. El valor 
   predeterminado es ``RGB``. Los siguientes formatos son compatibles:
   
      * ``RGB`` - Red, Green, Blue

      * ``YUV`` - Luma, Blue Chrominance, Red Chrominance

      * ``HSV`` - Hue, Saturation, Value

   .. method:: start

      | :sl:`opens, initializes, and starts capturing`
      | :sg:`start() -> None`

      Abre el dispositivo de la cámara, intenta inicializarlo y comienza a 
      grabar imágenes en un búfer. La cámara debe estar iniciada antes de 
      que se puedan utilizar las siguientes funciones.

      .. ## Camera.start ##

   .. method:: stop

      | :sl:`stops, uninitializes, and closes the camera`
      | :sg:`stop() -> None`

      Detiene la grabación, desinicializa la cámara y la cierra. Una vez que 
      la cámara se detiene, las funciones siguientes no se pueden utilizar 
      hasta que se inicie nuevamente.

      .. ## Camera.stop ##

   .. method:: get_controls

      | :sl:`gets current values of user controls`
      | :sg:`get_controls() -> (hflip = bool, vflip = bool, brightness)`

      Si la cámara lo admite, get_controls devolverá la configuración 
      actual para el volteo horizontal y vertical de la imagen como 
      booleanos y el brillo como un número entero. 
      Si no es compatible, devolverá los valores predeterminados (0, 0, 0).
      Hay que tener en cuenta que los valores de retorno acá pueden ser 
      diferentes a los devueltos por set_controls, aunque es más probable 
      que sean correctos.

      .. ## Camera.get_controls ##

   .. method:: set_controls

      | :sl:`changes camera settings if supported by the camera`
      | :sg:`set_controls(hflip = bool, vflip = bool, brightness) -> (hflip = bool, vflip = bool, brightness)`

      Te permite cambiar la configuración de la cámara si la cámara lo admite.
      Los valores devueltos serán los valores de la entrada si la cámara 
      afirma que tuvo éxito, o si no, los valores previamente utilizados. 
      Cada argumento es opcional y se puede elegir el deseado mediante 
      el suministro de una palabra clave, como hflip. Hay que tener en cuenta 
      que la configuración real siendo utilizada por la cámara puede no ser 
      la misma que la devuelta por set_controls. En Windows :code:`hflip` 
      y :code:`vflip` están implementados por pygame, no por la cámara, 
      por lo que siempre deberían funcionar, pero el brillo
      :code:`brightness` no está soportado.

      .. ## Camera.set_controls ##

   .. method:: get_size

      | :sl:`returns the dimensions of the images being recorded`
      | :sg:`get_size() -> (width, height)`

      Devuelve las dimensiones actuales de las imágenes capturadas por la 
      cámara. Esto devolverá el tamaño real, que puede ser diferente al 
      especificado durante la inicialización si la cámara no admite ese 
      tamaño.

      .. ## Camera.get_size ##

   .. method:: query_image

      | :sl:`checks if a frame is ready`
      | :sg:`query_image() -> bool`

      Si una imagen está lista, devuelve TRUE (verdadero). De lo contrario, 
      devuelve FALSE (falso). Hay que tener en cuenta que algunas webcams 
      siempre devolverán falso y solo pondrán en cola un cuadro cuando se les 
      llame con una función de bloqueo como :func:`get_image()`. 
      En Windows (MSMF), y en los backends de OpenCV, la función :func:`query_image()`
      debería ser confiable. Esto es útil para separar la frecuencia de cuadros 
      del juego de la velocidad de la cámara sin tener que usar subprocesos.

      .. ## Camera.query_image ##

   .. method:: get_image

      | :sl:`captures an image as a Surface`
      | :sg:`get_image(Surface = None) -> Surface`

      Extrae una imagen del búfer como una superficie ``RGB``. Opcionalmente,
      se puede reutilizar una superficie existente para ahorrar tiempo. La 
      profundidad de bits de la superficie es de 24 bits en Linux, 32 bits 
      en Windows, o la misma que la superficie suministrada opcionalmente.

      .. ## Camera.get_image ##

   .. method:: get_raw

      | :sl:`returns an unmodified image as bytes`
      | :sg:`get_raw() -> bytes`

      Obtiene una imagen de la cámara como una cadena en el formato de píxel 
      nativo de la cámara. Útil para la integración con las otras bibliotecas.
      Esto devuelve un objeto de bytes.

      .. ## Camera.get_raw ##

   .. ## pygame.camera.Camera ##

.. ## pygame.camera ##
