.. include:: ../../reST/common.txt

:mod:`pygame.cdrom`
===================

.. module:: pygame.cdrom
   :synopsis: módulo de pygame para el control de CD de audio.

| :sl:`pygame module for audio cdrom control`

.. warning::
	Este módulo no es funcional en pygame 2.0 y versiones superiores, a menos que hayas compilado manualmente pygame con SDL1.
	Este módulo no estará soportado en el futuro.
	Una alternativa para la funcionalidad de cdrom de Python es `pycdio <https://pypi.org/project/pycdio/>`_.

El módulo cdrom administra las unidades de ``CD`` y ``DVD`` en la computadora.
También puede controlar la reproducción de CD de audio. Este módulo debe 
inicializarse antes de poder hacer algo. Cada objeto ``CD``que crees 
representa una unidad de cdrom y también debe inicializarse individualmente 
antes de poder realizar la mayoría de las acciones.

.. function:: init

   | :sl:`initialize the cdrom module`
   | :sg:`init() -> None`

   Inicializa el módulo de cdrom. Esto escaneará el sistema en busca de 
   todos los dispositivos ``CD``. El módulo debe inicializarse antes de 
   que funcionen cualquier otra función. Esto ocurre automáticamente 
   cuando llamas ``pygame.init()``.

   Es seguro llamar a este función más de una vez.

   .. ## pygame.cdrom.init ##

.. function:: quit

   | :sl:`uninitialize the cdrom module`
   | :sg:`quit() -> None`

   Decinicializa el módulo cdrom. Después de llamar a esta función, cualquier 
   objeto ``CD`` existente dejará de funcionar.

   Es seguro llamar a esta función más de una vez.

   .. ## pygame.cdrom.quit ##

.. function:: get_init

   | :sl:`true if the cdrom module is initialized`
   | :sg:`get_init() -> bool`

   Comprueba si el módulo cdrom está inicializado o no. Esto es diferente de 
   ``CD.init()`` ya que cada unidad también debe inicializarse individualmente.

   .. ## pygame.cdrom.get_init ##

.. function:: get_count

   | :sl:`number of cd drives on the system`
   | :sg:`get_count() -> count`

   Devuelve el número de unidades de CD en el sistema. Cuando creas objetos 
   ``CD``, debes pasar un ID entero que debe ser menor que este recuento. El 
   recuento será 0 si no hay unidades en el sistema.

   .. ## pygame.cdrom.get_count ##

.. class:: CD

   | :sl:`class to manage a cdrom drive`
   | :sg:`CD(id) -> CD`

   Podés crear un objeto ``CD`` para cada unidad de CD en el sistema. 
   Usa ``pygame.cdrom.get_count()`` para determinar cuántas unidades 
   existen realmente. El argumento 'id' es un número entero que 
   representa la unidad, comenzando en cero.
   
   El objeto ``CD`` no está inicializado, solo podés llamar ``CD.get_id()` y 
   ``CD.get_name()`` en una unidad no inicializada.

   Es seguro crear múltiples objetos ``CD``para la misma unidad, todos 
   cooperarán normalmente. 

   .. method:: init

      | :sl:`initialize a cdrom drive for use`
      | :sg:`init() -> None`

      Inicializa la unidad de CD para ser utilizada. El debe estar inicializada 
      para que la mayoría de los métodos ``CD`` funcionen. Incluso si el resto 
      de pygame está inicializado.

      Puede haber una breve pausa mientras la unidad se inicializa. Evitá 
      utilizar ``CD.init()`` si el programa no debe detenerse durante uno 
      o dos segundos.

      .. ## CD.init ##

   .. method:: quit

      | :sl:`uninitialize a cdrom drive for use`
      | :sg:`quit() -> None`

      Desinicializa una unidad para su uso. Hacé un llamado a esto cuando tu 
      programa no vaya a acceder a la unidad durante un tiempo.

      .. ## CD.quit ##

   .. method:: get_init

      | :sl:`true if this cd device initialized`
      | :sg:`get_init() -> bool`

      Comprueba si este dispositivo ``CDROM`` está inicializado. Esto es 
      diferente de ``pygame.cdrom.init()`` ya que cada unidad también 
      debe inicializarse individualmente.

      .. ## CD.get_init ##

   .. method:: play

      | :sl:`start playing audio`
      | :sg:`play(track, start=None, end=None) -> None`

      Reproduce audio desde un CD de audio en la unidad. Además del argumento 
      del número de pista, también podés introducir un tiempo de inicio y fin 
      para la reproducción. El tiempo de inicio y fin está en segundos y  puede 
      limintar la selección de una pista de audio reproducida.

      Si introducir un tiempo de inicio pero no de fin, el audio se reproducirá 
      hasta el final de la pista. Si introducis un tiempo de inicio y 'None' 
      para el tiempo final, el audio se reproducirá hasta el final de todo el 
      disco.

      Véase ``CD.get_numtracks()`` y ``CD.get_track_audio()`` para encontrar 
      las pistas que se van a reproducir.

      Nota: la pista 0 es la primera pista en el ``CD``. Los números de pistas 
      comienzan en 0.

      .. ## CD.play ##

   .. method:: stop

      | :sl:`stop audio playback`
      | :sg:`stop() -> None`

      Detiene la reproducción del audio desde el CD-ROM. También se perderá la 
      posición actual de reproducción. Este método no hace nada si la unidad 
      no está reproduciendo audio.

      .. ## CD.stop ##

   .. method:: pause

      | :sl:`temporarily stop audio playback`
      | :sg:`pause() -> None`

      Detiene temporalmente la reproducción del audio en el ``CD``. La 
      reproducción puede reanudarse en el mismo punto con el método 
      ``CD.resume()``. Si el ``CD`` no está reproduciendo, este método 
      no hace nada.

      Nota: la pista 0 es la primera en el ``CD``. Los números de pista 
      comienzan en cero.

      .. ## CD.pause ##

   .. method:: resume

      | :sl:`unpause audio playback`
      | :sg:`resume() -> None`

      Reanuda la reproducción de un ``CD``. Si el ``CD`` no está en pausa 
      o ya se está reproduciendo, este método no hace nada.

      .. ## CD.resume ##

   .. method:: eject

      | :sl:`eject or open the cdrom drive`
      | :sg:`eject() -> None`

      Esto abrirá la unidad de CD y expulsará el CD-ROM. Si la unidad 
      está reproduciendo o en pausa, se detendrá.

      .. ## CD.eject ##

   .. method:: get_id

      | :sl:`the index of the cdrom drive`
      | :sg:`get_id() -> id`

      Devuelve el ID entero que se utilizó para crear la instancia de ``CD``.
      Este método puede funcionar en un ``CD`` no inicializado.

      .. ## CD.get_id ##

   .. method:: get_name

      | :sl:`the system name of the cdrom drive`
      | :sg:`get_name() -> name`

      Devuelve el nombre de la unidad en forma de cadena. Este es el nombre 
      de sitema utilizado para representar la unidad, a menudo es la letra 
      de la unidad o el nombre del dispositivo. Este método puede funcionar 
      en un ``CD`` no inicializado. 

      .. ## CD.get_name ##

   .. method:: get_busy

      | :sl:`true if the drive is playing audio`
      | :sg:`get_busy() -> bool`

      Devuelve True (verdadero) si la unidad está ocupada reproduciendo audio.
      

      .. ## CD.get_busy ##

   .. method:: get_paused

      | :sl:`true if the drive is paused`
      | :sg:`get_paused() -> bool`

      Devuelve True (verdadero) si la unidad está actualmente en pausa.

      .. ## CD.get_paused ##

   .. method:: get_current

      | :sl:`the current audio playback position`
      | :sg:`get_current() -> track, seconds`

      Devuelve tanto la pista actual como el tiempo de esa pista. Este método 
      funciona cuando la unidad está reproduciendo o en pausa.
      
      Nota: la pista 0 es la primera pista en el ``CD``. Los números de pista 
      comienzan en cero.

      .. ## CD.get_current ##

   .. method:: get_empty

      | :sl:`False if a cdrom is in the drive`
      | :sg:`get_empty() -> bool`

      Devuelve False (falso) si hay un CD-ROM en la unidad actualmente. Si la 
      unidad está vacía devolverá True (verdadero).

      .. ## CD.get_empty ##

   .. method:: get_numtracks

      | :sl:`the number of tracks on the cdrom`
      | :sg:`get_numtracks() -> count`

      Devuelve el número de pistas en el CD-ROM de la unidad. Esto devolverá 
      cero si la unidad está vacía o no tiene pistas.

      .. ## CD.get_numtracks ##

   .. method:: get_track_audio

      | :sl:`true if the cdrom track has audio data`
      | :sg:`get_track_audio(track) -> bool`

      Determina si una pista en un CD-ROM contiene datos de audio. También 
      podés llamar a ``CD.num_tracks()`` y ``CD.get_all()`` para obtener 
      más información sobre el CD-ROM.

      Nota: la pista 0 es la primera pista en el ``CD``. Los números de pistas 
      comienzan en cero.

      .. ## CD.get_track_audio ##

   .. method:: get_all

      | :sl:`get all track information`
      | :sg:`get_all() -> [(audio, start, end, length), ...]`

      Devuelve una lista con información para cada pista en el CD-ROM. La 
      información consiste en una tupla con cuatro valores. El valor "audio"
      es True (verdadero) si la pista contiene data de audio. Los valores 
      de inicio, fin y longitud son números de puntos flotantes en segundos.
      "Start" (inicio) y "end" (fin) representan tiempos absolutos en todo 
      el disco.
      
      .. ## CD.get_all ##

   .. method:: get_track_start

      | :sl:`start time of a cdrom track`
      | :sg:`get_track_start(track) -> seconds`

      Devuelve el tiempo absoluto en segundos al inicio de la pista del CD-ROM.

      Nota: la pista 0 es la primera pista del ``CD``. Los números de pista 
      comienzan en cero.

      .. ## CD.get_track_start ##

   .. method:: get_track_length

      | :sl:`length of a cdrom track`
      | :sg:`get_track_length(track) -> seconds`

      Devuelve un valor de punto flotante en segundos de la duración de 
      la pista del CD-ROM. 

      Nota: la pista 0 es la primera pista del ``CD``. Los números de pista 
      comienzan en cero.

      .. ## CD.get_track_length ##

   .. ## pygame.cdrom.CD ##

.. ## pygame.cdrom ##
