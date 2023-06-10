.. include:: ../../reST/common.txt

.. default-domain:: py

:class:`pygame.BufferProxy`
===========================

.. currentmodule:: pygame

.. class:: BufferProxy

   | :sl:`pygame object to export a surface buffer through an array protocol`
   | :sg:`BufferProxy(<parent>) -> BufferProxy`

   :class:`BufferProxy` es un tipo de soporte de pygame, diseñado como el valor de retorno
   de los métodos :meth:`Surface.get_buffer` y :meth:`Surface.get_view`.
   Para todas las versiones de Python, un objeto :class:`BufferProxy` exporta una estructura C
   y un array de interface a nivel de Python en nombre del búfer del objeto principal.
   También se exporta una nueva interfaz de búfer.
   En pygame, :class:`BufferProxy` es clave para implementar el módulo :mod:`pygame.surfarray`.

   Las instancias :class:`BufferProxy` pueden ser creadas directamente desde 
   el código de Python, ya sea para un buffer superior que exporta una interfaz
   o a partir de un ``dict`` de Python que describe el diseño del búfer de un objeto.
   Las entradas del dict se basan en el mapeo de la interfaz de matriz a nivel de Python.
   Se reconocen las siguientes claves:

      ``"shape"`` : tupla
         La longitud de cada elemento del array como una tupla de enteros. 
         La longitud de la tupla es el número de dimensiones en el array.

      ``"typestr"`` : string
         El tipo de elemento del array como una cadena de longitud 3. El primer 
         carácter indica el orden de bytes, '<' para para formato little-endian, 
         ">" para formato big-endian, y '\|' si no es aplicable. El segundo 
         carácter es el tipo de elemento, 'i' para los enteros con signo, 'u' 
         para los enteros sin signo, 'f' para números de puntos flotantes, y 
         'V' para los conjuntos de bytes. El tercer carácter indica el tamaño 
         en bytes del elemento, desde '1' a '9' bytes. Por ejemplo, "<u4" es 
         un entero sin signo de 4 bytes en formato little-endian, como un píxel 
         de 32 bits en una PC, mientras que "\|V3" representaría un píxel de 
         24 bits, que no tiene un equivalente entero.

      ``"data"`` : tupla
         La dirección de inicio de el búfer físico y una bandera de solo lectura
         como una tupla de longitud 2. La dirección es un valor entero, mientras 
         que la bandera de solo lectura es un valor boleano- "Falso" para escribir,
         "Verdadero" para solo lectura.

      ``"strides"`` : tupla : (opcional)
        La información del array stride (el número de bytes a saltar para alcanzar 
        el próximo valor) representado como una tupla de enteros. Es requerido 
        únicamente para arreys que no son contiguos en C. La longitud de la tupla 
        debe coincidir con la de ``"shape"``.

      ``"parent"`` : object : (opcional)
         El objeto exportador. Puede ser usado para mantener vivo el objeto 
         superior (antecesor) mientras su búfer es visible.

      ``"before"`` : callable : (opcional)
         Callback (rellamada) invocado cuando la instancia 
         :class:`BufferProxy` exporta el búfer. El callback recibe un 
         argumento, el objeto ``"parent"`` se se proporciona, de lo 
         contrario, ``None``. El callback es útil para establecer un 
         bloqueo en el objeto superior o antecesor (parent).

      ``"after"`` : callable : (opcional)
         Callback (rellamada) invocado cuando se libera un búfer exportado.
         El callback recibe un argumento, el objeto ``"parent"``(antecesor)
         si se proporciona, de lo contrario, ``None``. El callback es útil
         para liberar un bloqueo en el objeto parent (superior o antecesor)
         
      
   La clase BufferProxy soporta subclases, variables de instancias y referencias 
   débiles

   .. versionadded:: 1.8.0
   .. versionextended:: 1.9.2

   .. attribute:: parent

      | :sl:`Return wrapped exporting object.`
      | :sg:`parent -> Surface`
      | :sg:`parent -> <parent>`

      La clase :class:`Surface` que devolvió el objeto de clase :class:`BufferProxy` o
      el objeto pasado a una llamada de :class:`BufferProxy`.

   .. attribute:: length

      | :sl:`The size, in bytes, of the exported buffer.`
      | :sg:`length -> int`

      El número de bytes validos de datos exportados. Para datos discotinuous,
      es decir, datos que no forman un solo bloque de memoria, los bytes dentro 
      de los espacios vacios se excluyen del conteo. Esta propiedad es equivalente 
      al campo "len" de la estructura C ``Py_buffer``.

   .. attribute:: raw

      | :sl:`A copy of the exported buffer as a single block of bytes.`
      | :sg:`raw -> bytes`

      Los datos del búfer como un objeto ``str``/``bytes``.
      Cualquier espacio vacío en los datos exportados se elimina.

   .. method:: write

      | :sl:`Write raw bytes to object buffer.`
      | :sg:`write(buffer, offset=0)`

      Sobreescribe bytes en el objeto superior (o antecesor). Los datos 
      deben ser contiguos en C o F, de lo contrario se genera un ValueError.
      El argumento `buffer` es un objeto ``str``/``bytes``. Un desplazamiento
      opcional proporciona una posición de inicio, en bytes, dentro del búfer 
      donde comienza la sobreescritura.
      Si el desplazamiento es negativo o mayor o igual que el valor :attr:`length` 
      del proxy, se genera un excepción ``IndexException``.
      Si ``len(buffer) > proxy.length + offset``, se genera un ``ValueError``.
