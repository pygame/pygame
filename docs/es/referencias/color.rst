.. include:: ../../reST/common.txt

:mod:`pygame.Color`
===================

.. currentmodule:: pygame

.. class:: Color

   | :sl:`pygame object for color representations`
   | :sg:`Color(r, g, b) -> Color`
   | :sg:`Color(r, g, b, a=255) -> Color`
   | :sg:`Color(color_value) -> Color`

   La clase ``Color`` representa valores de color ``RGBA`` utilizando un rango 
   de valores de 0 a 255 inclusive. Permite realizar operaciones aritméticas 
   básicas, como operaciones binarias ``+``, ``-``, ``*``, ``//``, ``%``, y 
   unaria ``~`` para crear nuevos colores. Admite conversiones a otros espacios 
   de colores como ``HSV`` o ``HSL``, y te permite ajustar canales individuales 
   de color. 
   El valor alfa se establece en 255 (completamente opaco) de forma predeterminada 
   si no se proporciona.
   Las operaciones aritméticas y método ``correct_gamma()`` conservan las subclases. 
   Para los operadores binarios, la clase de color devuelto es la del objeto de 
   color de la parte izquierda del operador.

   Los objetos de color admiten comparación de igualdad con otros objetos de 
   color y tuplas de 3 o 4 elementos de enteros. Hubo un error en pygame 1.8.1 
   donde el valor alfa predeterminado era 0, no 255 como antes.

   Los objetos de color exportan la interfaz de array a nivel C. La interfaz 
   exporta un array de bytes no firmados unidimensional de solo lectura con 
   la misma longitud asignada que el color. También se exporta la nueva 
   interfaz del búfer, con la mismas características que la interfaz del 
   array.

   Los operadores de división entera, ``//``, y módulo, ``%``, no generan 
   una excepción por división por cero. En su lugar, si un canal de color, 
   o alfa, en el color de la parte derecha es 0, entonces el resultado es 
   1. Por ejemplo: ::

       # Estas expresiones son True (verdaderas)
       Color(255, 255, 255, 255) // Color(0, 64, 64, 64) == Color(0, 3, 3, 3)
       Color(255, 255, 255, 255) % Color(64, 64, 64, 0) == Color(63, 63, 63, 0)

   Usa ``int(color)`` para obtener el valor entero inmutable del color, 
   que se puede utilizar como clave en un diccionario. Este valor entero 
   difiere de los valores de píxeles mapeados de los métodos 
   :meth:`pygame.Surface.get_at_mapped`, :meth:`pygame.Surface.map_rgb` 
   y :meth:`pygame.Surface.unmap_rgb`. 
   Se puede pasar como argumento ``color_value`` a :class:`Color`
   (útil con conjuntos).

   Ver :doc:`color_list` para ejemplos de nombres de colores disponibles.

   :param int r: el valor rojo en el rango de 0 a 255 inclusive
   :param int g: el valor verde en el rango de 0 a 255 inclusive
   :param int b: el color azul en el rango de 0 a 255 inclusive
   :param int a: (opcional) valor alfa en el rango de 0 a 255 inclusive,
      predeterminado es 255
   :param color_value: valor del color (ver nota abajo para los formatos admitidos)

      .. note::
         Formatos de ``color_value`` admitidos:
            | - **Objeto Color:** clona el objeto de clase :class:`Color` 
            | - **Nombre de color: str:** nombre del color a utilizar, por ejemplo ``'red'``
              (todos los nombres admitidos se pueden encontrar en :doc:`color_list`, 
              con muestras de ejemplo)
            | - **Formato de color HTML str:** ``'#rrggbbaa'`` o ``'#rrggbb'``,
              donde rr, gg, bb, y aa son números hexadecimales de 2 digitos 
              en el rango de 0 a 0xFF inclusive, el valor aa (alfa) se 
              establece en 0xFF de forma predeterminada si no se proporciona
            | - **Número hexadecimal str:** ``'0xrrggbbaa'`` o ``'0xrrggbb'``,
              donde rr, gg, bb, y aa son números hexadecimales de 2 digitos 
              en el rango de 0x00 a 0xFF inclsuvie, el valor aa (alfa) se 
              establece en 0xFF de forma predeterminada si no se proporciona.
            | - **int:** valor entero del color a utilizar, usar números 
              hexadecimales pueden hacer que este parámetro sea más legible, 
              por ejemplo, ``0xrrggbbaa``, donde rr, gg, bb, y aa son números
              hexadecimales de dos dígitos en el rango de 0x00 a 0xFF inclusive, 
              notese que el valor aa (alfa) no es opcional para el formato int y 
              debe ser proporcionado.
            | - **tupla/lista de valores enteros de color:** ``(R, G, B, A)`` o
              ``(R, G, B)``, donde R, G, B, y A son valores enteros en el rango 
              de 0 a 255 inclusive, el valor A (alfa) se establece en 255 de 
              forma predeterminada si no se proporciona.

   :type color_value: Color or str or int or tuple(int, int, int, [int]) or
      list(int, int, int, [int])

   :returns: a newly created :class:`Color` object
   :rtype: Color

   .. versionchanged:: 2.0.0
      Soporte para tuplas, listas y objetos :class:`Color` al crear objetos
      :class:`Color`.
   .. versionchanged:: 1.9.2 Color objects export the C level array interface.
   .. versionchanged:: 1.9.0 Color objects support 4-element tuples of integers.
   .. versionchanged:: 1.8.1 New implementation of the class.

   .. attribute:: r

      | :sl:`Gets or sets the red value of the Color.`
      | :sg:`r -> int`

      El valor rojo del color.

      .. ## Color.r ##

   .. attribute:: g

      | :sl:`Gets or sets the green value of the Color.`
      | :sg:`g -> int`

      El valor verde del color.

      .. ## Color.g ##

   .. attribute:: b

      | :sl:`Gets or sets the blue value of the Color.`
      | :sg:`b -> int`

      El valor azul del color.

      .. ## Color.b ##

   .. attribute:: a

      | :sl:`Gets or sets the alpha value of the Color.`
      | :sg:`a -> int`

      El valor alfa del color.

      .. ## Color.a ##

   .. attribute:: cmy

      | :sl:`Gets or sets the CMY representation of the Color.`
      | :sg:`cmy -> tuple`

      La representación ``CMY`` del color. 
      Los componentes ``CMY`` están en los rangos ``C`` = [0, 1], ``M`` = [0, 1], 
      ``Y`` = [0, 1].
      Tené en cuenta que estos no devolverá los valores ``CMY`` exactos para 
      los valores ``RGB`` establecidos en todos los casos. Debido a la asignación 
      de ``RGB`` de 0-255 y la asignación de ``CMY``de 0-1, los errores de 
      redondeo pueden hacer que los valores ``CMY`` difieran ligeramente de lo
      que podrías esperar.

      .. ## Color.cmy ##

   .. attribute:: hsva

      | :sl:`Gets or sets the HSVA representation of the Color.`
      | :sg:`hsva -> tuple`

      La representación ``HSVA`` del color. 
      Los componentes ``HSVA`` están en los rangos ``H`` = [0, 360], ``S`` = [0, 100],
      ``V`` = [0, 100], A = [0, 100]. 
      Tené en cuenta que esto devolverá los valores ``HSV`` exactos para los 
      valores ``RGB`` establecidos en todos los casos. Debido a la asignación 
      de ``RGB`` de 0-255 y la asignación de ``HSV`` de 0-100 y 0-360, los errores
      de redondeo pueden hacer que los valores ``HSV`` difieran ligeramente de 
      lo que podrías esperar.

      .. ## Color.hsva ##

   .. attribute:: hsla

      | :sl:`Gets or sets the HSLA representation of the Color.`
      | :sg:`hsla -> tuple`

      La representación ``HSLA`` del color.
      Los componentes ``HSLA`` están en rangos ``H`` = [0, 360], ``S`` = [0, 100],
      ``L`` = [0, 100], A = [0, 100]. Tené en cuenta que esto no devolverá los 
      valores ``HSL`` exactos para los valores ``RGB`` establecidos en todos los 
      casos. Debido a la asignación de ``RGB`` de 0-255 y la asignación de ``HSL`` 
      de 0-100 y 0-360, los errores de redondeo pueden hacer que los valores ``HSL`` 
      difieran ligeramente de lo que podrías esperar.

      .. ## Color.hsla ##

   .. attribute:: i1i2i3

      | :sl:`Gets or sets the I1I2I3 representation of the Color.`
      | :sg:`i1i2i3 -> tuple`

      La representación ``I1I2I3`` del color.
      Los componentes ``I1I2I3`` están en los rangos ``I1`` = [0, 1],
      ``I2`` = [-0.5, 0.5], ``I3`` = [-0.5, 0.5]. 
      Tené en cuenta que esto no devolverá los valores ``I1I2I3`` exactos 
      para los valores ``RGB`` establecidos en todos los cosas. Debido a 
      la asignación de ``RGB`` de 0-255 y la asignación ``I1I2I3`` de 0-1, 
      los errores de redondeo pueden hacer que los valores ``I1I2I3``difieran 
      ligeramente de lo que podrías esperar.

      .. ## Color.i1i2i3 ##

   .. method:: normalize

      | :sl:`Returns the normalized RGBA values of the Color.`
      | :sg:`normalize() -> tuple`

      Devuelve los valores ``RGBA`` del color como valores de punto flotante.

      .. ## Color.normalize ##

   .. method:: correct_gamma

      | :sl:`Applies a certain gamma value to the Color.`
      | :sg:`correct_gamma (gamma) -> Color`

      Aplica un cierto valor de gamma al color y devuelve un nuevo color con 
      los valores ``RGBA`` ajustados.

      .. ## Color.correct_gamma ##

   .. method:: set_length

      | :sl:`Set the number of elements in the Color to 1,2,3, or 4.`
      | :sg:`set_length(len) -> None`

      DEPRECATED: Puedes desempaquetar los valores que necesitas
      de la siguiente manera: 
      ``r, g, b, _ = pygame.Color(100, 100, 100)``
      si solo deseas r, g and b
      o
      ``r, g, *_ = pygame.Color(100, 100, 100)`` 
      si solo deseas r y g

      La longitud predeterminada de un color es 4. Los colores pueden tener 
      longitudes 1, 2, 3 o 4. Esto es útil si querés desempaquetar a r,g,b,a.
      Si querés obtener la longitud de un color, usa ``len(acolor)``.

      .. deprecated:: 2.1.3
      .. versionadded:: 1.9.0

      .. ## Color.set_length ##

   .. method:: grayscale

      | :sl:`returns the grayscale of a Color`
      | :sg:`grayscale() -> Color`

      Devuelve un color que representa la versión en escala de grises de sí mismo 
      utilizando la fórmula de luminosidad que pondera el rojo, verde y azul 
      según sus longitudes de onda.

      .. ## Color.grayscale ##

   .. method:: lerp

      | :sl:`returns a linear interpolation to the given Color.`
      | :sg:`lerp(Color, float) -> Color`

      Devuelve un color que es una interpolación entre sí mismo y el color 
      dado en el espacio RGBA. El segundo parámetro determina qué tan lejos 
      estará el resultado entre sí mismo y el otro color. Debe ser un valor 
      entre 0 y 1, donde 0 significa que devolverá el color inicial, el de 
      sí mismo, y 1 significa que devolverá otro color.

      .. versionadded:: 2.0.1

      .. ## Color.lerp ##

   .. method:: premul_alpha

      | :sl:`returns a Color where the r,g,b components have been multiplied by the alpha.`
      | :sg:`premul_alpha() -> Color`

      Devuelve un nuevo color en el que cada uno de los canales de rojo, 
      verde y azul ha sido multiplicado por el canal alfa del color 
      original. El canal alfa permanece sin cambios.

      Esto es útil cuando se traba con la bandera de modo ``BLEND_PREMULTIPLIED`` 
      de mezcla para :meth:`pygame.Surface.blit()`, que asume que todas las superficies 
      que lo utilizan están utilizando colores con alfa pre-multiplicado.

      .. versionadded:: 2.0.0

      .. ## Color.premul_alpha ##

   .. method:: update

      | :sl:`Sets the elements of the color`
      | :sg:`update(r, g, b) -> None`
      | :sg:`update(r, g, b, a=255) -> None`
      | :sg:`update(color_value) -> None`

      Establece los elementos del color. Consulta los parámetros de :meth:`pygame.Color` 
      para los parámetros de esta función. Si el valor alfa no se estableció, no cambiará.

      .. versionadded:: 2.0.1

      .. ## Color.update ##
   .. ## pygame.Color ##
