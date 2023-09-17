## Descripción General de la Documentación de Pygame

### Acceso a la Documentación

Obviamente podés visitar pygame.org/docs para ver la documentación,
pero la documentación también está publicada con `python -m pygame.docs`

### Generación de Documentación

Pasos:
- Tener Python 3.6 or superior
- Instalar Sphinx (`pip install Sphinx==4.5.0`)
- Bifurcar (fork) el repositorio de pygame, descargar y navegar en la terminal
- ejecutar `python setup.py docs`

Esto va a crear una nueva carpeta dentro de la carpeta `docs`
En `docs/generated`, vas a encontrar una copia de la documentación de pygame.

Podés ejecutar esto haciendo click en index.html o ejecutando el comando 
`python -m docs` desde la carpeta de pygame. (Es lo mismo que ejecutar 
manualmente __main__.py en `docs/`). El comando de ejecución de la documentación
te dirigirá a un sitio web de pygame si no hay documentación generada localmente.

También hay un comando `docs --fullgeneration` o `docs --f` para regenerar 
todo sin importar si Sphinx considera que debería ser regenerado. Esto 
es útil cuando se edita el CSS del tema.

###  Contribuir

Sí ves errores gramaticales o errores en la documentación, 
contribuir en la documentación es una gran forma de ayudar.

Para cosas simples, no es necesario un issue -- pero si querés 
cambiar algo complejo lo mejor sería abrir un issue primero.

Algunos antecedentes que pueden ayudar con los cambios: la documentación 
de Pygame está escrita en archivo rst, que significa "ReStructured Text". 
Usamos Sphinx ([Sphinx Documentation](https://www.sphinx-doc.org/en/master/)) 
para convertir estos archivos rst en html, que luego se alojan en el sitio web 
de Pygame.

Sphinx tiene un buen tutorial de ReStructure Text para aprender lo básico:
https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html

Pasos para contribuir:
- Tener una idea para mejorar el docs, quizá crear un issue en GitHub
- Encontrar el archivo que querés editar: probablemente este en `docs/reST/ref`.
OR
- Las páginas de docs de Pygame tienen un botón "Edit on Github" (Editar en GitHub), que te mostrará el archivo
- Descarga el código fuente de pygame desde GitHub localmente.
     ^ Una forma de hacer esto es bifurcando (fork) y usar el cliente de Git para hacer de eso un repositorio local
- Implementa tu idea.
- Seguí los pasos en "Generando la Documentación"
     ^ Esto es importante para probar que los cambios funcionen bien
- Confirma (commit) tus cambios, crea una solicitud de incorporación de cambios (a pull request)

## Estilo de Documentación

Los archivos de documentación de pygame han adoptado la conveción de límite de línea de 79 caracteres, 
proveniente de PEP8.

También utilizan una indentación de 3 espacios.

## Detalles de la Implementación de la Documentación de Pygame

Este es un lugar para proporcionar explicaciones sobre cosas que pueden confundir a la gente en un 
futuro. 

### Módulos Ocultos

Pygame todavía tiene documentación para los antiguos módulos 'cdrom' y 'Overlay', 
que han sido descontinuados en SDL2 que se basa en pygame (pygame 2). Sin embargo, 
estos módulos no se muestran porque `docs/reST/themes/classic/elements.html` tiene 
ahora una lista de los módulos "prohibidos" para que no se los incluya en la barra 
superior. También se utiliza para el la documentación experimental sdl2_video.

### Diseño visual / Temas

Las reglas de CSS para el HTML generado provienen de 
`docs/reST/themes/classic/static/pygame.css_t`. A su vez, este hereda las reglas 
de basic.css de Sphinx, el cual se genera automáticamente cuando Sphinx construye 
la documentación.

Este es un ejemplo de una 
[plantilla estática de Sphinx](https://www.sphinx-doc.org/en/master/development/theming.html#static-templates)