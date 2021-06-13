## Access documentation locally

### Prerequisite

- Python version 3.6 or greater
- Sphinx version 3.5.4

### Generation command

`python setup.py docs` is the key command to generate locally the pygame documentation.
It will create a `generated` folder at the root of the `docs` package containing all the 
required files to render the documentation exactly the same way as in the [online version](https://www.pygame.org/docs/).

This command could also be launch with `--fullgeneration` (or `--f`) option, specifying that the
documentation must be completely regenerated (unnecessary for the initial build).

Check the [online documentation about generating docs](https://www.pygame.org/wiki/Hacking?parent=#Generating%20docs)
for more technical information.

### Access to documentation

`python docs` is the command that will open the root page of the generated docs in your favorite
browser. In fact, it opens the `index.html` file that can be found in the `generated` folder.

If you didn't generate the documentation first, this command opens the **online** version instead.

## Contribute to docs

We strongly advice you to check the [Sphinx documentation](https://www.sphinx-doc.org/)
before making any change on this module.

### Editing theme

Sphinx is working with a system of themes to define the rendering of the generated documentation.
Currently, pygame defines its own theme which inherits from `basic`, a native theme of Sphinx.

The theme files can be found in `docs/reST/themes/classic` folder.

Some important files are:

- `theme.conf`: defines basic settings of the theme and options that can be used in other files
- `elements.html`: defines how the docs should be structured in HTML format, with possibility to use
  [Sphinx templating options](https://www.sphinx-doc.org/en/master/development/theming.html#templating) 
- `static/pygame.css_t`: defines how the docs should be stylised in CSS format, with possibility to
use templating options too

