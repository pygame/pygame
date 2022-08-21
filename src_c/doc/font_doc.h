/* Auto generated file: with makeref.py .  Docs go in docs/reST/ref/ . */
#define DOC_PYGAMEFONT "pygame module for loading and rendering fonts"
#define DOC_PYGAMEFONTINIT "init() -> None\ninitialize the font module"
#define DOC_PYGAMEFONTQUIT "quit() -> None\nuninitialize the font module"
#define DOC_PYGAMEFONTGETINIT "get_init() -> bool\ntrue if the font module is initialized"
#define DOC_PYGAMEFONTGETDEFAULTFONT "get_default_font() -> string\nget the filename of the default font"
#define DOC_PYGAMEFONTGETSDLTTFVERSION "get_sdl_ttf_version(linked=True) -> (major, minor, patch)\ngets SDL_ttf version"
#define DOC_PYGAMEFONTGETFONTS "get_fonts() -> list of strings\nget all available fonts"
#define DOC_PYGAMEFONTMATCHFONT "match_font(name, bold=False, italic=False) -> path\nfind a specific font on the system"
#define DOC_PYGAMEFONTSYSFONT "SysFont(name, size, bold=False, italic=False) -> Font\ncreate a Font object from the system fonts"
#define DOC_PYGAMEFONTFONT "Font(filename, size) -> Font\nFont(pathlib.Path, size) -> Font\nFont(object, size) -> Font\ncreate a new Font object from a file"
#define DOC_FONTBOLD "bold -> bool\nGets or sets whether the font should be rendered in (faked) bold."
#define DOC_FONTITALIC "italic -> bool\nGets or sets whether the font should be rendered in (faked) italics."
#define DOC_FONTUNDERLINE "underline -> bool\nGets or sets whether the font should be rendered with an underline."
#define DOC_FONTSTRIKETHROUGH "strikethrough -> bool\nGets or sets whether the font should be rendered with a strikethrough."
#define DOC_FONTRENDER "render(text, antialias, color, background=None) -> Surface\ndraw text on a new Surface"
#define DOC_FONTSIZE "size(text) -> (width, height)\ndetermine the amount of space needed to render text"
#define DOC_FONTSETUNDERLINE "set_underline(bool) -> None\ncontrol if text is rendered with an underline"
#define DOC_FONTGETUNDERLINE "get_underline() -> bool\ncheck if text will be rendered with an underline"
#define DOC_FONTSETSTRIKETHROUGH "set_strikethrough(bool) -> None\ncontrol if text is rendered with a strikethrough"
#define DOC_FONTGETSTRIKETHROUGH "get_strikethrough() -> bool\ncheck if text will be rendered with a strikethrough"
#define DOC_FONTSETBOLD "set_bold(bool) -> None\nenable fake rendering of bold text"
#define DOC_FONTGETBOLD "get_bold() -> bool\ncheck if text will be rendered bold"
#define DOC_FONTSETITALIC "set_italic(bool) -> None\nenable fake rendering of italic text"
#define DOC_FONTMETRICS "metrics(text) -> list\ngets the metrics for each character in the passed string"
#define DOC_FONTGETITALIC "get_italic() -> bool\ncheck if the text will be rendered italic"
#define DOC_FONTGETLINESIZE "get_linesize() -> int\nget the line space of the font text"
#define DOC_FONTGETHEIGHT "get_height() -> int\nget the height of the font"
#define DOC_FONTGETASCENT "get_ascent() -> int\nget the ascent of the font"
#define DOC_FONTGETDESCENT "get_descent() -> int\nget the descent of the font"


/* Docs in a comment... slightly easier to read. */

/*

pygame.font
pygame module for loading and rendering fonts

pygame.font.init
 init() -> None
initialize the font module

pygame.font.quit
 quit() -> None
uninitialize the font module

pygame.font.get_init
 get_init() -> bool
true if the font module is initialized

pygame.font.get_default_font
 get_default_font() -> string
get the filename of the default font

pygame.font.get_sdl_ttf_version
 get_sdl_ttf_version(linked=True) -> (major, minor, patch)
gets SDL_ttf version

pygame.font.get_fonts
 get_fonts() -> list of strings
get all available fonts

pygame.font.match_font
 match_font(name, bold=False, italic=False) -> path
find a specific font on the system

pygame.font.SysFont
 SysFont(name, size, bold=False, italic=False) -> Font
create a Font object from the system fonts

pygame.font.Font
 Font(filename, size) -> Font
 Font(pathlib.Path, size) -> Font
 Font(object, size) -> Font
create a new Font object from a file

pygame.font.Font.bold
 bold -> bool
Gets or sets whether the font should be rendered in (faked) bold.

pygame.font.Font.italic
 italic -> bool
Gets or sets whether the font should be rendered in (faked) italics.

pygame.font.Font.underline
 underline -> bool
Gets or sets whether the font should be rendered with an underline.

pygame.font.Font.strikethrough
 strikethrough -> bool
Gets or sets whether the font should be rendered with a strikethrough.

pygame.font.Font.render
 render(text, antialias, color, background=None) -> Surface
draw text on a new Surface

pygame.font.Font.size
 size(text) -> (width, height)
determine the amount of space needed to render text

pygame.font.Font.set_underline
 set_underline(bool) -> None
control if text is rendered with an underline

pygame.font.Font.get_underline
 get_underline() -> bool
check if text will be rendered with an underline

pygame.font.Font.set_strikethrough
 set_strikethrough(bool) -> None
control if text is rendered with a strikethrough

pygame.font.Font.get_strikethrough
 get_strikethrough() -> bool
check if text will be rendered with a strikethrough

pygame.font.Font.set_bold
 set_bold(bool) -> None
enable fake rendering of bold text

pygame.font.Font.get_bold
 get_bold() -> bool
check if text will be rendered bold

pygame.font.Font.set_italic
 set_italic(bool) -> None
enable fake rendering of italic text

pygame.font.Font.metrics
 metrics(text) -> list
gets the metrics for each character in the passed string

pygame.font.Font.get_italic
 get_italic() -> bool
check if the text will be rendered italic

pygame.font.Font.get_linesize
 get_linesize() -> int
get the line space of the font text

pygame.font.Font.get_height
 get_height() -> int
get the height of the font

pygame.font.Font.get_ascent
 get_ascent() -> int
get the ascent of the font

pygame.font.Font.get_descent
 get_descent() -> int
get the descent of the font

*/