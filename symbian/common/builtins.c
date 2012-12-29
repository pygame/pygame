#include <Python.h>

extern void initbase(void);
extern void initconstants(void);
extern void initrect(void);
extern void initrwobject(void);
extern void init_numericsndarray (void);
extern void init_numericsurfarray (void);
extern void initbufferproxy (void);
extern void initcolor (void);
extern void initconstants (void);
extern void initdisplay (void);
extern void initdraw(void);
extern void initevent (void);
extern void initfastevent (void);
extern void initfont (void);
extern void initgfxdraw(void);
extern void initimage (void);
extern void initimageext (void);
extern void initkey (void);
extern void initmixer (void);
extern void initmovie (void);
extern void initmovieext(void);
extern void initmixer_music (void);
extern void initoverlay (void);
extern void initpixelarray (void);
extern void initsurface(void);
extern void initsurflock (void);
extern void initpygame_time (void);
extern void inittransform (void);

struct _inittab _PyGame_Inittab[] = {
    {"pygame_base", initbase},
    {"pygame_constants", initconstants},
    {"pygame_rect", initrect},
    {"pygame_rwobject", initrwobject},
    {"pygame_bufferproxy", initbufferproxy},
    {"pygame_color", initcolor},
    {"pygame_constants", initconstants},
    {"pygame_display", initdisplay},
    {"pygame_draw", initdraw},
    {"pygame_event", initevent},
    {"pygame_fastevent", initfastevent},
    {"pygame_font", initfont},
    {"pygame_gfxdraw", initgfxdraw},
    {"pygame_image", initimage},
    {"pygame_imageext", initimageext},
    {"pygame_key", initkey},
    {"pygame_mixer", initmixer},
//    {"movie", initmovie},
//    {"movieext", initmovieext},
    {"pygame_mixer_music", initmixer_music},
    {"pygame_overlay", initoverlay},
    {"pygame_pixelarray", initpixelarray},
    {"pygame_surface", initsurface},
    {"pygame_surflock", initsurflock},
    {"pygame_time", initpygame_time},
    {"pygame_transform", inittransform},
    {0, 0}
};
