/*
  pygame - Python Game Library

  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU Library General Public
  License as published by the Free Software Foundation; either
  version 2 of the License, or (at your option) any later version.

  This library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Library General Public License for more details.

  You should have received a copy of the GNU Library General Public
  License along with this library; if not, write to the Free
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

#include "../pygame.h"
#include "../pgcompat.h"
#include "../doc/sdl2_video_doc.h"
#include <structmember.h>

static PyTypeObject pgRenderer_Type;
static PyTypeObject pgTexture_Type;
static PyTypeObject pgWindow_Type;
static PyTypeObject pgImage_Type;

typedef struct pgRendererObject pgRendererObject;

#define pgRenderer_Check(x) (((PyObject*)(x))->ob_type == &pgRenderer_Type)
#define pgTexture_Check(x) (((PyObject*)(x))->ob_type == &pgTexture_Type)
#define pgWindow_Check(x) (((PyObject*)(x))->ob_type == &pgWindow_Type)
#define pgImage_Check(x) (((PyObject*)(x))->ob_type == &pgImage_Type)

static PyObject *drawfnc_str = NULL;

typedef struct {
    PyObject_HEAD
    SDL_Texture *texture;
    pgRendererObject *renderer;
    int width;
    int height;
    pgColorObject *color;
    Uint8 alpha;
} pgTextureObject;

typedef struct {
    PyObject_HEAD
    SDL_Window *_win;
    int _is_borrowed;
} pgWindowObject;

typedef struct {
    PyObject_HEAD
    pgTextureObject *texture;
    pgColorObject *color;
    float alpha;
    float originx;
    float originy;
    float angle;
    int flipX;
    int flipY;
    pgRectObject *srcrect;
} pgImageObject;

struct pgRendererObject {
    PyObject_HEAD
    pgWindowObject *window;
    SDL_Renderer *renderer;
    pgColorObject *drawcolor;
    pgTextureObject *target;
    int _is_borrowed;
};

/* forward declarations */
static int
pgTexture_DrawObj(pgTextureObject *self, PyObject *srcrect, PyObject *dstrect);

static int
pgTexture_Draw(pgTextureObject *self,
               SDL_Rect *srcrect, SDL_Rect *dstrect,
               float angle, const int * origin,
               int flipX, int flipY);

static pgTextureObject *
pg_texture_from_surface(PyObject *self, PyObject *args, PyObject *kw);

static PyObject *
pg_renderer_get_viewport(pgRendererObject *self, PyObject *args);

static PyObject *
pg_renderer_new(PyTypeObject *type, PyObject *args, PyObject *kwds);

static PyObject *
pg_window_new(PyTypeObject *type, PyObject *args, PyObject *kwds);

#include "video_renderer.c"
#include "video_texture.c"
#include "video_window.c"
#include "video_image.c"

static PyObject * 
pg_video_get_grabbed_window(PyObject *args) {
    // return the Window with input grab enabled,
    // or None if input isn't grabbed.
    SDL_Window *win = SDL_GetGrabbedWindow();
    void* ptr = NULL;

    if (win) {
        ptr = SDL_GetWindowData(win, "pg_window");
        if (!ptr) {
            Py_RETURN_NONE;
        }
        return (PyObject *)ptr;
    }
    Py_RETURN_NONE;
}

static PyMethodDef _pg_video_methods[] = {
    {"get_grabbed_window", pg_video_get_grabbed_window, METH_NOARGS, NULL /* TODO */},
    {NULL, NULL, 0, NULL}};

MODINIT_DEFINE(video)
{
    PyObject *module, *dict, *apiobj;
    int ecode;
    static void *c_api[PYGAMEAPI_VIDEO_NUMSLOTS];
#if PY3
    static struct PyModuleDef _module = {PyModuleDef_HEAD_INIT,
                                         "video",
                                         NULL /* TODO */,
                                         -1,
                                         _pg_video_methods,
                                         NULL,
                                         NULL,
                                         NULL,
                                         NULL};
#endif
    import_pygame_base();
    import_pygame_color();
    import_pygame_surface();
    import_pygame_rect();
    if (PyErr_Occurred()) {
        MODINIT_ERROR;
    }
    if (PyType_Ready(&pgRenderer_Type) < 0) {
        MODINIT_ERROR;
    }
    if (PyType_Ready(&pgTexture_Type) < 0) {
        MODINIT_ERROR;
    }
    if (PyType_Ready(&pgWindow_Type) < 0) {
        MODINIT_ERROR;
    }
    if (PyType_Ready(&pgImage_Type) < 0) {
        MODINIT_ERROR;
    }

#if PY3
    module = PyModule_Create(&_module);
#else
    module = Py_InitModule3(MODPREFIX "video", _pg_video_methods,
                            DOC_PYGAMESDL2VIDEO);
#endif
    if (module == NULL) {
        MODINIT_ERROR;
    }

    drawfnc_str = PyUnicode_FromString("draw");
    if (!drawfnc_str) {
        DECREF_MOD(module);
        MODINIT_ERROR;
    }
    /* TODO: cleanup for drawfnc_str */

    dict = PyModule_GetDict(module);

    if (PyDict_SetItemString(dict, "Renderer", (PyObject *)&pgRenderer_Type)) {
        DECREF_MOD(module);
        MODINIT_ERROR;
    }

    if (PyDict_SetItemString(dict, "Texture", (PyObject *)&pgTexture_Type)) {
        DECREF_MOD(module);
        MODINIT_ERROR;
    }

    if (PyDict_SetItemString(dict, "Window", (PyObject *)&pgWindow_Type)) {
        DECREF_MOD(module);
        MODINIT_ERROR;
    }

    if (PyDict_SetItemString(dict, "Image", (PyObject *)&pgImage_Type)) {
        DECREF_MOD(module);
        MODINIT_ERROR;
    }

    PyObject* centered = PyInt_FromLong(SDL_WINDOWPOS_CENTERED);
    if (PyDict_SetItemString(dict, "WINDOWPOS_CENTERED", centered)) {
        DECREF_MOD(module);
        MODINIT_ERROR;
    }

    PyObject* undefined = PyInt_FromLong(SDL_WINDOWPOS_UNDEFINED);
    if (PyDict_SetItemString(dict, "WINDOWPOS_UNDEFINED", undefined)) {
        DECREF_MOD(module);
        MODINIT_ERROR;
    }

    // export the c api
    c_api[0] = 42;
    apiobj = encapsulate_api(c_api, "video");
    if (apiobj == NULL) {
        DECREF_MOD(module);
        MODINIT_ERROR;
    }
    ecode = PyDict_SetItemString(dict, PYGAMEAPI_LOCAL_ENTRY, apiobj);
    Py_DECREF(apiobj);
    if (ecode) {
        DECREF_MOD(module);
        MODINIT_ERROR;
    }

    MODINIT_RETURN(module);
}
