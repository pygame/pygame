#define PYGAMEAPI_RENDER_INTERNAL

#include "pygame.h"

#include "pgcompat.h"

#include "doc/sdl2_video_doc.h"

static PyTypeObject pgRenderer_Type;

static PyTypeObject pgTexture_Type;

static PyTypeObject pgImage_Type;

#define pgRenderer_Check(x) \
    (PyObject_IsInstance((x), (PyObject *)&pgRenderer_Type))

#define pgTexture_Check(x) \
    (PyObject_IsInstance((x), (PyObject *)&pgTexture_Type))

#define pgImage_Check(x) (PyObject_IsInstance((x), (PyObject *)&pgImage_Type))

#define RENDERER_ERROR_CHECK(x)                       \
    if (x < 0) {                                      \
        return RAISE(pgExc_SDLError, SDL_GetError()); \
    }

#define RENDERER_PROPERTY_ERROR_CHECK(x)                 \
    if (x < 0) {                                         \
        RAISERETURN(pgExc_SDLError, SDL_GetError(), -1); \
    }

#define PARSE_POINT(obj, x, y, name)                                 \
    if (!pg_TwoFloatsFromObj(obj, &x, &y)) {                         \
        return RAISE(PyExc_TypeError, "invalid " #name " argument"); \
    }

#define PARSE_COLOR(obj, rgba)                                   \
    if (Py_IsNone(obj)) {                                        \
        rgba[0] = 255;                                           \
        rgba[1] = 255;                                           \
        rgba[2] = 255;                                           \
        rgba[3] = 255;                                           \
    }                                                            \
    else {                                                       \
        if (!pg_RGBAFromObjEx(obj, rgba, PG_COLOR_HANDLE_ALL)) { \
            return NULL;                                         \
        }                                                        \
    }

#define SET_VERTEX_COLOR(vertex, texture_mods, vertex_mods)     \
    vertex.color.r = (Uint8)(texture_mods[0] * vertex_mods[0]); \
    vertex.color.g = (Uint8)(texture_mods[1] * vertex_mods[1]); \
    vertex.color.b = (Uint8)(texture_mods[2] * vertex_mods[2]); \
    vertex.color.a = (Uint8)(texture_mods[3] * vertex_mods[3]);

static int
texture_renderer_draw(pgTextureObject *self, PyObject *area, PyObject *dest);

static int
image_renderer_draw(pgImageObject *self, PyObject *area, PyObject *dest);

/* Renderer implementation */
static PyObject *
renderer_from_window(PyTypeObject *cls, PyObject *args, PyObject *kwargs)
{
    PyObject *window;
    static char *keywords[] = {"window", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", keywords,
                                     &pgWindow_Type, &window)) {
        return NULL;
    }
    pgRendererObject *self =
        (pgRendererObject *)(cls->tp_new(cls, NULL, NULL));
    self->window = (pgWindowObject *)window;
    if (self->window->_is_borrowed) {
        self->_is_borrowed = SDL_TRUE;
    }
    else {
        return RAISE(pgExc_SDLError,
                     "Window is not created from display module");
    }
    self->renderer = SDL_GetRenderer(self->window->_win);
    if (!self->renderer) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }
    self->target = NULL;
    Py_INCREF(self);
    return (PyObject *)self;
}

static PyObject *
renderer_draw_point(pgRendererObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject *point;
    SDL_FPoint pos;
    static char *keywords[] = {"point", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", keywords, &point)) {
        return NULL;
    }
    if (!pg_TwoFloatsFromObj(point, &pos.x, &pos.y)) {
        return RAISE(PyExc_TypeError, "invalid argument");
    }
    RENDERER_ERROR_CHECK(SDL_RenderDrawPointF(self->renderer, pos.x, pos.y))
    Py_RETURN_NONE;
}

static PyObject *
renderer_draw_line(pgRendererObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject *start, *end;
    SDL_FPoint start_pos, end_pos;
    static char *keywords[] = {"p1", "p2", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", keywords, &start,
                                     &end)) {
        return NULL;
    }
    PARSE_POINT(start, start_pos.x, start_pos.y, "p1")
    PARSE_POINT(end, end_pos.x, end_pos.y, "p2")
    RENDERER_ERROR_CHECK(SDL_RenderDrawLineF(
        self->renderer, start_pos.x, start_pos.y, end_pos.x, end_pos.y))
    Py_RETURN_NONE;
}

static PyObject *
renderer_draw_rect(pgRendererObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject *rectobj;
    SDL_FRect *rect = NULL, temp;
    static char *keywords[] = {"rect", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", keywords, &rectobj)) {
        return NULL;
    }
    if (!(rect = pgFRect_FromObject(rectobj, &temp))) {
        return RAISE(PyExc_TypeError, "rect argument is invalid");
    }
    RENDERER_ERROR_CHECK(SDL_RenderDrawRectF(self->renderer, rect))
    Py_RETURN_NONE;
}

static PyObject *
renderer_fill_rect(pgRendererObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject *rectobj;
    SDL_FRect *rect = NULL, temp;
    static char *keywords[] = {"rect", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", keywords, &rectobj)) {
        return NULL;
    }
    if (!(rect = pgFRect_FromObject(rectobj, &temp))) {
        return RAISE(PyExc_TypeError, "rect argument is invalid");
    }
    RENDERER_ERROR_CHECK(SDL_RenderFillRectF(self->renderer, rect))
    Py_RETURN_NONE;
}

static PyObject *
renderer_draw_triangle(pgRendererObject *self, PyObject *args,
                       PyObject *kwargs)
{
    PyObject *p1, *p2, *p3;
    SDL_FPoint points[4];
    static char *keywords[] = {"p1", "p2", "p3", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO", keywords, &p1, &p2,
                                     &p3)) {
        return NULL;
    }
    PARSE_POINT(p1, points[0].x, points[0].y, "p1")
    PARSE_POINT(p2, points[1].x, points[1].y, "p2")
    PARSE_POINT(p3, points[2].x, points[2].y, "p3")
    points[3] = points[0];
    RENDERER_ERROR_CHECK(SDL_RenderDrawLinesF(self->renderer, points, 4))
    Py_RETURN_NONE;
}

static PyObject *
renderer_fill_triangle(pgRendererObject *self, PyObject *args,
                       PyObject *kwargs)
{
#if SDL_VERSION_ATLEAST(2, 0, 18)
    PyObject *p1, *p2, *p3;
    SDL_Vertex vertices[3];
    static char *keywords[] = {"p1", "p2", "p3", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO", keywords, &p1, &p2,
                                     &p3)) {
        return NULL;
    }
    PARSE_POINT(p1, vertices[0].position.x, vertices[0].position.y, "p1")
    PARSE_POINT(p2, vertices[1].position.x, vertices[1].position.y, "p2")
    PARSE_POINT(p3, vertices[2].position.x, vertices[2].position.y, "p3")
    RENDERER_ERROR_CHECK(SDL_GetRenderDrawColor(
        self->renderer, &vertices[0].color.r, &vertices[0].color.g,
        &vertices[0].color.b, &vertices[0].color.a))
    vertices[1].color = vertices[0].color;
    vertices[2].color = vertices[0].color;
    RENDERER_ERROR_CHECK(
        SDL_RenderGeometry(self->renderer, NULL, vertices, 3, NULL, 0))
    Py_RETURN_NONE;
#else
    RAISE(PyExc_TypeError, "fill_triangle() requires SDL 2.0.18 or newer");
    Py_RETURN_NONE;
#endif
}

static PyObject *
renderer_draw_quad(pgRendererObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject *p1, *p2, *p3, *p4;
    SDL_FPoint points[5];
    static char *keywords[] = {"p1", "p2", "p3", "p4", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOO", keywords, &p1, &p2,
                                     &p3, &p4)) {
        return NULL;
    }
    PARSE_POINT(p1, points[0].x, points[0].y, "p1")
    PARSE_POINT(p2, points[1].x, points[1].y, "p2")
    PARSE_POINT(p3, points[2].x, points[2].y, "p3")
    PARSE_POINT(p4, points[3].x, points[3].y, "p4")
    points[4] = points[0];
    RENDERER_ERROR_CHECK(SDL_RenderDrawLinesF(self->renderer, points, 5))
    Py_RETURN_NONE;
}

static PyObject *
renderer_fill_quad(pgRendererObject *self, PyObject *args, PyObject *kwargs)
{
#if SDL_VERSION_ATLEAST(2, 0, 18)
    PyObject *p1, *p2, *p3, *p4;
    SDL_Vertex vertices[4];
    static char *keywords[] = {"p1", "p2", "p3", "p4", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOO", keywords, &p1, &p2,
                                     &p3, &p4)) {
        return NULL;
    }
    PARSE_POINT(p1, vertices[0].position.x, vertices[0].position.y, "p1")
    PARSE_POINT(p2, vertices[1].position.x, vertices[1].position.y, "p2")
    PARSE_POINT(p3, vertices[2].position.x, vertices[2].position.y, "p3")
    PARSE_POINT(p4, vertices[3].position.x, vertices[3].position.y, "p4")
    RENDERER_ERROR_CHECK(SDL_GetRenderDrawColor(
        self->renderer, &vertices[0].color.r, &vertices[0].color.g,
        &vertices[0].color.b, &vertices[0].color.a))
    for (int i = 1; i < 4; i++) {
        vertices[i].color = vertices[0].color;
    }
    const int indices[] = {0, 1, 2, 2, 3, 0};
    RENDERER_ERROR_CHECK(
        SDL_RenderGeometry(self->renderer, NULL, vertices, 4, indices, 6))
    Py_RETURN_NONE;
#else
    RAISE(PyExc_TypeError, "fill_quad() requires SDL 2.0.18 or newer");
    Py_RETURN_NONE;
#endif
}

static PyObject *
renderer_present(pgRendererObject *self, PyObject *_null)
{
    SDL_RenderPresent(self->renderer);
    Py_RETURN_NONE;
}

static PyObject *
renderer_clear(pgRendererObject *self, PyObject *_null)
{
    RENDERER_ERROR_CHECK(SDL_RenderClear(self->renderer))
    Py_RETURN_NONE;
}

static PyObject *
renderer_get_viewport(pgRendererObject *self, PyObject *_null)
{
    SDL_Rect rect;
    SDL_RenderGetViewport(self->renderer, &rect);
    return pgRect_New(&rect);
}

static PyObject *
renderer_set_viewport(pgRendererObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject *rectobj;
    SDL_Rect *rect = NULL, temp;
    static char *keywords[] = {"area", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", keywords, &rectobj)) {
        return NULL;
    }
    if (rectobj == Py_None) {
        RENDERER_ERROR_CHECK(SDL_RenderSetViewport(self->renderer, NULL))
    }
    else {
        if (!(rect = pgRect_FromObject(rectobj, &temp))) {
            return RAISE(PyExc_TypeError, "area must be rectangle or None");
        }
        RENDERER_ERROR_CHECK(SDL_RenderSetViewport(self->renderer, rect))
    }
    Py_RETURN_NONE;
}

static PyObject *
renderer_compose_custom_blend_mode(PyObject *self, PyObject *args,
                                   PyObject *kwargs)
{
    int mode[6];
    SDL_BlendMode blend_mode;
    static char *keywords[] = {"color_mode", "alpha_mode", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "(iii)(iii)", keywords,
                                     &mode[0], &mode[1], &mode[2], &mode[3],
                                     &mode[4], &mode[5])) {
        return NULL;
    }
    blend_mode = SDL_ComposeCustomBlendMode(mode[0], mode[1], mode[2], mode[3],
                                            mode[4], mode[5]);
    return PyLong_FromLong((long)blend_mode);
}

static PyObject *
renderer_to_surface(pgRendererObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject *surfobj = Py_None, *rectobj = Py_None;
    SDL_Surface *surf;
    pgSurfaceObject *surface;
    SDL_Rect viewport, *areaparam, temp, *rect = &temp;
    Uint32 format;
    static char *keywords[] = {"surface", "area", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|OO", keywords, &surfobj,
                                     &rectobj)) {
        return NULL;
    }
    if (!Py_IsNone(rectobj)) {
        if (!(rect = pgRect_FromObject(rectobj, &temp))) {
            return RAISE(PyExc_TypeError, "area must be None or a rect");
        }
        SDL_RenderGetViewport(self->renderer, &viewport);
        SDL_IntersectRect(rect, &viewport, rect);
        areaparam = rect;
    }
    else {
        SDL_RenderGetViewport(self->renderer, rect);
        areaparam = NULL;
    }
    if (!Py_IsNone(surfobj)) {
        if (!(pgSurface_Check(surfobj))) {
            return RAISE(PyExc_TypeError, "surface must be None or a Surface");
        }
        surface = (pgSurfaceObject *)surfobj;
        Py_INCREF(surface);
        surf = surface->surf;
        if (surf->w < rect->w || surf->h < rect->h) {
            return RAISE(PyExc_ValueError, "the surface is too small");
        }
        format = surf->format->format;
    }
    else {
        format = SDL_GetWindowPixelFormat(self->window->_win);
        if (format == SDL_PIXELFORMAT_UNKNOWN) {
            return RAISE(pgExc_SDLError, SDL_GetError());
        }
        surf = SDL_CreateRGBSurfaceWithFormat(
            0, rect->w, rect->h, SDL_BITSPERPIXEL(format), format);
        if (surf == NULL) {
            return RAISE(pgExc_SDLError, SDL_GetError());
        }
        surface = pgSurface_New(surf);
    }
    RENDERER_ERROR_CHECK(SDL_RenderReadPixels(
        self->renderer, areaparam, format, surf->pixels, surf->pitch));
    return (PyObject *)surface;
}

static PyObject *
renderer_blit(pgRendererObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject *sourceobj, *destobj = Py_None, *areaobj = Py_None;
    int special_flags = 0;
    static char *keywords[] = {"source", "dest", "area", "special_flags",
                               NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|OOi", keywords,
                                     &sourceobj, &destobj, &areaobj,
                                     &special_flags)) {
        return NULL;
    }

    if (pgTexture_Check(sourceobj)) {
        if (!texture_renderer_draw((pgTextureObject *)sourceobj, areaobj,
                                   destobj)) {
            return NULL;
        }
    }
    else if (pgImage_Check(sourceobj)) {
        if (!image_renderer_draw((pgImageObject *)sourceobj, areaobj,
                                 destobj)) {
            return NULL;
        }
    }
    else {
        if (!PyObject_CallFunctionObjArgs(
                PyObject_GetAttrString(sourceobj, "draw"), areaobj, destobj,
                NULL)) {
            return NULL;
        }
    }

    if (Py_IsNone(destobj)) {
        return renderer_get_viewport(self, NULL);
    }
    Py_INCREF(destobj);
    return destobj;
}

static PyObject *
renderer_get_draw_color(pgRendererObject *self, void *closure)
{
    Uint8 rgba[4];
    RENDERER_ERROR_CHECK(SDL_GetRenderDrawColor(self->renderer, &rgba[0],
                                                &rgba[1], &rgba[2], &rgba[3]))
    return pgColor_NewLength(rgba, 4);
}

static int
renderer_set_draw_color(pgRendererObject *self, PyObject *arg, void *closure)
{
    Uint8 color[4];
    if (!pg_RGBAFromObjEx(arg, color, PG_COLOR_HANDLE_ALL)) {
        return -1;
    }
    RENDERER_PROPERTY_ERROR_CHECK(SDL_SetRenderDrawColor(
        self->renderer, color[0], color[1], color[2], color[3]))
    return 0;
}

static PyObject *
renderer_get_draw_blend_mode(pgRendererObject *self, void *closure)
{
    SDL_BlendMode blend_mode;
    RENDERER_ERROR_CHECK(
        SDL_GetRenderDrawBlendMode(self->renderer, &blend_mode))
    return PyLong_FromLong((long)blend_mode);
}

static int
renderer_set_draw_blend_mode(pgRendererObject *self, PyObject *arg,
                             void *closure)
{
    if (!PyLong_Check(arg)) {
        RAISERETURN(PyExc_TypeError, "Draw blend mode must be int", -1);
    }
    RENDERER_PROPERTY_ERROR_CHECK(
        SDL_SetRenderDrawBlendMode(self->renderer, (int)PyLong_AsLong(arg)))
    return 0;
}

static PyObject *
renderer_get_logical_size(pgRendererObject *self, void *closure)
{
    int w, h;
    SDL_RenderGetLogicalSize(self->renderer, &w, &h);
    return pg_tuple_couple_from_values_int(w, h);
}

static int
renderer_set_logical_size(pgRendererObject *self, PyObject *arg, void *closure)
{
    int w, h;
    if (!pg_TwoIntsFromObj(arg, &w, &h)) {
        RAISERETURN(PyExc_TypeError, "invalid logical size", -1);
    }
    RENDERER_PROPERTY_ERROR_CHECK(
        SDL_RenderSetLogicalSize(self->renderer, w, h))
    return 0;
}

static PyObject *
renderer_coordinates_to_window(pgRendererObject *self, PyObject *args,
                               PyObject *kwargs)
{
    float lx, ly;
    PyObject *point;

    static char *keywords[] = {"point", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", keywords, &point)) {
        return NULL;
    }
    if (!pg_TwoFloatsFromObj(point, &lx, &ly)) {
        return RAISE(PyExc_TypeError,
                     "point must be a sequence of two numbers");
    }

#if SDL_VERSION_ATLEAST(3, 0, 0)
    float wx, wy;
    SDL_RenderCoordinatesToWindow(self->renderer, lx, ly, &wx, &wy);

    return pg_tuple_couple_from_values_double(wx, wy);
#else
    int wx, wy;
    SDL_RenderLogicalToWindow(self->renderer, lx, ly, &wx, &wy);

    return pg_tuple_couple_from_values_double((float)wx, (float)wy);
#endif
}

static PyObject *
renderer_coordinates_from_window(pgRendererObject *self, PyObject *args,
                                 PyObject *kwargs)
{
    float lx, ly;
    float wx, wy;
    PyObject *point;

    static char *keywords[] = {"point", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", keywords, &point)) {
        return NULL;
    }
    if (!pg_TwoFloatsFromObj(point, &wx, &wy)) {
        return RAISE(PyExc_TypeError,
                     "point must be a sequence of two numbers");
    }

#if SDL_VERSION_ATLEAST(3, 0, 0)
    SDL_RenderCoordinatesFromWindow(self->renderer, wx, wy, &lx, &ly);
#else
    SDL_RenderWindowToLogical(self->renderer, (int)wx, (int)wy, &lx, &ly);
#endif

    return pg_tuple_couple_from_values_double(lx, ly);
}

static PyObject *
renderer_get_scale(pgRendererObject *self, void *closure)
{
    float x, y;
    SDL_RenderGetScale(self->renderer, &x, &y);
    return pg_tuple_couple_from_values_double(x, y);
}

static int
renderer_set_scale(pgRendererObject *self, PyObject *arg, void *closure)
{
    float x, y;
    if (!pg_TwoFloatsFromObj(arg, &x, &y)) {
        RAISERETURN(PyExc_TypeError, "invalid scale", -1);
    }
    RENDERER_PROPERTY_ERROR_CHECK(SDL_RenderSetScale(self->renderer, x, y))
    return 0;
}

static PyObject *
renderer_get_target(pgRendererObject *self, void *closure)
{
    if (self->target == NULL) {
        Py_RETURN_NONE;
    }
    Py_INCREF(self->target);
    return (PyObject *)self->target;
}

static int
renderer_set_target(pgRendererObject *self, PyObject *arg, void *closure)
{
    Py_XDECREF(self->target);
    if (Py_IsNone(arg)) {
        self->target = NULL;
        RENDERER_PROPERTY_ERROR_CHECK(
            SDL_SetRenderTarget(self->renderer, NULL))
        return 0;
    }
    else if (pgTexture_Check(arg)) {
        self->target = (pgTextureObject *)arg;
        Py_XINCREF(self->target);
        RENDERER_PROPERTY_ERROR_CHECK(
            SDL_SetRenderTarget(self->renderer, self->target->texture))
        return 0;
    }
    else {
        RAISERETURN(PyExc_TypeError, "target must be Texture object or None",
                    -1);
    }
}

static int
renderer_init(pgRendererObject *self, PyObject *args, PyObject *kwargs)
{
    SDL_Renderer *renderer = NULL;
    pgWindowObject *window;
    int index = -1;
    int accelerated = -1;
    int vsync = 0;
    int target_texture = 0;
    Uint32 flags = 0;

    char *keywords[] = {"window", "index",          "accelerated",
                        "vsync",  "target_texture", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!|iipp", keywords,
                                     &pgWindow_Type, &window, &index,
                                     &accelerated, &vsync, &target_texture)) {
        return -1;
    }
    if (accelerated >= 0) {
        flags |=
            accelerated ? SDL_RENDERER_ACCELERATED : SDL_RENDERER_SOFTWARE;
    }
    if (vsync) {
        flags |= SDL_RENDERER_PRESENTVSYNC;
    }
    if (target_texture) {
        flags |= SDL_RENDERER_TARGETTEXTURE;
    }
    renderer = SDL_CreateRenderer(window->_win, index, flags);
    if (!renderer) {
        PyErr_SetString(pgExc_SDLError, SDL_GetError());
        return -1;
    }
    self->renderer = renderer;
    self->window = window;
    self->target = NULL;
    self->_is_borrowed = SDL_FALSE;
    return 0;
}

static void
renderer_dealloc(pgRendererObject *self, PyObject *_null)
{
    if (!self->_is_borrowed && self->renderer) {
        SDL_DestroyRenderer(self->renderer);
    }
    Py_TYPE(self)->tp_free(self);
}

/* Texture implementation */
static int
texture_renderer_draw(pgTextureObject *self, PyObject *area, PyObject *dest)
{
    SDL_Rect srcrect, *srcrectptr = NULL;
    SDL_FRect dstrect, *dstrectptr = NULL;
    if (!Py_IsNone(area)) {
        if (!(srcrectptr = pgRect_FromObject(area, &srcrect))) {
            PyErr_SetString(PyExc_ValueError,
                            "srcrect must be a Rect or None");
            return 0;
        }
    }
    if (!Py_IsNone(dest)) {
        if (!(dstrectptr = pgFRect_FromObject(dest, &dstrect))) {
            if (!pg_TwoFloatsFromObj(dest, &dstrect.x, &dstrect.y)) {
                PyErr_SetString(PyExc_ValueError,
                                "dstrect must be a point, Rect, or None");
                return 0;
            }
            dstrect.w = (float)self->width;
            dstrect.h = (float)self->height;
        }
    }
    if (SDL_RenderCopyExF(self->renderer->renderer, self->texture, srcrectptr,
                          dstrectptr, 0, NULL, SDL_FLIP_NONE) < 0) {
        PyErr_SetString(pgExc_SDLError, SDL_GetError());
        return 0;
    }
    return 1;
}

static PyObject *
texture_from_surface(PyObject *self, PyObject *args, PyObject *kwargs)
{
    pgRendererObject *renderer;
    pgSurfaceObject *surfobj;
    pgTextureObject *new_texture;
    SDL_Texture *texture;
    SDL_Surface *surf = NULL;
    static char *keywords[] = {"renderer", "surface", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO!", keywords, &renderer,
                                     &pgSurface_Type, &surfobj)) {
        return NULL; /* Exception already set. */
    }
    surf = pgSurface_AsSurface(surfobj);
    SURF_INIT_CHECK(surf)
    texture = SDL_CreateTextureFromSurface(renderer->renderer, surf);
    if (!texture) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }
    new_texture =
        (pgTextureObject *)(&(pgTexture_Type))->tp_alloc(&pgTexture_Type, 0);
    new_texture->texture = texture;
    new_texture->renderer = renderer;
    Py_XINCREF(new_texture->renderer);
    new_texture->width = surf->w;
    new_texture->height = surf->h;
    new_texture->weakreflist = NULL;
    return (PyObject *)new_texture;
}

static PyObject *
texture_get_rect(pgTextureObject *self, PyObject *const *args,
                 Py_ssize_t nargs, PyObject *kwargs)
{
    PyObject *rect = pgRect_New4(0, 0, self->width, self->height);
    return pgObject_getRectHelper(rect, args, nargs, kwargs, "rect");
}

static PyObject *
texture_draw(pgTextureObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject *srcrectobj = Py_None, *dstrectobj = Py_None,
             *originobj = Py_None;
    SDL_Rect srcrect, *srcrectptr = NULL;
    SDL_FRect dstrect, *dstrectptr = NULL;
    SDL_FPoint origin, *originptr = NULL;
    double angle = 0;
    int flip_x = 0;
    int flip_y = 0;
    SDL_RendererFlip flip = SDL_FLIP_NONE;
    static char *keywords[] = {"srcrect", "dstrect", "angle", "origin",
                               "flip_x",  "flip_y",  NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|OOdOpp", keywords,
                                     &srcrectobj, &dstrectobj, &angle,
                                     &originobj, &flip_x, &flip_y)) {
        return NULL; /* Exception already set. */
    }
    if (!Py_IsNone(srcrectobj)) {
        if (!(srcrectptr = pgRect_FromObject(srcrectobj, &srcrect))) {
            return RAISE(PyExc_ValueError, "srcrect must be a Rect or None");
        }
    }
    if (!Py_IsNone(dstrectobj)) {
        if (!(dstrectptr = pgFRect_FromObject(dstrectobj, &dstrect))) {
            if (!pg_TwoFloatsFromObj(dstrectobj, &dstrect.x, &dstrect.y)) {
                return RAISE(PyExc_ValueError,
                             "dstrect must be a point, Rect, or None");
            }
            dstrect.w = (float)self->width;
            dstrect.h = (float)self->height;
        }
    }
    if (!Py_IsNone(originobj)) {
        if (!pg_TwoFloatsFromObj(originobj, &origin.x, &origin.y)) {
            return RAISE(PyExc_ValueError, "origin must be a point or None");
        }
        originptr = &origin;
    }
    if (flip_x) {
        flip |= SDL_FLIP_HORIZONTAL;
    }
    if (flip_y) {
        flip |= SDL_FLIP_VERTICAL;
    }
    RENDERER_ERROR_CHECK(SDL_RenderCopyExF(self->renderer->renderer,
                                           self->texture, srcrectptr,
                                           dstrectptr, angle, originptr, flip))
    Py_RETURN_NONE;
}

static PyObject *
texture_draw_triangle(pgTextureObject *self, PyObject *args, PyObject *kwargs)
{
#if SDL_VERSION_ATLEAST(2, 0, 18)
    PyObject *p1_xyobj, *p2_xyobj, *p3_xyobj,
        *p1_uvobj = Py_None, *p2_uvobj = Py_None, *p3_uvobj = Py_None,
        *p1_modobj = Py_None, *p2_modobj = Py_None, *p3_modobj = Py_None;
    Uint8 texture_mods[4];
    float mods[4];
    Uint8 p1_mod[4], p2_mod[4], p3_mod[4];
    SDL_Vertex vertices[3];
    static char *keywords[] = {"p1_xy", "p2_xy",  "p3_xy",  "p1_uv",  "p2_uv",
                               "p3_uv", "p1_mod", "p2_mod", "p3_mod", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO|OOOOOO", keywords,
                                     &p1_xyobj, &p2_xyobj, &p3_xyobj,
                                     &p1_uvobj, &p2_uvobj, &p3_uvobj,
                                     &p1_modobj, &p2_modobj, &p3_modobj)) {
        return NULL; /* Exception already set. */
    }
    /* Parse position */
    PARSE_POINT(p1_xyobj, vertices[0].position.x, vertices[0].position.y,
                "p1_xy")
    PARSE_POINT(p2_xyobj, vertices[1].position.x, vertices[1].position.y,
                "p2_xy")
    PARSE_POINT(p3_xyobj, vertices[2].position.x, vertices[2].position.y,
                "p3_xy")
    /* Parse tex_coord */
    if (Py_IsNone(p1_uvobj)) {
        vertices[0].tex_coord.x = 0.0;
        vertices[0].tex_coord.y = 0.0;
    }
    else {
        PARSE_POINT(p1_uvobj, vertices[0].tex_coord.x, vertices[0].tex_coord.y,
                    "p1_uv")
    }
    if (Py_IsNone(p2_uvobj)) {
        vertices[1].tex_coord.x = 1.0;
        vertices[1].tex_coord.y = 1.0;
    }
    else {
        PARSE_POINT(p2_uvobj, vertices[1].tex_coord.x, vertices[1].tex_coord.y,
                    "p2_uv")
    }
    if (Py_IsNone(p3_uvobj)) {
        vertices[2].tex_coord.x = 0.0;
        vertices[2].tex_coord.y = 1.0;
    }
    else {
        PARSE_POINT(p3_uvobj, vertices[2].tex_coord.x, vertices[2].tex_coord.y,
                    "p3_uv")
    }
    /* Parse color */
    PARSE_COLOR(p1_modobj, p1_mod)
    PARSE_COLOR(p2_modobj, p2_mod)
    PARSE_COLOR(p3_modobj, p3_mod)
    RENDERER_ERROR_CHECK(SDL_GetTextureColorMod(
        self->texture, &texture_mods[0], &texture_mods[1], &texture_mods[2]));
    RENDERER_ERROR_CHECK(
        SDL_GetTextureAlphaMod(self->texture, &texture_mods[3]));
    mods[0] = texture_mods[0] / (float)255.0;
    mods[1] = texture_mods[1] / (float)255.0;
    mods[2] = texture_mods[2] / (float)255.0;
    mods[3] = texture_mods[3] / (float)255.0;
    SET_VERTEX_COLOR(vertices[0], mods, p1_mod);
    SET_VERTEX_COLOR(vertices[1], mods, p2_mod);
    SET_VERTEX_COLOR(vertices[2], mods, p3_mod);
    RENDERER_ERROR_CHECK(SDL_RenderGeometry(
        self->renderer->renderer, self->texture, vertices, 3, NULL, 0))
    Py_RETURN_NONE;
#else
    RAISE(PyExc_TypeError, "draw_triangle() requires SDL 2.0.18 or newer");
    Py_RETURN_NONE;
#endif
}

static PyObject *
texture_draw_quad(pgTextureObject *self, PyObject *args, PyObject *kwargs)
{
#if SDL_VERSION_ATLEAST(2, 0, 18)
    PyObject *p1_xyobj, *p2_xyobj, *p3_xyobj, *p4_xyobj,
        *p1_uvobj = Py_None, *p2_uvobj = Py_None, *p3_uvobj = Py_None,
        *p4_uvobj = Py_None, *p1_modobj = Py_None, *p2_modobj = Py_None,
        *p3_modobj = Py_None, *p4_modobj = Py_None;
    Uint8 texture_mods[4];
    float mods[4];
    Uint8 p1_mod[4], p2_mod[4], p3_mod[4], p4_mod[4];
    SDL_Vertex vertices[6];
    static char *keywords[] = {"p1_xy",  "p2_xy",  "p3_xy", "p4_xy",  "p1_uv",
                               "p2_uv",  "p3_uv",  "p4_uv", "p1_mod", "p2_mod",
                               "p3_mod", "p4_mod", NULL};
    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OOOO|OOOOOOOO", keywords, &p1_xyobj, &p2_xyobj,
            &p3_xyobj, &p4_xyobj, &p1_uvobj, &p2_uvobj, &p3_uvobj, &p4_uvobj,
            &p1_modobj, &p2_modobj, &p3_modobj, &p4_modobj)) {
        return NULL; /* Exception already set. */
    }
    /* Parse position */
    PARSE_POINT(p1_xyobj, vertices[0].position.x, vertices[0].position.y,
                "p1_xy")
    PARSE_POINT(p2_xyobj, vertices[1].position.x, vertices[1].position.y,
                "p2_xy")
    PARSE_POINT(p3_xyobj, vertices[2].position.x, vertices[2].position.y,
                "p3_xy")
    vertices[3].position = vertices[2].position;
    PARSE_POINT(p4_xyobj, vertices[4].position.x, vertices[4].position.y,
                "p4_xy")
    vertices[5].position = vertices[0].position;
    /* Parse tex_coord */
    if (Py_IsNone(p1_uvobj)) {
        vertices[0].tex_coord.x = 0.0;
        vertices[0].tex_coord.y = 0.0;
    }
    else {
        PARSE_POINT(p1_uvobj, vertices[0].tex_coord.x, vertices[0].tex_coord.y,
                    "p1_uv")
    }
    if (Py_IsNone(p2_uvobj)) {
        vertices[1].tex_coord.x = 1.0;
        vertices[1].tex_coord.y = 0.0;
    }
    else {
        PARSE_POINT(p2_uvobj, vertices[1].tex_coord.x, vertices[1].tex_coord.y,
                    "p2_uv")
    }
    if (Py_IsNone(p3_uvobj)) {
        vertices[2].tex_coord.x = 1.0;
        vertices[2].tex_coord.y = 1.0;
    }
    else {
        PARSE_POINT(p3_uvobj, vertices[2].tex_coord.x, vertices[2].tex_coord.y,
                    "p3_uv")
    }
    vertices[3].tex_coord = vertices[2].tex_coord;
    if (Py_IsNone(p4_uvobj)) {
        vertices[4].tex_coord.x = 0.0;
        vertices[4].tex_coord.y = 1.0;
    }
    else {
        PARSE_POINT(p4_uvobj, vertices[4].tex_coord.x, vertices[4].tex_coord.y,
                    "p4_uv")
    }
    vertices[5].tex_coord = vertices[0].tex_coord;
    /* Parse color */
    PARSE_COLOR(p1_modobj, p1_mod)
    PARSE_COLOR(p2_modobj, p2_mod)
    PARSE_COLOR(p3_modobj, p3_mod)
    PARSE_COLOR(p4_modobj, p4_mod)
    RENDERER_ERROR_CHECK(SDL_GetTextureColorMod(
        self->texture, &texture_mods[0], &texture_mods[1], &texture_mods[2]));
    RENDERER_ERROR_CHECK(
        SDL_GetTextureAlphaMod(self->texture, &texture_mods[3]));
    mods[0] = texture_mods[0] / (float)255.0;
    mods[1] = texture_mods[1] / (float)255.0;
    mods[2] = texture_mods[2] / (float)255.0;
    mods[3] = texture_mods[3] / (float)255.0;
    SET_VERTEX_COLOR(vertices[0], mods, p1_mod);
    SET_VERTEX_COLOR(vertices[1], mods, p2_mod);
    SET_VERTEX_COLOR(vertices[2], mods, p3_mod);
    vertices[3].color = vertices[2].color;
    SET_VERTEX_COLOR(vertices[4], mods, p4_mod);
    vertices[5].color = vertices[0].color;
    RENDERER_ERROR_CHECK(SDL_RenderGeometry(
        self->renderer->renderer, self->texture, vertices, 6, NULL, 0))
    Py_RETURN_NONE;
#else
    RAISE(PyExc_TypeError, "draw_triangle() requires SDL 2.0.18 or newer");
    Py_RETURN_NONE;
#endif
}

static PyObject *
texture_update(pgTextureObject *self, PyObject *args, PyObject *kwargs)
{
    pgSurfaceObject *surfobj;
    PyObject *rectobj = Py_None;
    SDL_Surface *surf = NULL;
    SDL_Rect area, *areaptr = NULL;
    SDL_Surface *converted_surf = NULL;
    SDL_PixelFormat *pixel_format = NULL;
    SDL_BlendMode blend;
    Uint32 format;
    int res;
    int dst_width, dst_height;
    static char *keywords[] = {"surface", "area", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!|O", keywords,
                                     &pgSurface_Type, &surfobj, &rectobj)) {
        return NULL; /* Exception already set. */
    }
    surf = pgSurface_AsSurface(surfobj);
    SURF_INIT_CHECK(surf)
    area.x = 0;
    area.y = 0;
    if (!Py_IsNone(rectobj)) {
        if (!(areaptr = pgRect_FromObject(rectobj, &area))) {
            return RAISE(PyExc_ValueError, "area must be a rectangle or None");
        }
    }
    if (areaptr == NULL) {
        dst_width = self->width;
        dst_height = self->height;
    }
    else {
        dst_width = area.w;
        dst_height = area.h;
    }
    if (dst_width > surf->w || dst_height > surf->h) {
        areaptr = &area;
        area.w = surf->w;
        area.h = surf->h;
    }
    RENDERER_ERROR_CHECK(
        SDL_QueryTexture(self->texture, &format, NULL, NULL, NULL))
    if (format != surf->format->format) {
        RENDERER_ERROR_CHECK(SDL_GetSurfaceBlendMode(surf, &blend))
        pixel_format = SDL_AllocFormat(format);
        if (pixel_format == NULL) {
            return RAISE(pgExc_SDLError, SDL_GetError());
        }
        converted_surf = SDL_ConvertSurface(surf, pixel_format, 0);
        if (SDL_SetSurfaceBlendMode(converted_surf, blend) < 0) {
            SDL_FreeSurface(converted_surf);
            SDL_FreeFormat(pixel_format);
            return RAISE(pgExc_SDLError, SDL_GetError());
        }

        res = SDL_UpdateTexture(self->texture, areaptr, converted_surf->pixels,
                                converted_surf->pitch);
        SDL_FreeSurface(converted_surf);
        SDL_FreeFormat(pixel_format);
    }
    else {
        res = SDL_UpdateTexture(self->texture, areaptr, surf->pixels,
                                surf->pitch);
    }
    if (res < 0) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }
    Py_RETURN_NONE;
}

static PyObject *
texture_get_renderer(pgTextureObject *self, void *closure)
{
    Py_INCREF(self->renderer);
    return (PyObject *)self->renderer;
}

static PyObject *
texture_get_width(pgTextureObject *self, void *closure)
{
    return PyLong_FromLong(self->width);
}

static PyObject *
texture_get_height(pgTextureObject *self, void *closure)
{
    return PyLong_FromLong(self->height);
}

static PyObject *
texture_get_alpha(pgTextureObject *self, void *closure)
{
    Uint8 alpha;
    RENDERER_ERROR_CHECK(SDL_GetTextureAlphaMod(self->texture, &alpha));
    return PyLong_FromLong(alpha);
}

static int
texture_set_alpha(pgTextureObject *self, PyObject *arg, void *closure)
{
    if (PyLong_Check(arg)) {
        unsigned long longval = PyLong_AsUnsignedLong(arg);
        RENDERER_PROPERTY_ERROR_CHECK(
            SDL_SetTextureAlphaMod(self->texture, (Uint8)longval))
        return 0;
    }
    return -1;
}

static PyObject *
texture_get_blend_mode(pgTextureObject *self, void *closure)
{
    SDL_BlendMode blend_mode;
    RENDERER_ERROR_CHECK(SDL_GetTextureBlendMode(self->texture, &blend_mode));
    return PyLong_FromLong((long)blend_mode);
}

static int
texture_set_blend_mode(pgTextureObject *self, PyObject *arg, void *closure)
{
    long longval = PyLong_AsLong(arg);
    if (longval == -1 && PyErr_Occurred()) {
        return -1;
    }
    RENDERER_PROPERTY_ERROR_CHECK(
        SDL_SetTextureBlendMode(self->texture, (int)longval))
    return 0;
}

static PyObject *
texture_get_color(pgTextureObject *self, void *closure)
{
    Uint8 color[4];
    RENDERER_ERROR_CHECK(SDL_GetTextureColorMod(self->texture, &color[0],
                                                &color[1], &color[2]));
    color[3] = 255;
    return pgColor_NewLength(color, 4);
}

static int
texture_set_color(pgTextureObject *self, PyObject *arg, void *closure)
{
    Uint8 color[4];
    if (!pg_RGBAFromObjEx(arg, color, PG_COLOR_HANDLE_ALL)) {
        return -1;
    }
    RENDERER_PROPERTY_ERROR_CHECK(
        SDL_SetTextureColorMod(self->texture, color[0], color[1], color[2]))
    return 0;
}

static int
texture_init(pgTextureObject *self, PyObject *args, PyObject *kwargs)
{
    pgRendererObject *renderer;
    PyObject *sizeobj;
    int width;
    int height;
    int depth = 0;
    int staticc = 0;
    int streaming = 0;
    int target = 0;
    int scale_quality = -1;
    int access = SDL_TEXTUREACCESS_STATIC;
    Uint32 Rmask, Gmask, Bmask, Amask;
    Uint32 format;

    char *keywords[] = {"renderer",  "size",   "depth",         "static",
                        "streaming", "target", "scale_quality", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|ipppi", keywords,
                                     &renderer, &sizeobj, &depth, &staticc,
                                     &streaming, &target, &scale_quality)) {
        return -1;
    }
    if (depth == 0 || depth == 32) {
        Rmask = 0xFF << 16;
        Gmask = 0xFF << 8;
        Bmask = 0xFF;
        Amask = 0xFF << 24;
    }
    else if (depth == 16) {
        Rmask = 0xF << 8;
        Gmask = 0xF << 4;
        Bmask = 0xF;
        Amask = 0xF << 12;
    }
    else {
        RAISERETURN(PyExc_ValueError,
                    "no standard masks exist for given bitdepth with alpha",
                    -1)
    }
    format = SDL_MasksToPixelFormatEnum(depth, Rmask, Gmask, Bmask, Amask);
    if (!pg_TwoIntsFromObj(sizeobj, &width, &height)) {
        RAISERETURN(PyExc_TypeError, "invalid size argument", -1)
    }
    if (width <= 0 || height <= 0) {
        RAISERETURN(PyExc_ValueError, "size must contain two positive values",
                    -1)
    }
    if (streaming) {
        if (staticc || target) {
            RAISERETURN(PyExc_ValueError,
                        "only one of static, streaming, or target can be true",
                        -1)
        }
        access = SDL_TEXTUREACCESS_STREAMING;
    }
    else if (target) {
        if (staticc) {
            RAISERETURN(PyExc_ValueError,
                        "only one of static, streaming, or target can be true",
                        -1)
        }
        access = SDL_TEXTUREACCESS_TARGET;
    }
    else {
        access = SDL_TEXTUREACCESS_STATIC;
    }
    self->renderer = renderer;
    Py_XINCREF(self->renderer);
    self->texture =
        SDL_CreateTexture(renderer->renderer, format, access, width, height);
    if (!self->texture) {
        RAISERETURN(pgExc_SDLError, SDL_GetError(), -1)
    }
    if (scale_quality != -1) {
#if SDL_VERSION_ATLEAST(2, 0, 12)
        RENDERER_PROPERTY_ERROR_CHECK(
            SDL_SetTextureScaleMode(self->texture, scale_quality))
#else
        switch (scale_quality) {
            case 0:
                SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "nearest");
                break;
            case 1:
                SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "linear");
                break;
            case 2:
                SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "best");
                break;
        }
#endif
    }
    self->width = width;
    self->height = height;
    return 0;
}

static void
texture_dealloc(pgTextureObject *self, PyObject *_null)
{
    if (self->texture) {
        SDL_DestroyTexture(self->texture);
    }
    if (self->weakreflist) {
        PyObject_ClearWeakRefs((PyObject *)self);
    }
    Py_TYPE(self)->tp_free(self);
}

/* Image implementation */
static int
image_renderer_draw(pgImageObject *self, PyObject *area, PyObject *dest)
{
    return 1;
}

/* Module definition */
static PyMethodDef renderer_methods[] = {
    {"draw_point", (PyCFunction)renderer_draw_point,
     METH_VARARGS | METH_KEYWORDS, DOC_SDL2_VIDEO_RENDERER_DRAWPOINT},
    {"draw_line", (PyCFunction)renderer_draw_line,
     METH_VARARGS | METH_KEYWORDS, DOC_SDL2_VIDEO_RENDERER_DRAWLINE},
    {"draw_rect", (PyCFunction)renderer_draw_rect,
     METH_VARARGS | METH_KEYWORDS, DOC_SDL2_VIDEO_RENDERER_DRAWRECT},
    {"draw_triangle", (PyCFunction)renderer_draw_triangle,
     METH_VARARGS | METH_KEYWORDS, DOC_SDL2_VIDEO_RENDERER_DRAWTRIANGLE},
    {"draw_quad", (PyCFunction)renderer_draw_quad,
     METH_VARARGS | METH_KEYWORDS, DOC_SDL2_VIDEO_RENDERER_DRAWQUAD},
    {"fill_rect", (PyCFunction)renderer_fill_rect,
     METH_VARARGS | METH_KEYWORDS, DOC_SDL2_VIDEO_RENDERER_FILLRECT},
    {"fill_triangle", (PyCFunction)renderer_fill_triangle,
     METH_VARARGS | METH_KEYWORDS, DOC_SDL2_VIDEO_RENDERER_FILLTRIANGLE},
    {"fill_quad", (PyCFunction)renderer_fill_quad,
     METH_VARARGS | METH_KEYWORDS, DOC_SDL2_VIDEO_RENDERER_FILLQUAD},
    {"present", (PyCFunction)renderer_present, METH_NOARGS,
     DOC_SDL2_VIDEO_RENDERER_PRESENT},
    {"clear", (PyCFunction)renderer_clear, METH_NOARGS,
     DOC_SDL2_VIDEO_RENDERER_CLEAR},
    {"set_viewport", (PyCFunction)renderer_set_viewport,
     METH_VARARGS | METH_KEYWORDS, DOC_SDL2_VIDEO_RENDERER_SETVIEWPORT},
    {"get_viewport", (PyCFunction)renderer_get_viewport, METH_NOARGS,
     DOC_SDL2_VIDEO_RENDERER_GETVIEWPORT},
    {"coordinates_to_window", (PyCFunction)renderer_coordinates_to_window,
     METH_VARARGS | METH_KEYWORDS,
     DOC_SDL2_VIDEO_RENDERER_COORDINATESTOWINDOW},
    {"coordinates_from_window", (PyCFunction)renderer_coordinates_from_window,
     METH_VARARGS | METH_KEYWORDS,
     DOC_SDL2_VIDEO_RENDERER_COORDINATESFROMWINDOW},
    {"compose_custom_blend_mode",
     (PyCFunction)renderer_compose_custom_blend_mode,
     METH_VARARGS | METH_KEYWORDS | METH_CLASS,
     DOC_SDL2_VIDEO_RENDERER_COMPOSECUSTOMBLENDMODE},
    {"from_window", (PyCFunction)renderer_from_window,
     METH_CLASS | METH_VARARGS | METH_KEYWORDS,
     DOC_SDL2_VIDEO_GETGRABBEDWINDOW},
    {"to_surface", (PyCFunction)renderer_to_surface,
     METH_VARARGS | METH_KEYWORDS, DOC_SDL2_VIDEO_RENDERER_TOSURFACE},
    {"blit", (PyCFunction)renderer_blit, METH_VARARGS | METH_KEYWORDS,
     DOC_SDL2_VIDEO_RENDERER_SETVIEWPORT},
    {NULL, NULL, 0, NULL}};

static PyGetSetDef renderer_getset[] = {
    {"draw_color", (getter)renderer_get_draw_color,
     (setter)renderer_set_draw_color, DOC_SDL2_VIDEO_RENDERER_DRAWCOLOR, NULL},
    {"draw_blend_mode", (getter)renderer_get_draw_blend_mode,
     (setter)renderer_set_draw_blend_mode, DOC_SDL2_VIDEO_RENDERER_DRAWCOLOR,
     NULL},
    {"logical_size", (getter)renderer_get_logical_size,
     (setter)renderer_set_logical_size, DOC_SDL2_VIDEO_RENDERER_LOGICALSIZE,
     NULL},
    {"scale", (getter)renderer_get_scale, (setter)renderer_set_scale,
     DOC_SDL2_VIDEO_RENDERER_SCALE, NULL},
    {"target", (getter)renderer_get_target, (setter)renderer_set_target,
     DOC_SDL2_VIDEO_RENDERER_TARGET, NULL},
    {NULL, 0, NULL, NULL, NULL}};

static PyMethodDef texture_methods[] = {
    {"get_rect", (PyCFunction)texture_get_rect, METH_FASTCALL | METH_KEYWORDS,
     DOC_SDL2_VIDEO_TEXTURE_GETRECT},
    {"draw", (PyCFunction)texture_draw, METH_VARARGS | METH_KEYWORDS,
     DOC_SDL2_VIDEO_TEXTURE_DRAW},
    {"draw_triangle", (PyCFunction)texture_draw_triangle,
     METH_VARARGS | METH_KEYWORDS, DOC_SDL2_VIDEO_TEXTURE_DRAWTRIANGLE},
    {"draw_quad", (PyCFunction)texture_draw_quad, METH_VARARGS | METH_KEYWORDS,
     DOC_SDL2_VIDEO_TEXTURE_DRAWQUAD},
    {"update", (PyCFunction)texture_update, METH_VARARGS | METH_KEYWORDS,
     DOC_SDL2_VIDEO_TEXTURE_UPDATE},
    {"from_surface", (PyCFunction)texture_from_surface,
     METH_VARARGS | METH_KEYWORDS | METH_CLASS,
     DOC_SDL2_VIDEO_TEXTURE_FROMSURFACE},
    {NULL, NULL, 0, NULL}};

static PyGetSetDef texture_getset[] = {
    {"renderer", (getter)texture_get_renderer, (setter)NULL,
     DOC_SDL2_VIDEO_TEXTURE_RENDERER, NULL},
    {"width", (getter)texture_get_width, (setter)NULL,
     DOC_SDL2_VIDEO_TEXTURE_WIDTH, NULL},
    {"height", (getter)texture_get_height, (setter)NULL,
     DOC_SDL2_VIDEO_TEXTURE_HEIGHT, NULL},
    {"alpha", (getter)texture_get_alpha, (setter)texture_set_alpha,
     DOC_SDL2_VIDEO_TEXTURE_ALPHA, NULL},
    {"blend_mode", (getter)texture_get_blend_mode,
     (setter)texture_set_blend_mode, DOC_SDL2_VIDEO_TEXTURE_BLENDMODE, NULL},
    {"color", (getter)texture_get_color, (setter)texture_set_color,
     DOC_SDL2_VIDEO_TEXTURE_COLOR, NULL},
    {NULL, 0, NULL, NULL, NULL}};

static PyMethodDef image_methods[] = {{NULL, NULL, 0, NULL}};

static PyGetSetDef image_getset[] = {{NULL, 0, NULL, NULL, NULL}};

static PyTypeObject pgRenderer_Type = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "pygame._render.Renderer",
    .tp_basicsize = sizeof(pgRendererObject),
    .tp_dealloc = (destructor)renderer_dealloc,
    .tp_doc = DOC_SDL2_VIDEO_RENDERER,
    .tp_methods = renderer_methods,
    .tp_init = (initproc)renderer_init,
    .tp_new = PyType_GenericNew,
    .tp_getset = renderer_getset};

static PyTypeObject pgTexture_Type = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "pygame._render.Texture",
    .tp_basicsize = sizeof(pgTextureObject),
    .tp_dealloc = (destructor)texture_dealloc,
    .tp_doc = DOC_SDL2_VIDEO_TEXTURE,
    .tp_weaklistoffset = offsetof(pgTextureObject, weakreflist),
    .tp_methods = texture_methods,
    .tp_init = (initproc)texture_init,
    .tp_new = PyType_GenericNew,
    .tp_getset = texture_getset};

static PyTypeObject pgImage_Type = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "pygame._render.Image",
    .tp_basicsize = sizeof(pgImageObject),
    //.tp_dealloc = (destructor)image_dealloc,
    .tp_doc = DOC_SDL2_VIDEO_IMAGE, .tp_methods = image_methods,
    //.tp_init = (initproc)image_init,
    .tp_new = PyType_GenericNew, .tp_getset = image_getset};

static PyMethodDef _render_methods[] = {{NULL, NULL, 0, NULL}};

MODINIT_DEFINE(_render)
{
    PyObject *module, *apiobj;
    static void *c_api[PYGAMEAPI_RENDER_NUMSLOTS];

    static struct PyModuleDef _module = {PyModuleDef_HEAD_INIT,
                                         "_render",
                                         "docs_needed",
                                         -1,
                                         _render_methods,
                                         NULL,
                                         NULL,
                                         NULL,
                                         NULL};

    /* imported needed apis; Do this first so if there is an error
       the module is not loaded.
    */
    import_pygame_base();
    if (PyErr_Occurred()) {
        return NULL;
    }

    import_pygame_surface();
    if (PyErr_Occurred()) {
        return NULL;
    }

    import_pygame_rect();
    if (PyErr_Occurred()) {
        return NULL;
    }

    import_pygame_color();
    if (PyErr_Occurred()) {
        return NULL;
    }

    import_pygame_window();
    if (PyErr_Occurred()) {
        return NULL;
    }

    if (PyType_Ready(&pgRenderer_Type) < 0) {
        return NULL;
    }

    if (PyType_Ready(&pgTexture_Type) < 0) {
        return NULL;
    }

    if (PyType_Ready(&pgImage_Type) < 0) {
        return NULL;
    }

    /* create the module */
    module = PyModule_Create(&_module);
    if (module == 0) {
        return NULL;
    }

    if (PyModule_AddObjectRef(module, "Renderer",
                              (PyObject *)&pgRenderer_Type)) {
        Py_DECREF(module);
        return NULL;
    }

    if (PyModule_AddObjectRef(module, "Texture",
                              (PyObject *)&pgTexture_Type)) {
        Py_DECREF(module);
        return NULL;
    }

    if (PyModule_AddObjectRef(module, "Image", (PyObject *)&pgImage_Type)) {
        Py_DECREF(module);
        return NULL;
    }

    c_api[0] = &pgRenderer_Type;
    c_api[1] = &pgTexture_Type;
    c_api[2] = &pgImage_Type;
    apiobj = encapsulate_api(c_api, "_render");
    if (PyModule_AddObject(module, PYGAMEAPI_LOCAL_ENTRY, apiobj)) {
        Py_XDECREF(apiobj);
        Py_DECREF(module);
        return NULL;
    }

    return module;
}
