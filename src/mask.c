/*
  Copyright (C) 2002-2007 Ulf Ekstrom except for the bitcount function.
  This wrapper code was originally written by Danny van Bruggen(?) for
  the SCAM library, it was then converted by Ulf Ekstrom to wrap the
  bitmask library, a spinoff from SCAM.

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

#include "pygame.h"
#include "pygamedocs.h"
#include "structmember.h"
#include "bitmask.h"


typedef struct {
  PyObject_HEAD
  bitmask_t *mask;
} PyMaskObject;

staticforward PyTypeObject PyMask_Type;
#define PyMask_Check(x) ((x)->ob_type == &PyMask_Type)
#define PyMask_AsBitmap(x) (((PyMaskObject*)x)->mask)


/* mask object methods */

static PyObject* mask_get_size(PyObject* self, PyObject* args)
{
    bitmask_t *mask = PyMask_AsBitmap(self);

    if(!PyArg_ParseTuple(args, ""))
        return NULL;

    return Py_BuildValue("(ii)", mask->w, mask->h);
}

static PyObject* mask_get_at(PyObject* self, PyObject* args)
{
    bitmask_t *mask = PyMask_AsBitmap(self);
    int x, y, val;

    if(!PyArg_ParseTuple(args, "(ii)", &x, &y))
            return NULL;
    if (x >= 0 && x < mask->w && y >= 0 && y < mask->h) {
        val = bitmask_getbit(mask, x, y);
    } else {
        PyErr_Format(PyExc_IndexError, "%d, %d is out of bounds", x, y);
        return NULL;
    }

    return PyInt_FromLong(val);
}

static PyObject* mask_set_at(PyObject* self, PyObject* args)
{
    bitmask_t *mask = PyMask_AsBitmap(self);
    int x, y, value = 1;

    if(!PyArg_ParseTuple(args, "(ii)|i", &x, &y, &value))
            return NULL;
    if (x >= 0 && x < mask->w && y >= 0 && y < mask->h) {
        if (value) {
            bitmask_setbit(mask, x, y);
        } else {
          bitmask_clearbit(mask, x, y);
        }
    } else {
        PyErr_Format(PyExc_IndexError, "%d, %d is out of bounds", x, y);
        return NULL;
    }
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject* mask_overlap(PyObject* self, PyObject* args)
{
    bitmask_t *mask = PyMask_AsBitmap(self);
    bitmask_t *othermask;
    PyObject *maskobj;
    int x, y, val;
    int xp,yp;

    if(!PyArg_ParseTuple(args, "O!(ii)", &PyMask_Type, &maskobj, &x, &y))
            return NULL;
    othermask = PyMask_AsBitmap(maskobj);

    val = bitmask_overlap_pos(mask, othermask, x, y, &xp, &yp);
    if (val) {
      return Py_BuildValue("(ii)", xp,yp);
    } else {
        Py_INCREF(Py_None);
        return Py_None;
    }
}


static PyObject* mask_overlap_area(PyObject* self, PyObject* args)
{
    bitmask_t *mask = PyMask_AsBitmap(self);
    bitmask_t *othermask;
    PyObject *maskobj;
    int x, y, val;

    if(!PyArg_ParseTuple(args, "O!(ii)", &PyMask_Type, &maskobj, &x, &y)) {
        return NULL;
    }
    othermask = PyMask_AsBitmap(maskobj);

    val = bitmask_overlap_area(mask, othermask, x, y);
    return PyInt_FromLong(val);
}

/*
def maskFromSurface(surface, threshold = 127):
    mask = pygame.Mask(surface.get_size())
    key = surface.get_colorkey()
    if key:
        for y in range(surface.get_height()):
            for x in range(surface.get_width()):
                if surface.get_at((x+0.1,y+0.1)) != key:
                    mask.set_at((x,y),1)
    else:
        for y in range(surface.get_height()):
            for x in range (surface.get_width()):
                if surface.get_at((x,y))[3] > threshold:
                    mask.set_at((x,y),1)
    return mask
*/




static PyObject* mask_from_surface(PyObject* self, PyObject* args)
{
    bitmask_t *mask;
    SDL_Surface* surf;

    PyObject* surfobj;
    PyMaskObject *maskobj;

    int x, y, threshold;
    Uint8 *pixels;

    SDL_PixelFormat *format;
    Uint32 color;
    Uint8 *pix;
    Uint8 r, g, b, a;

    /* set threshold as 127 default argument. */
    threshold = 127;

    /* get the surface from the passed in arguments. 
     *   surface, threshold
     */

    if (!PyArg_ParseTuple (args, "O!|i", &PySurface_Type, &surfobj, &threshold)) {
        return NULL;
    }

    surf = PySurface_AsSurface(surfobj);

    /* lock the surface, release the GIL. */
    PySurface_Lock (surfobj);



    Py_BEGIN_ALLOW_THREADS;



    /* get the size from the surface, and create the mask. */
    mask = bitmask_create(surf->w, surf->h);


    if(!mask) {
        /* Py_END_ALLOW_THREADS;
         */
        return NULL; /*RAISE(PyExc_Error, "cannot create bitmask");*/
    }
    


    /* TODO: this is the slow, but easy to code way.  Could make the loop 
     *         just increment a pointer depending on the format & duff unroll.
     *         It's faster than in python anyhow.
     */
    pixels = (Uint8 *) surf->pixels;
    format = surf->format;

    for(y=0; y < surf->h; y++) {
        for(x=0; x < surf->w; x++) {
            /* Get the color.  TODO: should use an inline helper 
             *   function for this common function. */
            switch (format->BytesPerPixel)
            {
                case 1:
                    color = (Uint32)*((Uint8 *) pixels + y * surf->pitch + x);
                    break;
                case 2:
                    color = (Uint32)*((Uint16 *) (pixels + y * surf->pitch) + x);
                    break;
                case 3:
                    pix = ((Uint8 *) (pixels + y * surf->pitch) + x * 3);
                #if SDL_BYTEORDER == SDL_LIL_ENDIAN
                    color = (pix[0]) + (pix[1] << 8) + (pix[2] << 16);
                #else
                    color = (pix[2]) + (pix[1] << 8) + (pix[0] << 16);
                #endif
                    break;
                default:                  /* case 4: */
                    color = *((Uint32 *) (pixels + y * surf->pitch) + x);
                    break;
            }


            if (!(surf->flags & SDL_SRCCOLORKEY)) {

                SDL_GetRGBA (color, format, &r, &g, &b, &a);

                /* no colorkey, so we check the threshold of the alpha */
                if (a > threshold) {
                    bitmask_setbit(mask, x, y);
                }
            } else {
                /*  test against the colour key. */
                if (format->colorkey != color) {
                    bitmask_setbit(mask, x, y);
                }
            }
        }
    }


    Py_END_ALLOW_THREADS;

    /* unlock the surface, release the GIL.
     */
    PySurface_Unlock (surfobj);

    /*create the new python object from mask*/        
    maskobj = PyObject_New(PyMaskObject, &PyMask_Type);
    if(maskobj)
        maskobj->mask = mask;


    return (PyObject*)maskobj;
}



/*

def get_bounding_boxes(surf):
    """
    """
    width, height = surf.get_width(), surf.get_height()

    regions = []

    # used pixels is Rects[y][x]
    used_pixels = []
    for y in xrange(height):
        widthones = []
        for x in xrange(width):
            widthones.append(None)
        used_pixels.append(widthones)


    for y in xrange(height):
        for x in xrange(width):
            c = surf.get_at((x, y))
            # if the pixel has been set.
            if c[0]:
                if not used_pixels[y][x]:
                    used_pixels[y][x] = pygame.Rect(x,y,1,1)
                    regions.append( used_pixels[y][x] )

                aregion = used_pixels[y][x] 

                # check other directions, clockwise.  mark a pixel as used if it is.
                for dx, dy in [(0,-1), (1,-1), (1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1)]:
                    try:
                        ax, ay = x+dx, y+dy
                        if surf.get_at((ax,ay))[0]:
                            if not used_pixels[ay][ax]:
                                aregion.union_ip( pygame.Rect(ax,ay, 1, 1) )
                            used_pixels[ay][ax] = aregion
                    except:
                        pass


    return regions
*/


/* returns an array of regions in regions. */

static GAME_Rect* get_bounding_rects(bitmask_t *mask, int *num_bounding_boxes) {

    int x, y, p, i, width, height;
    GAME_Rect **used_pixels;
    GAME_Rect *a_used_pixels;
    GAME_Rect *direction_used_pixels;
    GAME_Rect *regions;

    GAME_Rect *aregion, *the_regions;
    int num_regions;
    int nx, ny, nh, nw, ay, ax;
    int directions[8][2];


    /* for dx, dy in [(0,-1), (1,-1), (1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1)]:
     */

    directions[0][0] = 0; directions[0][1] = -1;
    directions[1][0] = 1; directions[1][1] = -1;
    directions[2][0] = 1; directions[2][1] = 0;
    directions[3][0] = 1; directions[3][1] = 1;
    directions[4][0] = 0; directions[4][1] = 1;
    directions[5][0] = -1; directions[5][1] = 1;
    directions[6][0] = -1; directions[6][1] = 0;
    directions[7][0] = -1; directions[7][1] = -1;



    num_regions = 0;

    height = mask->h;
    width = mask->w;


    /* used_pixels are pointers to rects held in the regions array.  */
    used_pixels = (GAME_Rect**) malloc(sizeof(GAME_Rect*) * height * width);


    for(y=0; y < height; y++) {
        for(x=0; x < width; x++) {
            /* used_pixels[y][x] = (GAME_Rect*)NULL; */
            *((GAME_Rect **) (used_pixels + y * width) + x) = NULL;
        }
    }

    regions = (GAME_Rect*) malloc(sizeof(GAME_Rect) * height * width);

    the_regions = regions;

    for(y=0; y < height; y++) {
        for(x=0; x < width; x++) {
            p = bitmask_getbit(mask, x, y);

            if(p) {
                /* a_used_pixels is the pointer used_pixels[y][x].  */
                a_used_pixels = *((GAME_Rect **) (used_pixels + y * width) + x);

                /*
                if not used_pixels[y][x]:
                    used_pixels[y][x] = pygame.Rect(x,y,1,1)
                    regions.append( used_pixels[y][x] )

                aregion = used_pixels[y][x] 

                */
                

                if( !a_used_pixels ) {
                    /* Add the pixels as a rect on the_regions */
                    the_regions[num_regions].x = x;
                    the_regions[num_regions].y = y;
                    the_regions[num_regions].w = 1;
                    the_regions[num_regions].h = 1;
                    a_used_pixels = the_regions + num_regions;
                    num_regions++;

                }
                aregion = a_used_pixels;

                /* check other directions, clockwise.  mark a pixel as used if it is.  */

                for(i=0; i < 8; i++) {

                    ax = directions[i][0] + x;
                    ay = directions[i][1] + y;
                    
                    /* if we are within the bounds of the mask, check it. */
                    
                    if(ax >=0 && ax < width && ay < height && ay >= 0) {
                        
                        
                        /* printf("ax, ay: %d,%d\n", ax, ay);
                        */

                        if(bitmask_getbit(mask, ax, ay)) {
                            /*
                            if not used_pixels[ay][ax]:
                                aregion.union_ip( pygame.Rect(ax,ay, 1, 1) )
                            */
                            direction_used_pixels = *((GAME_Rect **) (used_pixels + ay * width) + ax);
                            if (!direction_used_pixels) {
                                nx = MIN (aregion->x, ax);
                                ny = MIN (aregion->y, ay);
                                nw = MAX (aregion->x + aregion->w, ax + 1) - nx;
                                nh = MAX (aregion->y + aregion->h, ay + 1) - ny;
                                
                                aregion->x = nx;
                                aregion->y = ny;
                                aregion->w = nw;
                                aregion->h = nh;
                            }
                            /* used_pixels[ay][ax] = aregion */
                            *((GAME_Rect **) (used_pixels + ay * width) + ax) = aregion;
                        }
                    }

                }



            }
        }
    }


    *num_bounding_boxes = num_regions;


    free(used_pixels);


    return regions;
}


static PyObject* mask_get_bounding_rects(PyObject* self, PyObject* args)
{
    GAME_Rect *regions;
    GAME_Rect *aregion;
    int num_bounding_boxes, i;
    PyObject* ret;

    PyObject* rect;



    bitmask_t *mask = PyMask_AsBitmap(self);


    ret = NULL;
    num_bounding_boxes = 0;


    if(!PyArg_ParseTuple(args, ""))
        return NULL;

    ret = PyList_New (0);
    if (!ret)
        return NULL;


    Py_BEGIN_ALLOW_THREADS;

    regions = get_bounding_rects(mask, &num_bounding_boxes);

    Py_END_ALLOW_THREADS;


    /* printf("num_bounding_boxes:%d\n", num_bounding_boxes); */


    /* build a list of rects to return.  */
    for(i=0; i < num_bounding_boxes; i++) {
        aregion = regions + i;
        /* printf("aregion x,y,w,h:%d,%d,%d,%d\n", aregion->x, aregion->y, aregion->w, aregion->h);
        */

        rect = PyRect_New4 ( aregion->x, aregion->y, aregion->w, aregion->h );
        PyList_Append (ret, rect);
        Py_DECREF (rect);
    }

    free(regions);


    return ret;
}




static PyMethodDef maskobj_builtins[] =
{
    { "get_size", mask_get_size, METH_VARARGS, DOC_MASKGETSIZE},
    { "get_at", mask_get_at, METH_VARARGS, DOC_MASKGETAT },
    { "set_at", mask_set_at, METH_VARARGS, DOC_MASKSETAT },
    { "overlap", mask_overlap, METH_VARARGS, DOC_MASKOVERLAP },
    { "overlap_area", mask_overlap_area, METH_VARARGS,
      DOC_MASKOVERLAPAREA },
    { "get_bounding_rects", mask_get_bounding_rects, METH_VARARGS,
      DOC_MASKGETBOUNDINGRECTS },

    { NULL, NULL, 0, NULL }
};



/*mask object internals*/

static void mask_dealloc(PyObject* self)
{
    bitmask_t *mask = PyMask_AsBitmap(self);
    bitmask_free(mask);
    PyObject_DEL(self);
}


static PyObject* mask_getattr(PyObject* self, char* attrname)
{
    return Py_FindMethod(maskobj_builtins, self, attrname);
}


static PyTypeObject PyMask_Type = 
{
    PyObject_HEAD_INIT(NULL)
    0,
    "pygame.mask.Mask",
    sizeof(PyMaskObject),
    0,
    mask_dealloc,
    0,
    mask_getattr,
    0,
    0,
    0,
    0,
    NULL,
    0, 
    (hashfunc)NULL,
    (ternaryfunc)NULL,
    (reprfunc)NULL,
    0L,0L,0L,0L,
    DOC_PYGAMEMASKMASK /* Documentation string */
};


/*mask module methods*/

static PyObject* Mask(PyObject* self, PyObject* args)
{
    bitmask_t *mask;
    int w,h;
    PyMaskObject *maskobj;
    if(!PyArg_ParseTuple(args, "(ii)", &w, &h))
        return NULL;
    mask = bitmask_create(w,h);

    if(!mask)
      return NULL; /*RAISE(PyExc_Error, "cannot create bitmask");*/
        
        /*create the new python object from mask*/        
    maskobj = PyObject_New(PyMaskObject, &PyMask_Type);
    if(maskobj)
        maskobj->mask = mask;
    return (PyObject*)maskobj;
}



static PyMethodDef mask_builtins[] =
{
    { "Mask", Mask, METH_VARARGS, DOC_PYGAMEMASKMASK },
    { "from_surface", mask_from_surface, METH_VARARGS,
      DOC_PYGAMEMASKFROMSURFACE},
    { NULL, NULL, 0, NULL }
};

void initmask(void)
{
  PyObject *module, *dict;
  PyType_Init(PyMask_Type);
  
  /* create the module */
  module = Py_InitModule3("mask", mask_builtins, DOC_PYGAMEMASK);
  dict = PyModule_GetDict(module);
  PyDict_SetItemString(dict, "MaskType", (PyObject *)&PyMask_Type);
  import_pygame_base ();
  import_pygame_surface ();
  import_pygame_rect ();
}

