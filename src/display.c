/*
    PyGame - Python Game Library
    Copyright (C) 2000  Pete Shinners

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

    Pete Shinners
    pete@shinners.org
*/

/*
 *  PyGAME display module
 */
#define PYGAMEAPI_DISPLAY_INTERNAL
#include "pygame.h"



staticforward PyTypeObject PyVidInfo_Type;
static PyObject* PyVidInfo_New(const SDL_VideoInfo* info);
#define PyVidInfo_Check(x) ((x)->ob_type == &PyVidInfo_Type)


/*quick internal test to see if gamma is supported*/
static int check_hasgamma()
{
	int result = 0;
/*
	Uint8 r[256], g[256], b[256];
printf("checking for gamma...\n");
	result =  SDL_GetGammaRamp(r, g, b) != -1;
printf("...done\n");
*/
	return result;
}




/* init routines */


    /*DOC*/ static char doc_quit[] =
    /*DOC*/    "pygame.display.quit() -> None\n"
    /*DOC*/    "uninitialize the display module\n"
    /*DOC*/    "\n"
    /*DOC*/    "Manually uninitialize SDL's video subsystem. It is\n"
    /*DOC*/    "safe to call this if the video is currently not\n"
    /*DOC*/    "initialized.\n"
    /*DOC*/ ;

static PyObject* quit(PyObject* self, PyObject* arg)
{
	if(!PyArg_ParseTuple(arg, ""))
		return NULL;

	PyGame_Video_AutoQuit();

	RETURN_NONE
}


    /*DOC*/ static char doc_init[] =
    /*DOC*/    "pygame.display.init() -> None\n"
    /*DOC*/    "initialize the display module\n"
    /*DOC*/    "\n"
    /*DOC*/    "Manually initialize SDL's video subsystem. Will\n"
    /*DOC*/    "raise an exception if it cannot be initialized. It\n"
    /*DOC*/    "is safe to call this function if the video has is\n"
    /*DOC*/    "currently initialized.\n"
    /*DOC*/ ;

static PyObject* init(PyObject* self, PyObject* arg)
{
	if(!PyArg_ParseTuple(arg, ""))
		return NULL;

	if(!PyGame_Video_AutoInit())
		return RAISE(PyExc_SDLError, SDL_GetError());

	RETURN_NONE
}



    /*DOC*/ static char doc_get_init[] =
    /*DOC*/    "pygame.display.get_init() -> bool\n"
    /*DOC*/    "get status of display module initialization\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns true if SDL's video system is currently\n"
    /*DOC*/    "intialized.\n"
    /*DOC*/ ;

static PyObject* get_init(PyObject* self, PyObject* arg)
{
	if(!PyArg_ParseTuple(arg, ""))
		return NULL;

	return PyInt_FromLong(SDL_WasInit(SDL_INIT_CDROM)!=0);
}



    /*DOC*/ static char doc_get_active[] =
    /*DOC*/    "pygame.display.get_active() -> bool\n"
    /*DOC*/    "get state of display mode\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns true if the current display is active on\n"
    /*DOC*/    "the screen. This done with the call to\n"
    /*DOC*/    "pygame.display.set_mode(). It is potentially\n"
    /*DOC*/    "subject to the activity of a running window\n"
    /*DOC*/    "manager.\n"
    /*DOC*/ ;

static PyObject* get_active(PyObject* self, PyObject* arg)
{
	if(!PyArg_ParseTuple(arg, ""))
		return NULL;

	return PyInt_FromLong((SDL_GetAppState()&SDL_APPACTIVE) != 0);
}



/* vidinfo object */

static void vidinfo_dealloc(PyObject* self)
{
	PyMem_DEL(self);	
}



static PyObject *vidinfo_getattr(PyObject *self, char *name)
{
	SDL_VideoInfo* info = &((PyVidInfoObject*)self)->info;

	if(!strcmp(name, "hw"))
		return PyInt_FromLong(info->hw_available);
	else if(!strcmp(name, "wm"))
		return PyInt_FromLong(info->wm_available);
/*	else if(!strcmp(name, "gamma"))
		return PyInt_FromLong(check_hasgamma());
*/	else if(!strcmp(name, "blit_hw"))
		return PyInt_FromLong(info->blit_hw);
	else if(!strcmp(name, "blit_hw_CC"))
		return PyInt_FromLong(info->blit_hw_CC);
	else if(!strcmp(name, "blit_hw_A"))
		return PyInt_FromLong(info->blit_hw_A);
	else if(!strcmp(name, "blit_sw"))
		return PyInt_FromLong(info->blit_hw);
	else if(!strcmp(name, "blit_sw_CC"))
		return PyInt_FromLong(info->blit_hw_CC);
	else if(!strcmp(name, "blit_sw_A"))
		return PyInt_FromLong(info->blit_hw_A);
	else if(!strcmp(name, "blit_fill"))
		return PyInt_FromLong(info->blit_fill);
	else if(!strcmp(name, "video_mem"))
		return PyInt_FromLong(info->video_mem);
	else if(!strcmp(name, "bitsize"))
		return PyInt_FromLong(info->vfmt->BitsPerPixel);
	else if(!strcmp(name, "bytesize"))
		return PyInt_FromLong(info->vfmt->BytesPerPixel);
	else if(!strcmp(name, "masks"))
		return Py_BuildValue("(iiii)", info->vfmt->Rmask, info->vfmt->Gmask,
					info->vfmt->Bmask, info->vfmt->Amask);
	else if(!strcmp(name, "shifts"))
		return Py_BuildValue("(iiii)", info->vfmt->Rshift, info->vfmt->Gshift,
					info->vfmt->Bshift, info->vfmt->Ashift);
	else if(!strcmp(name, "losses"))
		return Py_BuildValue("(iiii)", info->vfmt->Rloss, info->vfmt->Gloss,
					info->vfmt->Bloss, info->vfmt->Aloss);
	
	return RAISE(PyExc_AttributeError, "does not exist in vidinfo");
}


PyObject* vidinfo_str(PyObject* self)
{
	char str[1024];
	SDL_VideoInfo* info = &((PyVidInfoObject*)self)->info;

	sprintf(str, "<VideoInfo(hw = %d, wm = %d,video_mem = %d\n"
				 "           blit_hw = %d, blit_hw_CC = %d, blit_hw_A = %d,\n"
				 "           blit_sw = %d, blit_sw_CC = %d, blit_sw_A = %d,\n"
				 "           bitsize  = %d, bytesize = %d,\n"
				 "           masks =  (%d, %d, %d, %d),\n"
				 "           shifts = (%d, %d, %d, %d),\n"
				 "           losses =  (%d, %d, %d, %d)>\n",
				info->hw_available, info->wm_available, info->video_mem,
				info->blit_hw, info->blit_hw_CC, info->blit_hw_A,
				info->blit_sw, info->blit_sw_CC, info->blit_sw_A,
				info->vfmt->BitsPerPixel, info->vfmt->BytesPerPixel,
				info->vfmt->Rmask, info->vfmt->Gmask, info->vfmt->Bmask, info->vfmt->Amask,
				info->vfmt->Rshift, info->vfmt->Gshift, info->vfmt->Bshift, info->vfmt->Ashift,
				info->vfmt->Rloss, info->vfmt->Gloss, info->vfmt->Bloss, info->vfmt->Aloss);

	return PyString_FromString(str);
}


static PyTypeObject PyVidInfo_Type =
{
	PyObject_HEAD_INIT(NULL)
	0,						/*size*/
	"VidInfo",				/*name*/
	sizeof(PyVidInfoObject),/*basic size*/
	0,						/*itemsize*/
	vidinfo_dealloc,		/*dealloc*/
	0,						/*print*/
	vidinfo_getattr,		/*getattr*/
	NULL,					/*setattr*/
	NULL,					/*compare*/
	vidinfo_str,			/*repr*/
	NULL,					/*as_number*/
	NULL,					/*as_sequence*/
	NULL,					/*as_mapping*/
	(hashfunc)NULL,			/*hash*/
	(ternaryfunc)NULL,		/*call*/
	(reprfunc)NULL,			/*str*/
};


static PyObject* PyVidInfo_New(const SDL_VideoInfo* i)
{
	PyVidInfoObject* info;

	if(!i) return RAISE(PyExc_SDLError, SDL_GetError());
	
	info = PyObject_NEW(PyVidInfoObject, &PyVidInfo_Type);
	if(!info) return NULL;

	memcpy(&info->info, i, sizeof(SDL_VideoInfo));
	return (PyObject*)info;
}



/* display functions */

    /*DOC*/ static char doc_set_driver[] =
    /*DOC*/    "pygame.display.set_driver(name) -> None\n"
    /*DOC*/    "override the default sdl video driver\n"
    /*DOC*/    "\n"
    /*DOC*/    "Changes the SDL environment to initialize with the\n"
    /*DOC*/    "given named videodriver. This can only be changed\n"
    /*DOC*/    "before the display is initialized. If this is not\n"
    /*DOC*/    "called, SDL will use it's default video driver, or\n"
    /*DOC*/    "the one in the environment variable\n"
    /*DOC*/    "SDL_VIDEODRIVER.\n"
    /*DOC*/ ;

static PyObject* set_driver(PyObject* self, PyObject* arg)
{
	char* name;
	if(!PyArg_ParseTuple(arg, "s", &name))
		return NULL;

	if(SDL_WasInit(SDL_INIT_VIDEO))
		return RAISE(PyExc_SDLError, "cannot switch video driver while initialized");

	/*override the default video driver*/
	/*environment variable: SDL_VIDEODRIVER*/

	RETURN_NONE
}


    /*DOC*/ static char doc_get_driver[] =
    /*DOC*/    "pygame.display.get_driver() -> name\n"
    /*DOC*/    "get the current sdl video driver\n"
    /*DOC*/    "\n"
    /*DOC*/    "Once the display is initialized, this will return\n"
    /*DOC*/    "the name of the currently running video driver.\n"
    /*DOC*/    "There is no way to get a list of all the supported\n"
    /*DOC*/    "video drivers.\n"
    /*DOC*/ ;

static PyObject* get_driver(PyObject* self, PyObject* args)
{
	char buf[256];
	
	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	VIDEO_INIT_CHECK();

	if(!SDL_VideoDriverName(buf, sizeof(buf)))
	{
		Py_INCREF(Py_None);
		return Py_None;
	}
	return PyString_FromString(buf);
}



    /*DOC*/ static char doc_get_info[] =
    /*DOC*/    "pygame.display.get_info() -> VidInfo\n"
    /*DOC*/    "get display capabilities and settings\n"
    /*DOC*/    "\n"
    /*DOC*/    "Gets a vidinfo object that contains information\n"
    /*DOC*/    "about the capabilities and current state of the\n"
    /*DOC*/    "video driver. This can be called before the\n"
    /*DOC*/    "display mode is set, to determine the current\n"
    /*DOC*/    "video mode of a display.\n"
    /*DOC*/ ;

static PyObject* get_info(PyObject* self, PyObject* arg)
{
	const SDL_VideoInfo* info;

	if(!PyArg_ParseTuple(arg, ""))
		return NULL;

	VIDEO_INIT_CHECK();

	info = SDL_GetVideoInfo();
	return PyVidInfo_New(info);
}




    /*DOC*/ static char doc_get_surface[] =
    /*DOC*/    "pygame.display.get_surface() -> Surface\n"
    /*DOC*/    "get current display surface\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns a Surface object representing the current\n"
    /*DOC*/    "display. Will return None if called before the\n"
    /*DOC*/    "display mode is set.\n"
    /*DOC*/ ;

static PyObject* get_surface(PyObject* self, PyObject* arg)
{
	SDL_Surface* surf;
	if(!PyArg_ParseTuple(arg, ""))
		return NULL;

	VIDEO_INIT_CHECK();

	surf = SDL_GetVideoSurface();
	if(!surf)
		RETURN_NONE

	return PySurface_New(surf);
}


    /*DOC*/ static char doc_set_mode[] =
    /*DOC*/    "pygame.display.set_mode(size, [flags, [depth]]) -> Surface\n"
    /*DOC*/    "set the display mode\n"
    /*DOC*/    "\n"
    /*DOC*/    "Sets the current display mode. If calling this\n"
    /*DOC*/    "after the mode has already been set, this will\n"
    /*DOC*/    "change the display mode to the desired type.\n"
    /*DOC*/    "Sometimes an exact match for the requested video\n"
    /*DOC*/    "mode is not available. In this case SDL will try\n"
    /*DOC*/    "to find the closest match and work with that\n"
    /*DOC*/    "instead.\n"
    /*DOC*/    "\n"
    /*DOC*/    "The size is a 2-number-sequence containing the\n"
    /*DOC*/    "width and height of the desired display mode.\n"
    /*DOC*/    "Flags represents a set of different options for\n"
    /*DOC*/    "the new display mode. If omitted or given as 0, it\n"
    /*DOC*/    "will default to a simple software window. You can\n"
    /*DOC*/    "mix several flags together with the bitwise-or (|)\n"
    /*DOC*/    "operator. Possible flags are HWSURFACE (or the\n"
    /*DOC*/    "value 1), HWPALETTE, DOUBLEBUF, and/or FULLSCREEN.\n"
    /*DOC*/    "There are other flags available but these are the\n"
    /*DOC*/    "most usual. A full list of flags can be found in\n"
    /*DOC*/    "the SDL documentation.\n"
    /*DOC*/    "The optional depth arguement is the requested bits\n"
    /*DOC*/    "per pixel. It will usually be left omitted, in\n"
    /*DOC*/    "which case the display will use the best/fastest\n"
    /*DOC*/    "pixel depth available.\n"
    /*DOC*/ ;

static PyObject* set_mode(PyObject* self, PyObject* arg)
{
	SDL_Surface* surf;
	int flags = SDL_SWSURFACE, depth = 0;
	short w, h;
	char* title, *icontitle;

	if(!PyArg_ParseTuple(arg, "(ii)|ii", &w, &h, &flags, &depth))
		return NULL;

	VIDEO_INIT_CHECK();

	surf = SDL_SetVideoMode(w, h, depth, flags);
	if(!surf)
		return RAISE(PyExc_SDLError, SDL_GetError());

	SDL_WM_GetCaption(&title, &icontitle);
	if(!title || !*title)
		SDL_WM_SetCaption("PyGame Window", "PyGame");

	return PySurface_New(surf);
}



    /*DOC*/ static char doc_mode_ok[] =
    /*DOC*/    "pygame.display.mode_ok(size, [flags, [depth]]) -> int\n"
    /*DOC*/    "query a specific display mode\n"
    /*DOC*/    "\n"
    /*DOC*/    "This uses the same arguments as the call to\n"
    /*DOC*/    "pygame.display.set_mode(). It is used to determine\n"
    /*DOC*/    "if a requested display mode is available. It will\n"
    /*DOC*/    "return 0 if the requested mode is not possible.\n"
    /*DOC*/    "Otherwise it will return the best and closest\n"
    /*DOC*/    "matching bit depth for the mode requested.\n"
    /*DOC*/    "\n"
    /*DOC*/    "The size is a 2-number-sequence containing the\n"
    /*DOC*/    "width and height of the desired display mode.\n"
    /*DOC*/    "Flags represents a set of different options for\n"
    /*DOC*/    "the display mode. If omitted or given as 0, it\n"
    /*DOC*/    "will default to a simple software window. You can\n"
    /*DOC*/    "mix several flags together with the bitwise-or (|)\n"
    /*DOC*/    "operator. Possible flags are HWSURFACE (or the\n"
    /*DOC*/    "value 1), HWPALETTE, DOUBLEBUF, and/or FULLSCREEN.\n"
    /*DOC*/    "There are other flags available but these are the\n"
    /*DOC*/    "most usual. A full list of flags can be found in\n"
    /*DOC*/    "the SDL documentation.\n"
    /*DOC*/    "The optional depth arguement is the requested bits\n"
    /*DOC*/    "per pixel. It will usually be left omitted, in\n"
    /*DOC*/    "which case the display will use the best/fastest\n"
    /*DOC*/    "pixel depth available.\n"
    /*DOC*/ ;

static PyObject* mode_ok(PyObject* self, PyObject* args)
{
	int flags=SDL_SWSURFACE, depth=0;
	short w, h;

	VIDEO_INIT_CHECK();

	if(!PyArg_ParseTuple(args, "(ii)|ii", &w, &h, &flags, &depth))
		return NULL;

	if(!depth)
		depth = SDL_GetVideoInfo()->vfmt->BitsPerPixel;

	return PyInt_FromLong(SDL_VideoModeOK(w, h, depth, flags));
}



    /*DOC*/ static char doc_list_modes[] =
    /*DOC*/    "pygame.display.list_modes([depth, [flags]]) -> [[x,y],...] | -1\n"
    /*DOC*/    "query all resolutions for requested mode\n"
    /*DOC*/    "\n"
    /*DOC*/    "This function returns a list of possible\n"
    /*DOC*/    "dimensions for a specified color depth. The return\n"
    /*DOC*/    "value will be an empty list of no display modes\n"
    /*DOC*/    "are available with the given arguments. A return\n"
    /*DOC*/    "value of -1 means that any requested resolution\n"
    /*DOC*/    "should work (this is likely the case for windowed\n"
    /*DOC*/    "modes). Mode sizes are sorted from biggest to\n"
    /*DOC*/    "smallest.\n"
    /*DOC*/    "\n"
    /*DOC*/    "If depth is not passed or 0, SDL will choose the\n"
    /*DOC*/    "current/best color depth for the display. You will\n"
    /*DOC*/    "usually want to pass FULLSCREEN when using the\n"
    /*DOC*/    "flags, if flags is omitted, FULLSCREEN is the\n"
    /*DOC*/    "default.\n"
    /*DOC*/ ;

static PyObject* list_modes(PyObject* self, PyObject* args)
{
	SDL_PixelFormat format;
	SDL_Rect** rects;
	int flags=SDL_FULLSCREEN;
	PyObject *list, *size;
	
	format.BitsPerPixel = 0;
	if(PyTuple_Size(args)!=0 && !PyArg_ParseTuple(args, "|bi", &format.BitsPerPixel, &flags))
		return NULL;

	VIDEO_INIT_CHECK();

	if(!format.BitsPerPixel)
		format.BitsPerPixel = SDL_GetVideoInfo()->vfmt->BitsPerPixel;

	rects = SDL_ListModes(&format, flags);

	if(rects == (SDL_Rect**)-1)
		return PyInt_FromLong(-1);

	if(!(list = PyList_New(0)))
		return NULL;
	if(!rects)
		return list;

	for(; *rects; ++rects)
	{
		if(!(size = Py_BuildValue("(ii)", (*rects)->w, (*rects)->h)))
		{
			Py_DECREF(list);
			return NULL;
		}
		PyList_Append(list, size);
		Py_DECREF(size);
	}
	return list;
}



    /*DOC*/ static char doc_flip[] =
    /*DOC*/    "pygame.display.flip() -> None\n"
    /*DOC*/    "update the display\n"
    /*DOC*/    "\n"
    /*DOC*/    "This will update the contents of the entire\n"
    /*DOC*/    "display. If your display mode is using the flags\n"
    /*DOC*/    "HWSURFACE and DOUBLEBUF, this will wait for a\n"
    /*DOC*/    "vertical retrace and swap the surfaces. If you are\n"
    /*DOC*/    "using a different type of display mode, it will\n"
    /*DOC*/    "simply update the entire contents of the surface.\n"
    /*DOC*/ ;

static PyObject* flip(PyObject* self, PyObject* arg)
{
	SDL_Surface* screen;

	if(!PyArg_ParseTuple(arg, ""))
		return NULL;

	VIDEO_INIT_CHECK();

	screen = SDL_GetVideoSurface();
	if(!screen)
		return RAISE(PyExc_SDLError, SDL_GetError());
	if(SDL_Flip(screen) == -1)
		return RAISE(PyExc_SDLError, SDL_GetError());
	
	RETURN_NONE
}


/*BAD things happen when out-of-bound rects go to updaterect*/
static void screencroprect(GAME_Rect* r, int w, int h)
{
	if(r->x >= w || r->y >= h)
		r->x = r->y = r->w = r->h = 0;
	else
	{
		if(r->x < 0) r->x = 0;
		if(r->y < 0) r->y = 0;
		if(r->x + r->w >= w) r->w = (w-1)-r->x;
		if(r->y + r->h >= h) r->h = (h-1)-r->y;
	}
}

    /*DOC*/ static char doc_update[] =
    /*DOC*/    "pygame.display.update([rectstyle]) -> None\n"
    /*DOC*/    "update an area of the display\n"
    /*DOC*/    "\n"
    /*DOC*/    "This call will update a section (or sections) of\n"
    /*DOC*/    "the display screen. You must update an area of\n"
    /*DOC*/    "your display when you change its contents. If\n"
    /*DOC*/    "passed with no arguments, this will update the\n"
    /*DOC*/    "entire display surface. If you have many lists\n"
    /*DOC*/    "that need updating, it is best to combine them\n"
    /*DOC*/    "into a sequence and pass them all at once. This call\n"
    /*DOC*/    "will accept a sequence of rectstyle arguments\n"
    /*DOC*/ ;

static PyObject* update(PyObject* self, PyObject* arg)
{
	SDL_Surface* screen;
	GAME_Rect *gr, temp = {0};

	if(PyTuple_Size(arg) == 0)
		gr = &temp;
	else
		gr = GameRect_FromObject(arg, &temp);
	VIDEO_INIT_CHECK();

	if(gr)
	{
		screen = SDL_GetVideoSurface();
		if(!screen)
			return RAISE(PyExc_SDLError, SDL_GetError());
		
		screencroprect(gr, screen->w, screen->h);
		SDL_UpdateRect(screen, gr->x, gr->y, gr->w, gr->h);
	}
	else
	{
		PyObject* seq;
		PyObject* r;
		int loop, num;
		SDL_Rect* rects;

		if(PyTuple_Size(arg) != 1)
			return RAISE(PyExc_ValueError, "update requires a rectstyle or sequence of recstyles");

		seq = PyTuple_GET_ITEM(arg, 0);
		if(!seq || !PySequence_Check(seq))
			return RAISE(PyExc_ValueError, "update requires a rectstyle or sequence of recstyles");

		VIDEO_INIT_CHECK();

		screen = SDL_GetVideoSurface();
		if(!screen)
			return RAISE(PyExc_SDLError, SDL_GetError());
		
		num = PySequence_Length(seq);
		rects = PyMem_New(SDL_Rect, num);
		if(!rects) return NULL;
		for(loop = 0; loop < num; ++loop)
		{
			r = PySequence_GetItem(seq, loop);
			gr = GameRect_FromObject(r, &temp);
			if(!gr)
			{
				Py_XDECREF(r);
				PyMem_Free(rects);
				return RAISE(PyExc_ValueError, "update_rects requires a single list of rects");
			}
			screencroprect(gr, screen->w, screen->h);
			rects[loop].x = gr->x;
			rects[loop].y = gr->y;
			rects[loop].w = (unsigned short)gr->w;
			rects[loop].h = (unsigned short)gr->h;
		}

		SDL_UpdateRects(screen, num, rects);
		PyMem_Free(rects);
	}
	
	RETURN_NONE
}


    /*DOC*/ static char doc_set_gamma[] =
    /*DOC*/    "pygame.display.set_gamma(r, [g, b]) -> bool\n"
    /*DOC*/    "change the brightness of the display\n"
    /*DOC*/    "\n"
    /*DOC*/    "Sets the display gamma to the given amounts. If\n"
    /*DOC*/    "green and blue are ommitted, the red value will be\n"
    /*DOC*/    "used for all three colors. The color arguments are\n"
    /*DOC*/    "floating point values with 1.0 being the normal\n"
    /*DOC*/    "value.\n"
    /*DOC*/    "If you are using a display mode with a hardware\n"
    /*DOC*/    "palette, this will simply update the palette you\n"
    /*DOC*/    "are using. Not all hardware supports gamma. The\n"
    /*DOC*/    "return value will be true on success.\n"
    /*DOC*/ ;

static PyObject* set_gamma(PyObject* self, PyObject* arg)
{
	float r, g, b;
	int result;

	if(!PyArg_ParseTuple(arg, "f|ff", &r, &g, &b))
		return NULL;
	if(PyTuple_Size(arg) == 1)
		g = b = r;

	VIDEO_INIT_CHECK();

	result = SDL_SetGamma(r, g, b);
	return PyInt_FromLong(result == 0);
}



    /*DOC*/ static char doc_set_caption[] =
    /*DOC*/    "pygame.display.set_caption(title, [icontitle]) -> None\n"
    /*DOC*/    "changes the title of the window\n"
    /*DOC*/    "\n"
    /*DOC*/    "If the display has a window title, this routine\n"
    /*DOC*/    "will change the\n"
    /*DOC*/    "name on the window. Some environments support a\n"
    /*DOC*/    "shorter icon title\n"
    /*DOC*/    "to be used when the display is minimized. If\n"
    /*DOC*/    "icontitle is omittied\n"
    /*DOC*/    "it will be the same as caption title.\n"
    /*DOC*/ ;

static PyObject* set_caption(PyObject* self, PyObject* arg)
{
	char* title, *icontitle=NULL;

	if(!PyArg_ParseTuple(arg, "s|s", &title, &icontitle))
		return NULL;

	if(!icontitle)
		icontitle = title;

	VIDEO_INIT_CHECK();

	SDL_WM_SetCaption(title, icontitle);

	RETURN_NONE
}



    /*DOC*/ static char doc_get_caption[] =
    /*DOC*/    "pygame.display.get_caption() -> title, icontitle\n"
    /*DOC*/    "get the current title of the window\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns the current title and icontitle for the\n"
    /*DOC*/    "display window.\n"
    /*DOC*/ ;

static PyObject* get_caption(PyObject* self, PyObject* arg)
{
	char* title, *icontitle;

	if(!PyArg_ParseTuple(arg, ""))
		return NULL;

	VIDEO_INIT_CHECK();

	SDL_WM_GetCaption(&title, &icontitle);
	
	if(title && *title)
		return Py_BuildValue("(ss)", title, icontitle);

	return Py_BuildValue("()");
}



    /*DOC*/ static char doc_iconify[] =
    /*DOC*/    "pygame.display.iconify() -> bool\n"
    /*DOC*/    "minimize the display window\n"
    /*DOC*/    "\n"
    /*DOC*/    "Tells the window manager (if available) to\n"
    /*DOC*/    "minimize the application. The call will return\n"
    /*DOC*/    "true if successful. You will receive an APPACTIVE\n"
    /*DOC*/    "event on the event queue when the window has been\n"
    /*DOC*/    "minimized.\n"
    /*DOC*/ ;

static PyObject* iconify(PyObject* self, PyObject* arg)
{
	int result;
	if(!PyArg_ParseTuple(arg, ""))
		return NULL;

	VIDEO_INIT_CHECK();

	result = SDL_WM_IconifyWindow();
	return PyInt_FromLong(result != 0);
}



    /*DOC*/ static char doc_toggle_fullscreen[] =
    /*DOC*/    "pygame.display.toggle_fullscreen() -> bool\n"
    /*DOC*/    "switch the display fullscreen mode\n"
    /*DOC*/    "\n"
    /*DOC*/    "Tells the window manager (if available) to switch\n"
    /*DOC*/    "between windowed and fullscreen mode. If available\n"
    /*DOC*/    "and successfull, will return true. Note, there is\n"
    /*DOC*/    "currently limited platform support for this call.\n"
    /*DOC*/ ;

static PyObject* toggle_fullscreen(PyObject* self, PyObject* arg)
{
	SDL_Surface* screen;
	int result;

	if(!PyArg_ParseTuple(arg, ""))
		return NULL;

	VIDEO_INIT_CHECK();

	screen = SDL_GetVideoSurface();
	if(!screen)
		return RAISE(PyExc_SDLError, SDL_GetError());

	result = SDL_WM_ToggleFullScreen(screen);
	return PyInt_FromLong(result != 0);
}



static PyMethodDef display_builtins[] =
{
	{ "init", init, 1, doc_init },
	{ "quit", quit, 1, doc_quit },
	{ "get_init", get_init, 1, doc_get_init },
	{ "get_active", get_active, 1, doc_get_active },

/*	{ "set_driver", set_driver, 1, doc_set_driver },
	{ "get_driver", get_driver, 1, doc_get_driver },
*/	{ "get_info", get_info, 1, doc_get_info },
	{ "get_surface", get_surface, 1, doc_get_surface },

	{ "set_mode", set_mode, 1, doc_set_mode },
	{ "mode_ok", mode_ok, 1, doc_mode_ok },
	{ "list_modes", list_modes, 1, doc_list_modes },

	{ "flip", flip, 1, doc_flip },
	{ "update", update, 1, doc_update },

	{ "set_gamma", set_gamma, 1, doc_set_gamma },
	/*gammaramp support will be added later, if needed?*/

	{ "set_caption", set_caption, 1, doc_set_caption },
	{ "get_caption", get_caption, 1, doc_get_caption },

	/*{ "set_icon", set_icon, 1, doc_set_icon }, need to wait for surface objects*/
	{ "iconify", iconify, 1, doc_iconify },
	{ "toggle_fullscreen", toggle_fullscreen, 1, doc_toggle_fullscreen },

	{ NULL, NULL }
};



    /*DOC*/ static char doc_pygame_display_MODULE[] =
    /*DOC*/    "Contains routines to work with the display. Mainly\n"
    /*DOC*/    "used for setting the display mode and updating the\n"
    /*DOC*/    "display surface.\n"
    /*DOC*/ ;

void initdisplay()
{
	PyObject *module, *dict, *apiobj;
	static void* c_api[PYGAMEAPI_DISPLAY_NUMSLOTS];

	PyType_Init(PyVidInfo_Type);


    /* create the module */
	module = Py_InitModule3("display", display_builtins, doc_pygame_display_MODULE);
	dict = PyModule_GetDict(module);

	/* export the c api */
	c_api[0] = &PyVidInfo_Type;
	c_api[1] = PyVidInfo_New;
	apiobj = PyCObject_FromVoidPtr(c_api, NULL);
	PyDict_SetItemString(dict, PYGAMEAPI_LOCAL_ENTRY, apiobj);

	/*imported needed apis*/
	import_pygame_base();
	import_pygame_rect();
	import_pygame_surface();
}

