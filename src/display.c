/*
    pygame - Python Game Library
    Copyright (C) 2000-2001  Pete Shinners

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
 *  pygame display module
 */
#define PYGAMEAPI_DISPLAY_INTERNAL
#include "pygame.h"

static char* icon_defaultname = "pygame_icon.bmp";
static PyObject* self_module = NULL;


staticforward PyTypeObject PyVidInfo_Type;
static PyObject* PyVidInfo_New(const SDL_VideoInfo* info);
static PyObject* DisplaySurfaceObject = NULL;
static int icon_was_set = 0;


#if 0
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
#endif



/* init routines */
static void display_autoquit(void)
{
        if(DisplaySurfaceObject)
        {
		((PySurfaceObject*)DisplaySurfaceObject)->surf = NULL;
                Py_DECREF(DisplaySurfaceObject);
                DisplaySurfaceObject = NULL;
        }
}

static PyObject* display_autoinit(PyObject* self, PyObject* arg)
{
	PyGame_RegisterQuit(display_autoquit);
	return PyInt_FromLong(1);
}


    /*DOC*/ static char doc_quit[] =
    /*DOC*/    "pygame.display.quit() -> None\n"
    /*DOC*/    "uninitialize the display module\n"
    /*DOC*/    "\n"
    /*DOC*/    "Manually uninitialize SDL's video subsystem. It is safe to call\n"
    /*DOC*/    "this if the video is currently not initialized.\n"
    /*DOC*/ ;

static PyObject* quit(PyObject* self, PyObject* arg)
{
	if(!PyArg_ParseTuple(arg, ""))
		return NULL;

	PyGame_Video_AutoQuit();
	display_autoquit();

	RETURN_NONE
}


    /*DOC*/ static char doc_init[] =
    /*DOC*/    "pygame.display.init() -> None\n"
    /*DOC*/    "initialize the display module\n"
    /*DOC*/    "\n"
    /*DOC*/    "Manually initialize SDL's video subsystem. Will raise an\n"
    /*DOC*/    "exception if it cannot be initialized. It is safe to call this\n"
    /*DOC*/    "function if the video has is currently initialized.\n"
    /*DOC*/ ;

static PyObject* init(PyObject* self, PyObject* arg)
{
/*we'll just ignore the args,
  i guess the user could pass anything they want,
  owell
	if(!PyArg_ParseTuple(arg, ""))
		return NULL;
*/
	if(!PyGame_Video_AutoInit())
		return RAISE(PyExc_SDLError, SDL_GetError());
	if(!display_autoinit(NULL, NULL))
		return NULL;

	RETURN_NONE
}



    /*DOC*/ static char doc_get_init[] =
    /*DOC*/    "pygame.display.get_init() -> bool\n"
    /*DOC*/    "get status of display module initialization\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns true if SDL's video system is currently intialized.\n"
    /*DOC*/ ;

static PyObject* get_init(PyObject* self, PyObject* arg)
{
	if(!PyArg_ParseTuple(arg, ""))
		return NULL;

	return PyInt_FromLong(SDL_WasInit(SDL_INIT_VIDEO)!=0);
}



    /*DOC*/ static char doc_get_active[] =
    /*DOC*/    "pygame.display.get_active() -> bool\n"
    /*DOC*/    "get state of display mode\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns true if the current display is active on the screen. This\n"
    /*DOC*/    "done with the call to pygame.display.set_mode(). It is\n"
    /*DOC*/    "potentially subject to the activity of a running window manager.\n"
    /*DOC*/    "\n"
    /*DOC*/    "Calling set_mode() will change all existing display surface\n"
    /*DOC*/    "to reference the new display mode. The old display surface will\n"
    /*DOC*/    "be lost after this call.\n"
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
	PyObject_DEL(self);
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
				 "	     blit_hw = %d, blit_hw_CC = %d, blit_hw_A = %d,\n"
				 "	     blit_sw = %d, blit_sw_CC = %d, blit_sw_A = %d,\n"
				 "	     bitsize  = %d, bytesize = %d,\n"
				 "	     masks =  (%d, %d, %d, %d),\n"
				 "	     shifts = (%d, %d, %d, %d),\n"
				 "	     losses =  (%d, %d, %d, %d)>\n",
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
	(hashfunc)NULL, 		/*hash*/
	(ternaryfunc)NULL,		/*call*/
	(reprfunc)NULL, 		/*str*/
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
#if 0
    /*DOC*/ static char docXX_set_driver[] =
    /*DOC*/    "pygame.display.set_driver(name) -> None\n"
    /*DOC*/    "override the default sdl video driver\n"
    /*DOC*/    "\n"
    /*DOC*/    "Changes the SDL environment to initialize with the given named\n"
    /*DOC*/    "videodriver. This can only be changed before the display is\n"
    /*DOC*/    "initialized. If this is not called, SDL will use it's default\n"
    /*DOC*/    "video driver, or the one in the environment variable\n"
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
#endif

    /*DOC*/ static char doc_get_driver[] =
    /*DOC*/    "pygame.display.get_driver() -> name\n"
    /*DOC*/    "get the current sdl video driver\n"
    /*DOC*/    "\n"
    /*DOC*/    "Once the display is initialized, this will return the name of the\n"
    /*DOC*/    "currently running video driver. There is no way to get a list of\n"
    /*DOC*/    "all the supported video drivers.\n"
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



    /*DOC*/ static char doc_Info[] =
    /*DOC*/    "pygame.display.Info() -> VidInfo\n"
    /*DOC*/    "get display capabilities and settings\n"
    /*DOC*/    "\n"
    /*DOC*/    "Gets a vidinfo object that contains information about the\n"
    /*DOC*/    "capabilities and current state of the video driver. This can be\n"
    /*DOC*/    "called before the display mode is set, to determine the current\n"
    /*DOC*/    "video mode of a display.\n"
    /*DOC*/    "You can print the VidInfo object to see all its members and values.\n"
    /*DOC*/ ;

static PyObject* Info(PyObject* self, PyObject* arg)
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
    /*DOC*/    "Returns a Surface object representing the current display. Will\n"
    /*DOC*/    "return None if called before the display mode is set.\n"
    /*DOC*/ ;

static PyObject* get_surface(PyObject* self, PyObject* arg)
{
	if(!PyArg_ParseTuple(arg, ""))
		return NULL;

	if(!DisplaySurfaceObject)
		RETURN_NONE

	Py_INCREF(DisplaySurfaceObject);
	return DisplaySurfaceObject;
}

    /*DOC*/ static char doc_gl_set_attribute[] =
    /*DOC*/    "pygame.display.gl_set_attribute(flag, value) -> None\n"
    /*DOC*/    "set special OPENGL attributes\n"
    /*DOC*/    "\n"
    /*DOC*/    "When calling pygame.display.set_mode() with the OPENGL flag,\n"
    /*DOC*/    "pygame automatically handles setting the opengl attributes like\n"
    /*DOC*/    "color and doublebuffering. OPENGL offers several other attributes\n"
    /*DOC*/    "you may want control over. Pass one of these attributes as the\n"
    /*DOC*/    "flag, and its appropriate value.\n"
    /*DOC*/    "\n"
    /*DOC*/    "This must be called before pygame.display.set_mode()\n"
    /*DOC*/    "\n"
    /*DOC*/    "The OPENGL flags are; GL_ALPHA_SIZE, GL_DEPTH_SIZE, GL_STENCIL_SIZE,\n"
    /*DOC*/    "GL_ACCUM_RED_SIZE, GL_ACCUM_GREEN_SIZE, GL_ACCUM_BLUE_SIZE, GL_ACCUM_ALPHA_SIZE\n"
    /*DOC*/    "GL_MULTISAMPLEBUFFERS, GL_MULTISAMPLESAMPLES, GL_STEREO\n"
    /*DOC*/ ;

static PyObject* gl_set_attribute(PyObject* self, PyObject* arg)
{
        int flag, value, result;

	VIDEO_INIT_CHECK();

    	if(!PyArg_ParseTuple(arg, "ii", &flag, &value))
		return NULL;
        if(flag == -1) /*an undefined/unsupported val, ignore*/
            RETURN_NONE

	result = SDL_GL_SetAttribute(flag, value);
        if(result == -1)
            return RAISE(PyExc_SDLError, SDL_GetError());
        RETURN_NONE
}


    /*DOC*/ static char doc_gl_get_attribute[] =
    /*DOC*/    "pygame.display.gl_get_attribute(flag) -> value\n"
    /*DOC*/    "get special OPENGL attributes\n"
    /*DOC*/    "\n"
    /*DOC*/    "After calling pygame.display.set_mode() with the OPENGL flag\n"
    /*DOC*/    "you will likely want to check the value of any special opengl\n"
    /*DOC*/    "attributes you requested. You will not always get what you\n"
    /*DOC*/    "requested.\n"
    /*DOC*/    "\n"
    /*DOC*/    "See pygame.display.gl_set_attribute() for a list of flags.\n"
    /*DOC*/    "\n"
    /*DOC*/    "The OPENGL flags are; GL_ALPHA_SIZE, GL_DEPTH_SIZE, GL_STENCIL_SIZE,\n"
    /*DOC*/    "GL_ACCUM_RED_SIZE, GL_ACCUM_GREEN_SIZE, GL_ACCUM_BLUE_SIZE, GL_ACCUM_ALPHA_SIZE,\n"
    /*DOC*/    "GL_RED_SIZE, GL_GREEN_SIZE, GL_BLUE_SIZE, GL_DEPTH_SIZE\n"
    /*DOC*/ ;

static PyObject* gl_get_attribute(PyObject* self, PyObject* arg)
{
        int flag, value, result;

	VIDEO_INIT_CHECK();

	if(!PyArg_ParseTuple(arg, "i", &flag))
		return NULL;

	result = SDL_GL_GetAttribute(flag, &value);
        if(result == -1)
            return RAISE(PyExc_SDLError, SDL_GetError());

        return PyInt_FromLong(value);
}



    /*DOC*/ static char doc_set_mode[] =
    /*DOC*/    "pygame.display.set_mode(size, [flags, [depth]]) -> Surface\n"
    /*DOC*/    "set the display mode\n"
    /*DOC*/    "\n"
    /*DOC*/    "Sets the current display mode. If calling this after the mode has\n"
    /*DOC*/    "already been set, this will change the display mode to the\n"
    /*DOC*/    "desired type. Sometimes an exact match for the requested video\n"
    /*DOC*/    "mode is not available. In this case SDL will try to find the\n"
    /*DOC*/    "closest match and work with that instead.\n"
    /*DOC*/    "\n"
    /*DOC*/    "The size is a 2-number-sequence containing the width and height\n"
    /*DOC*/    "of the desired display mode. Flags represents a set of different\n"
    /*DOC*/    "options for the new display mode. If omitted or given as 0, it\n"
    /*DOC*/    "will default to a simple software window. You can mix several\n"
    /*DOC*/    "flags together with the bitwise-or (|) operator. Possible flags\n"
    /*DOC*/    "are HWSURFACE (or the value 1), HWPALETTE, DOUBLEBUF, and/or\n"
    /*DOC*/    "FULLSCREEN. There are other flags available but these are the\n"
    /*DOC*/    "most usual. A full list of flags can be found in the pygame\n"
    /*DOC*/    "documentation.\n"
    /*DOC*/    "\n"
    /*DOC*/    "The optional depth arguement is the requested bits\n"
    /*DOC*/    "per pixel. It will usually be left omitted, in which case the\n"
    /*DOC*/    "display will use the best/fastest pixel depth available.\n"
    /*DOC*/    "\n"
    /*DOC*/    "You can create an OpenGL surface (for use with PyOpenGL)\n"
    /*DOC*/    "by passing the OPENGL flag. You will likely want to use the\n"
    /*DOC*/    "DOUBLEBUF flag when using OPENGL. In which case, the flip()\n"
    /*DOC*/    "function will perform the GL buffer swaps. When you are using\n"
    /*DOC*/    "an OPENGL video mode, you will not be able to perform most of the\n"
    /*DOC*/    "pygame drawing functions (fill, set_at, etc) on the display surface.\n"
    /*DOC*/ ;

static PyObject* set_mode(PyObject* self, PyObject* arg)
{
	SDL_Surface* surf;
	int flags = SDL_SWSURFACE, depth = 0;
	int w, h, hasbuf;
	char *title, *icontitle;

	if(!PyArg_ParseTuple(arg, "(ii)|ii", &w, &h, &flags, &depth))
		return NULL;

	if(!SDL_WasInit(SDL_INIT_VIDEO))
	{
		/*note SDL works special like this too*/
		if(!init(NULL, NULL))
			return NULL;
	}

	if(flags & SDL_OPENGL)
	{
		if(flags & SDL_DOUBLEBUF)
		{
			flags &= ~SDL_DOUBLEBUF;
			SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
		}
		else
			SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 0);
		if(depth)
			SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, depth);
		surf = SDL_SetVideoMode(w, h, depth, flags);
		if(!surf)
			return RAISE(PyExc_SDLError, SDL_GetError());

		SDL_GL_GetAttribute(SDL_GL_DOUBLEBUFFER, &hasbuf);
		if(hasbuf)
		{
			surf->flags |= SDL_DOUBLEBUF;
		}
	}
	else
	{
		if(!depth)
			flags |= SDL_ANYFORMAT;
		Py_BEGIN_ALLOW_THREADS
		surf = SDL_SetVideoMode(w, h, depth, flags);
		Py_END_ALLOW_THREADS
		if(!surf)
			return RAISE(PyExc_SDLError, SDL_GetError());
	}
	SDL_WM_GetCaption(&title, &icontitle);
	if(!title || !*title)
		SDL_WM_SetCaption("pygame window", "pygame");

	/*probably won't do much, but can't hurt, and might help*/
	SDL_PumpEvents();

	if(DisplaySurfaceObject)
		((PySurfaceObject*)DisplaySurfaceObject)->surf = surf;
	else
		DisplaySurfaceObject = PySurface_New(surf);


#if !defined(darwin)
	if(!icon_was_set)
	{
		SDL_Surface* icon;
		char* iconpath;
		char* path = PyModule_GetFilename(self_module);
		icon_was_set = 1;
		if(!path)
			PyErr_Clear();
		else
		{
			char* end = strstr(path, "display.");
			if(end)
			{
				iconpath = PyMem_Malloc(strlen(path) + 20);
				if(iconpath)
				{
					strcpy(iconpath, path);
					end = strstr(iconpath, "display.");
					strcpy(end, icon_defaultname);

					icon = SDL_LoadBMP(iconpath);
					if(icon)
					{
						SDL_SetColorKey(icon, SDL_SRCCOLORKEY, 0);
						SDL_WM_SetIcon(icon, NULL);
						SDL_FreeSurface(icon);
					}
					PyMem_Free(iconpath);
				}
			}
		}
	}
#endif
	Py_INCREF(DisplaySurfaceObject);
	return DisplaySurfaceObject;
}



    /*DOC*/ static char doc_mode_ok[] =
    /*DOC*/    "pygame.display.mode_ok(size, [flags, [depth]]) -> int\n"
    /*DOC*/    "query a specific display mode\n"
    /*DOC*/    "\n"
    /*DOC*/    "This uses the same arguments as the call to\n"
    /*DOC*/    "pygame.display.set_mode(). It is used to determine if a requested\n"
    /*DOC*/    "display mode is available. It will return 0 if the requested mode\n"
    /*DOC*/    "is not possible. Otherwise it will return the best and closest\n"
    /*DOC*/    "matching bit depth for the mode requested.\n"
    /*DOC*/    "\n"
    /*DOC*/    "The size is a 2-number-sequence containing the width and height\n"
    /*DOC*/    "of the desired display mode. Flags represents a set of different\n"
    /*DOC*/    "options for the display mode. If omitted or given as 0, it will\n"
    /*DOC*/    "default to a simple software window. You can mix several flags\n"
    /*DOC*/    "together with the bitwise-or (|) operator. Possible flags are\n"
    /*DOC*/    "HWSURFACE (or the value 1), HWPALETTE, DOUBLEBUF, and/or\n"
    /*DOC*/    "FULLSCREEN. There are other flags available but these are the\n"
    /*DOC*/    "most usual. A full list of flags can be found in the SDL\n"
    /*DOC*/    "documentation. The optional depth arguement is the requested bits\n"
    /*DOC*/    "per pixel. It will usually be left omitted, in which case the\n"
    /*DOC*/    "display will use the best/fastest pixel depth available.\n"
    /*DOC*/ ;

static PyObject* mode_ok(PyObject* self, PyObject* args)
{
	int flags=SDL_SWSURFACE, depth=0;
	int w, h;

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
    /*DOC*/    "This function returns a list of possible dimensions for a\n"
    /*DOC*/    "specified color depth. The return value will be an empty list if\n"
    /*DOC*/    "no display modes are available with the given arguments. A return\n"
    /*DOC*/    "value of -1 means that any requested resolution should work (this\n"
    /*DOC*/    "is likely the case for windowed modes). Mode sizes are sorted\n"
    /*DOC*/    "from biggest to smallest.\n"
    /*DOC*/    "\n"
    /*DOC*/    "If depth is not passed or 0, SDL will choose the current/best\n"
    /*DOC*/    "color depth for the display. You will usually want to pass\n"
    /*DOC*/    "FULLSCREEN when using the flags, if flags is omitted, FULLSCREEN\n"
    /*DOC*/    "is the default.\n"
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
    /*DOC*/    "This will update the contents of the entire display. If your\n"
    /*DOC*/    "display mode is using the flags HWSURFACE and DOUBLEBUF, this\n"
    /*DOC*/    "will wait for a vertical retrace and swap the surfaces. If you\n"
    /*DOC*/    "are using a different type of display mode, it will simply update\n"
    /*DOC*/    "the entire contents of the surface.\n"
    /*DOC*/    "\n"
    /*DOC*/    "When using an OPENGL display mode this will perform a gl buffer swap.\n"
    /*DOC*/ ;

static PyObject* flip(PyObject* self, PyObject* arg)
{
	SDL_Surface* screen;
	int status = 0;

	if(!PyArg_ParseTuple(arg, ""))
		return NULL;

	VIDEO_INIT_CHECK();

	screen = SDL_GetVideoSurface();
	if(!screen)
		return RAISE(PyExc_SDLError, "Display mode not set");

	Py_BEGIN_ALLOW_THREADS
	if(screen->flags & SDL_OPENGL)
		SDL_GL_SwapBuffers();
	else
		status = SDL_Flip(screen) == -1;
	Py_END_ALLOW_THREADS

	if(status == -1)
		return RAISE(PyExc_SDLError, SDL_GetError());

	RETURN_NONE
}


/*BAD things happen when out-of-bound rects go to updaterect*/
static SDL_Rect* screencroprect(GAME_Rect* r, int w, int h, SDL_Rect* cur)
{
	if(r->x > w || r->y > h || (r->x + r->w) <= 0 || (r->y + r->h) <= 0)
                return 0;
        else
	{
		int right = min(r->x + r->w, w);
		int bottom = min(r->y + r->h, h);
		cur->x = (short)max(r->x, 0);
		cur->y = (short)max(r->y, 0);
		cur->w = (unsigned short)right - cur->x;
		cur->h = (unsigned short)bottom - cur->y;
	}
	return cur;
}

    /*DOC*/ static char doc_update[] =
    /*DOC*/    "pygame.display.update([rectstyle]) -> None\n"
    /*DOC*/    "update an area of the display\n"
    /*DOC*/    "\n"
    /*DOC*/    "This call will update a section (or sections) of the display\n"
    /*DOC*/    "screen. You must update an area of your display when you change\n"
    /*DOC*/    "its contents. If passed with no arguments, this will update the\n"
    /*DOC*/    "entire display surface. If you have many rects that need\n"
    /*DOC*/    "updating, it is best to combine them into a sequence and pass\n"
    /*DOC*/    "them all at once. This call will accept a sequence of rectstyle\n"
    /*DOC*/    "arguments. Any None's in the list will be ignored.\n"
    /*DOC*/    "\n"
    /*DOC*/    "This call cannot be used on OPENGL displays, and will generate\n"
    /*DOC*/    "an exception.\n"
    /*DOC*/ ;

static PyObject* update(PyObject* self, PyObject* arg)
{
	SDL_Surface* screen;
	GAME_Rect *gr, temp = {0};
	int wide, high;
	PyObject* obj;

	VIDEO_INIT_CHECK();

	screen = SDL_GetVideoSurface();
	if(!screen)
		return RAISE(PyExc_SDLError, SDL_GetError());
	wide = screen->w;
	high = screen->h;
	if(screen->flags & SDL_OPENGL)
		return RAISE(PyExc_SDLError, "Cannot update an OPENGL display");

	/*determine type of argument we got*/
	if(PyTuple_Size(arg) == 0)
        {
            SDL_UpdateRect(screen, 0, 0, 0, 0);
            RETURN_NONE
        }
	else
	{
		obj = PyTuple_GET_ITEM(arg, 0);
		if(obj == Py_None)
		{
			gr = &temp;
		}
		else
		{
			gr = GameRect_FromObject(arg, &temp);
			if(!gr)
				PyErr_Clear();
			else if(gr != &temp)
			{
				memcpy(&temp, gr, sizeof(temp));
				gr = &temp;
			}
		}
	}

        if(gr)
        {
                SDL_Rect sdlr;
                if(screencroprect(gr, wide, high, &sdlr))
                        SDL_UpdateRect(screen, sdlr.x, sdlr.y, sdlr.w, sdlr.h);
        }
        else
        {
                PyObject* seq;
                PyObject* r;
                int loop, num, count;
                SDL_Rect* rects;
                if(PyTuple_Size(arg) != 1)
                        return RAISE(PyExc_ValueError, "update requires a rectstyle or sequence of recstyles");
                seq = PyTuple_GET_ITEM(arg, 0);
                if(!seq || !PySequence_Check(seq))
                        return RAISE(PyExc_ValueError, "update requires a rectstyle or sequence of recstyles");

                num = PySequence_Length(seq);
                rects = PyMem_New(SDL_Rect, num);
                if(!rects) return NULL;
                count = 0;
                for(loop = 0; loop < num; ++loop)
                {
                        SDL_Rect* cur_rect = (rects + count);

                        /*get rect from the sequence*/
                        r = PySequence_GetItem(seq, loop);
                        if(r == Py_None)
                        {
                                Py_DECREF(r);
                                continue;
                        }
                        gr = GameRect_FromObject(r, &temp);
                        Py_XDECREF(r);
                        if(!gr)
                        {
                                PyMem_Free((char*)rects);
                                return RAISE(PyExc_ValueError, "update_rects requires a single list of rects");
                        }

                        if(gr->w < 1 && gr->h < 1)
                                continue;

                        /*bail out if rect not onscreen*/
                        if(!screencroprect(gr, wide, high, cur_rect))
                                continue;

                        ++count;
                }

                if(count)
                    SDL_UpdateRects(screen, count, rects);
                PyMem_Free((char*)rects);
	}
	RETURN_NONE
}

    /*DOC*/ static char doc_set_palette[] =
    /*DOC*/    "pygame.display.set_palette([[r, g, b], ...]) -> None\n"
    /*DOC*/    "set the palette\n"
    /*DOC*/    "\n"
    /*DOC*/    "Displays with a HWPALETTE have two palettes. The display Surface\n"
    /*DOC*/    "palette and the visible 'onscreen' palette.\n"
    /*DOC*/    "\n"
    /*DOC*/    "This will change the video display's visible colormap. It does\n"
    /*DOC*/    "not effect the display Surface's base palette, only how it is\n"
    /*DOC*/    "displayed. Setting the palette for the display Surface will\n"
    /*DOC*/    "override this visible palette. Also passing no args will reset\n"
    /*DOC*/    "the display palette back to the Surface's palette.\n"
    /*DOC*/    "\n"
    /*DOC*/    "You can pass an incomplete list of RGB values, and\n"
    /*DOC*/    "this will only change the first colors in the palette.\n"
    /*DOC*/ ;

static PyObject* set_palette(PyObject* self, PyObject* args)
{
	SDL_Surface* surf;
	SDL_Palette* pal;
	SDL_Color* colors;
	PyObject* list, *item = NULL;
	int i, len;
	int r, g, b;

	VIDEO_INIT_CHECK();
	if(!PyArg_ParseTuple(args, "|O", &list))
		return NULL;
	surf = SDL_GetVideoSurface();
	if(!surf)
		return RAISE(PyExc_SDLError, "No display mode is set");
	pal = surf->format->palette;
	if(surf->format->BytesPerPixel != 1 || !pal)
		return RAISE(PyExc_SDLError, "Display mode is not colormapped");

	if(!list)
	{
		colors = pal->colors;
		len = pal->ncolors;
		SDL_SetPalette(surf, SDL_PHYSPAL, colors, 0, len);
		RETURN_NONE
	}


	if(!PySequence_Check(list))
		return RAISE(PyExc_ValueError, "Argument must be a sequence type");

	len = min(pal->ncolors, PySequence_Length(list));

	colors = (SDL_Color*)malloc(len * sizeof(SDL_Color));
	if(!colors)
		return NULL;

	for(i = 0; i < len; i++)
	{
		item = PySequence_GetItem(list, i);

		if(!PySequence_Check(item) || PySequence_Length(item) != 3)
		{
			Py_DECREF(item);
			free((char*)colors);
			return RAISE(PyExc_TypeError, "takes a sequence of sequence of RGB");
		}
		if(!IntFromObjIndex(item, 0, &r) || !IntFromObjIndex(item, 1, &g) || !IntFromObjIndex(item, 2, &b))
			return RAISE(PyExc_TypeError, "RGB sequence must contain numeric values");

		colors[i].r = (unsigned char)r;
		colors[i].g = (unsigned char)g;
		colors[i].b = (unsigned char)b;

		Py_DECREF(item);
	}

	SDL_SetPalette(surf, SDL_PHYSPAL, colors, 0, len);

	free((char*)colors);
	RETURN_NONE
}


    /*DOC*/ static char doc_set_gamma[] =
    /*DOC*/    "pygame.display.set_gamma(r, [g, b]) -> bool\n"
    /*DOC*/    "change the brightness of the display\n"
    /*DOC*/    "\n"
    /*DOC*/    "Sets the display gamma to the given amounts. If green and blue\n"
    /*DOC*/    "are ommitted, the red value will be used for all three colors.\n"
    /*DOC*/    "The color arguments are floating point values with 1.0 being the\n"
    /*DOC*/    "normal value. If you are using a display mode with a hardware\n"
    /*DOC*/    "palette, this will simply update the palette you are using. Not\n"
    /*DOC*/    "all hardware supports gamma. The return value will be true on\n"
    /*DOC*/    "success.\n"
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

static int convert_to_uint16(PyObject* python_array, Uint16* c_uint16_array)
{
        int i;
        PyObject* item;

	if (!c_uint16_array) {
	        RAISE(PyExc_RuntimeError, "Memory not allocated for c_uint16_array.");
		return 0;
	}

	if (!PySequence_Check(python_array))
        {
	        RAISE(PyExc_TypeError, "Array must be sequence type");
	        return 0;
        }

	if (PySequence_Size(python_array) != 256)
        {
		RAISE(PyExc_ValueError, "gamma ramp must be 256 elements long");
                return 0;
        }
	for (i=0; i<256; i++)
        {
                item = PySequence_GetItem(python_array, i);
                if(!PyInt_Check(item))
                {
		    RAISE(PyExc_ValueError, "gamma ramp must contain integer elements");
                    return 0;
                }
		c_uint16_array[i] = (Uint16)PyInt_AsLong(item);
        }
	return 1;
}

    /*DOC*/ static char doc_set_gamma_ramp[] =
    /*DOC*/    "pygame.display.set_gamma_ramp(r, g, b) -> bool\n"
    /*DOC*/    "advanced control over the display gamma ramps\n"
    /*DOC*/    "\n"
    /*DOC*/    "Pass three sequences with 256 elements. Each element must be a\n"
    /*DOC*/    "'16bit' unsigned integer value. This is from 0 to 65536.\n"
    /*DOC*/    "If you are using a display mode with a hardware\n"
    /*DOC*/    "palette, this will simply update the palette you are using.\n"
    /*DOC*/    "Not all hardware supports gamma. The return value will be\n"
    /*DOC*/    "true on success.\n";

static PyObject* set_gamma_ramp(PyObject* self, PyObject* arg)
{
 	Uint16 *r, *g, *b;

	int result;

	r = (Uint16 *)malloc(256 * sizeof(Uint16));
	if (!r)
	        return NULL;
	g = (Uint16 *)malloc(256 * sizeof(Uint16));
	if (!g)
        {
                free(r);
	        return NULL;
        }
	b = (Uint16 *)malloc(256 * sizeof(Uint16));
	if (!b)
        {
                free(r);
                free(g);
	        return NULL;
        }

	if(!PyArg_ParseTuple(arg, "O&O&O&",
			     convert_to_uint16, r,
			     convert_to_uint16, g,
			     convert_to_uint16, b))
        {
                free(r); free(g); free(b);
	        return NULL;
        }

        VIDEO_INIT_CHECK();

	result = SDL_SetGammaRamp(r, g, b);

	free((char*)r);
	free((char*)g);
	free((char*)b);

	return PyInt_FromLong(result == 0);
}

    /*DOC*/ static char doc_set_caption[] =
    /*DOC*/    "pygame.display.set_caption(title, [icontitle]) -> None\n"
    /*DOC*/    "changes the title of the window\n"
    /*DOC*/    "\n"
    /*DOC*/    "If the display has a window title, this routine will change the\n"
    /*DOC*/    "name on the window. Some environments support a shorter icon\n"
    /*DOC*/    "title to be used when the display is minimized. If icontitle is\n"
    /*DOC*/    "omittied it will be the same as caption title.\n"
    /*DOC*/ ;

static PyObject* set_caption(PyObject* self, PyObject* arg)
{
	char* title, *icontitle=NULL;

	if(!PyArg_ParseTuple(arg, "s|s", &title, &icontitle))
		return NULL;

	if(!icontitle)
		icontitle = title;

	SDL_WM_SetCaption(title, icontitle);

	RETURN_NONE
}



    /*DOC*/ static char doc_get_caption[] =
    /*DOC*/    "pygame.display.get_caption() -> title, icontitle\n"
    /*DOC*/    "get the current title of the window\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns the current title and icontitle for the display window.\n"
    /*DOC*/ ;

static PyObject* get_caption(PyObject* self, PyObject* arg)
{
	char* title, *icontitle;

	if(!PyArg_ParseTuple(arg, ""))
		return NULL;

	SDL_WM_GetCaption(&title, &icontitle);

	if(title && *title)
		return Py_BuildValue("(ss)", title, icontitle);

	return Py_BuildValue("()");
}


    /*DOC*/ static char doc_set_icon[] =
    /*DOC*/    "pygame.display.set_icon(Surface) -> None\n"
    /*DOC*/    "changes the window manager icon for the window\n"
    /*DOC*/    "\n"
    /*DOC*/    "Sets the runtime icon that your system uses to decorate\n"
    /*DOC*/    "the program window. It is also used when the application\n"
    /*DOC*/    "is iconified and in the window frame.\n"
    /*DOC*/    "\n"
    /*DOC*/    "You likely want this to be a smaller image, a size that\n"
    /*DOC*/    "your system window manager will be able to deal with. It will\n"
    /*DOC*/    "also use the Surface colorkey if available.\n"
    /*DOC*/    "\n"
    /*DOC*/    "Some window managers on X11 don't allow you to change the \n"
    /*DOC*/    "icon after the window has been shown the first time.\n"
    /*DOC*/ ;

static PyObject* set_icon(PyObject* self, PyObject* arg)
{
	PyObject* surface;
	SDL_Surface* surf;
	if(!PyArg_ParseTuple(arg, "O!", &PySurface_Type, &surface))
		return NULL;

	surf = PySurface_AsSurface(surface);
	PySurface_Lock(surface);
	SDL_WM_SetIcon(surf, NULL);
	PySurface_Unlock(surface);

	icon_was_set = 1;
	RETURN_NONE
}


    /*DOC*/ static char doc_iconify[] =
    /*DOC*/    "pygame.display.iconify() -> bool\n"
    /*DOC*/    "minimize the display window\n"
    /*DOC*/    "\n"
    /*DOC*/    "Tells the window manager (if available) to minimize the\n"
    /*DOC*/    "application. The call will return true if successful. You will\n"
    /*DOC*/    "receive an APPACTIVE event on the event queue when the window has\n"
    /*DOC*/    "been minimized.\n"
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
    /*DOC*/    "Tells the window manager (if available) to switch between\n"
    /*DOC*/    "windowed and fullscreen mode. If available and successfull, will\n"
    /*DOC*/    "return true. Note, there is currently limited platform support\n"
    /*DOC*/    "for this call.\n"
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
	{ "__PYGAMEinit__", display_autoinit, 1, doc_init },
	{ "init", init, 1, doc_init },
	{ "quit", quit, 1, doc_quit },
	{ "get_init", get_init, 1, doc_get_init },
	{ "get_active", get_active, 1, doc_get_active },

/*	{ "set_driver", set_driver, 1, doc_set_driver },*/
	{ "get_driver", get_driver, 1, doc_get_driver },
	{ "Info", Info, 1, doc_Info },
	{ "get_surface", get_surface, 1, doc_get_surface },

	{ "set_mode", set_mode, 1, doc_set_mode },
	{ "mode_ok", mode_ok, 1, doc_mode_ok },
	{ "list_modes", list_modes, 1, doc_list_modes },

        { "flip", flip, 1, doc_flip }, { "update", update, 1, doc_update },

	{ "set_palette", set_palette, 1, doc_set_palette },
	{ "set_gamma", set_gamma, 1, doc_set_gamma },
	{ "set_gamma_ramp", set_gamma_ramp, 1, doc_set_gamma_ramp },

	{ "set_caption", set_caption, 1, doc_set_caption },
	{ "get_caption", get_caption, 1, doc_get_caption },
	{ "set_icon", set_icon, 1, doc_set_icon },

	{ "iconify", iconify, 1, doc_iconify },
	{ "toggle_fullscreen", toggle_fullscreen, 1, doc_toggle_fullscreen },

	{ "gl_set_attribute", gl_set_attribute, 1, doc_gl_set_attribute },
	{ "gl_get_attribute", gl_get_attribute, 1, doc_gl_get_attribute },

	{ NULL, NULL }
};



    /*DOC*/ static char doc_pygame_display_MODULE[] =
    /*DOC*/    "Contains routines to work with the display. Mainly used for\n"
    /*DOC*/    "setting the display mode and updating the display surface.\n"
    /*DOC*/    "\n"
    /*DOC*/    "Pygame offers a fairly simple interface to the display buffer.\n"
    /*DOC*/    "The buffer is represented as an offscreen surface to which you\n"
    /*DOC*/    "can write directly. If you want the screen to show what you have\n"
    /*DOC*/    "written, the pygame.display.update() function will guarantee the\n"
    /*DOC*/    "the desired portion of the screen is updated. You can call\n"
    /*DOC*/    "pygame.display.flip() to update the entire screen, and also flip\n"
    /*DOC*/    "a hardware surface created with DOUBLEBUF.\n"
    /*DOC*/    "\n"
    /*DOC*/    "There are a number of ways to start the video display. The\n"
    /*DOC*/    "easiest way is to pick a common screen resolution and depth and\n"
    /*DOC*/    "just initialize the video, checking for exceptions. You will\n"
    /*DOC*/    "probably get what you want, but pygame may be emulating your\n"
    /*DOC*/    "requested mode and converting the display on update (this is not\n"
    /*DOC*/    "the fastest method). When calling pygame.display.set_mode() with\n"
    /*DOC*/    "the bit depth omitted or set to zero, pygame will determine the\n"
    /*DOC*/    "best video mode available and set to that. You can also query for\n"
    /*DOC*/    "more information on video modes with pygame.display.mode_ok(),\n"
    /*DOC*/    "pygame.display.list_modes(), and\n"
    /*DOC*/    "pygame.display.Info().\n"
    /*DOC*/    "\n"
    /*DOC*/    "When using a display depth other than what you graphic resources\n"
    /*DOC*/    "may be saved at, it is best to call the Surface.convert() routine\n"
    /*DOC*/    "to convert them to the same format as the display, this will\n"
    /*DOC*/    "result in the fastest blitting.\n"
    /*DOC*/    "\n"
    /*DOC*/    "Pygame currently supports any but depth >= 8 bits per pixl. 8bpp\n"
    /*DOC*/    "formats are considered to be 8-bit palettized modes, while 12,\n"
    /*DOC*/    "15, 16, 24, and 32 bits per pixel are considered 'packed pixel'\n"
    /*DOC*/    "modes, meaning each pixel contains the RGB color componsents\n"
    /*DOC*/    "packed into the bits of the pixel.\n"
    /*DOC*/    "\n"
    /*DOC*/    "After you have initialized your video mode, you can take the\n"
    /*DOC*/    "surface that was returned and write to it like any other Surface\n"
    /*DOC*/    "object. Be sure to call update() or flip() to keep what is on the\n"
    /*DOC*/    "screen synchronized with what is on the surface. Be sure not to call\n"
    /*DOC*/    "display routines that modify the display surface while it is locked.\n"
    /*DOC*/ ;

PYGAME_EXPORT
void initdisplay(void)
{
	PyObject *module, *dict, *apiobj;
	static void* c_api[PYGAMEAPI_DISPLAY_NUMSLOTS];

	PyType_Init(PyVidInfo_Type);


    /* create the module */
	module = Py_InitModule3("display", display_builtins, doc_pygame_display_MODULE);
	dict = PyModule_GetDict(module);
	self_module = module;

	/* export the c api */
	c_api[0] = &PyVidInfo_Type;
	c_api[1] = PyVidInfo_New;
	apiobj = PyCObject_FromVoidPtr(c_api, NULL);
	PyDict_SetItemString(dict, PYGAMEAPI_LOCAL_ENTRY, apiobj);
	Py_DECREF(apiobj);

	/*imported needed apis*/
	import_pygame_base();
	import_pygame_rect();
	import_pygame_surface();
}

