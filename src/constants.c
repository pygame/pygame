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

#include "pygame.h"



/* macros used to create each constant */
#define DEC_CONST(x)  PyModule_AddIntConstant(module, #x, SDL_##x);
#define DEC_CONSTK(x) PyModule_AddIntConstant(module, #x, SDL##x);
#define DEC_CONSTN(x) PyModule_AddIntConstant(module, #x, x);


static PyMethodDef builtins[] =
{
	{NULL}
};


    /*DOC*/ static char doc_pygame_constants_MODULE[] =
    /*DOC*/    "These constants are defined by SDL, and needed in pygame. Note\n"
    /*DOC*/    "that many of the flags for SDL are not needed in pygame, and are\n"
    /*DOC*/    "not included here. These constants are generally accessed from\n"
    /*DOC*/    "the pygame.locals module. This module is automatically placed in\n"
    /*DOC*/    "the pygame namespace, but you will usually want to place them\n"
    /*DOC*/    "directly into your module's namespace with the following command,\n"
    /*DOC*/    "'from pygame.locals import *'.\n"
    /*DOC*/ ;

PYGAME_EXPORT
void initconstants(void)
{
	PyObject* module;

	PyGAME_C_API[0] = PyGAME_C_API[0]; /*this cleans up compiler warning*/


	module = Py_InitModule3("constants", builtins, doc_pygame_constants_MODULE);

	DEC_CONST(SWSURFACE);
	DEC_CONST(HWSURFACE);
	DEC_CONST(RESIZABLE);
	DEC_CONST(ASYNCBLIT);
	DEC_CONST(OPENGL);
	DEC_CONST(OPENGLBLIT)
	DEC_CONST(ANYFORMAT);
	DEC_CONST(HWPALETTE);
	DEC_CONST(DOUBLEBUF);
	DEC_CONST(FULLSCREEN);
	DEC_CONST(HWACCEL);
	DEC_CONST(SRCCOLORKEY);
	DEC_CONST(RLEACCELOK);
	DEC_CONST(RLEACCEL);
	DEC_CONST(SRCALPHA);
	DEC_CONST(PREALLOC);
#if SDL_VERSIONNUM(1, 1, 8) <= SDL_VERSIONNUM(SDL_MAJOR_VERSION, SDL_MINOR_VERSION, SDL_PATCHLEVEL) 
	DEC_CONST(NOFRAME);
#else
	PyModule_AddIntConstant(module, "NOFRAME", 0);
#endif

	DEC_CONST(GL_RED_SIZE);
	DEC_CONST(GL_GREEN_SIZE);
	DEC_CONST(GL_BLUE_SIZE);
	DEC_CONST(GL_ALPHA_SIZE);
	DEC_CONST(GL_BUFFER_SIZE);
	DEC_CONST(GL_DOUBLEBUFFER);
	DEC_CONST(GL_DEPTH_SIZE);
	DEC_CONST(GL_STENCIL_SIZE);
	DEC_CONST(GL_ACCUM_RED_SIZE);
	DEC_CONST(GL_ACCUM_GREEN_SIZE);
	DEC_CONST(GL_ACCUM_BLUE_SIZE);
	DEC_CONST(GL_ACCUM_ALPHA_SIZE);

	DEC_CONSTN(AUDIO_U8);
	DEC_CONSTN(AUDIO_S8);
	DEC_CONSTN(AUDIO_U16LSB);
	DEC_CONSTN(AUDIO_S16LSB);
	DEC_CONSTN(AUDIO_U16MSB);
	DEC_CONSTN(AUDIO_S16MSB);
	DEC_CONSTN(AUDIO_U16);
	DEC_CONSTN(AUDIO_S16);
	DEC_CONSTN(AUDIO_U16SYS);
	DEC_CONSTN(AUDIO_S16SYS);
	
	DEC_CONST(NOEVENT);
	DEC_CONST(ACTIVEEVENT);
	DEC_CONST(KEYDOWN);
	DEC_CONST(KEYUP);
	DEC_CONST(MOUSEMOTION);
	DEC_CONST(MOUSEBUTTONDOWN);
	DEC_CONST(MOUSEBUTTONUP);
	DEC_CONST(JOYAXISMOTION);
	DEC_CONST(JOYBALLMOTION);
	DEC_CONST(JOYHATMOTION);
	DEC_CONST(JOYBUTTONDOWN);
	DEC_CONST(JOYBUTTONUP);
	DEC_CONST(VIDEORESIZE);
	DEC_CONST(QUIT);
	DEC_CONST(SYSWMEVENT);
	DEC_CONST(USEREVENT);
	DEC_CONST(NUMEVENTS);

	DEC_CONST(HAT_CENTERED);
	DEC_CONST(HAT_UP);
	DEC_CONST(HAT_RIGHTUP);
	DEC_CONST(HAT_RIGHT);
	DEC_CONST(HAT_RIGHTDOWN);
	DEC_CONST(HAT_DOWN);
	DEC_CONST(HAT_LEFTDOWN);
	DEC_CONST(HAT_LEFT);
	DEC_CONST(HAT_LEFTUP);


	DEC_CONSTK(K_UNKNOWN);
	DEC_CONSTK(K_FIRST);
	DEC_CONSTK(K_BACKSPACE);
	DEC_CONSTK(K_TAB);
	DEC_CONSTK(K_CLEAR);
	DEC_CONSTK(K_RETURN);
	DEC_CONSTK(K_PAUSE);
	DEC_CONSTK(K_ESCAPE);
	DEC_CONSTK(K_SPACE);
	DEC_CONSTK(K_EXCLAIM);
	DEC_CONSTK(K_QUOTEDBL);
	DEC_CONSTK(K_HASH);
	DEC_CONSTK(K_DOLLAR);
	DEC_CONSTK(K_AMPERSAND);
	DEC_CONSTK(K_QUOTE);
	DEC_CONSTK(K_LEFTPAREN);
	DEC_CONSTK(K_RIGHTPAREN);
	DEC_CONSTK(K_ASTERISK);
	DEC_CONSTK(K_PLUS);
	DEC_CONSTK(K_COMMA);
	DEC_CONSTK(K_MINUS);
	DEC_CONSTK(K_PERIOD);
	DEC_CONSTK(K_SLASH);
	DEC_CONSTK(K_0);
	DEC_CONSTK(K_1);
	DEC_CONSTK(K_2);
	DEC_CONSTK(K_3);
	DEC_CONSTK(K_4);
	DEC_CONSTK(K_5);
	DEC_CONSTK(K_6);
	DEC_CONSTK(K_7);
	DEC_CONSTK(K_8);
	DEC_CONSTK(K_9);
	DEC_CONSTK(K_COLON);
	DEC_CONSTK(K_SEMICOLON);
	DEC_CONSTK(K_LESS);
	DEC_CONSTK(K_EQUALS);
	DEC_CONSTK(K_GREATER);
	DEC_CONSTK(K_QUESTION);
	DEC_CONSTK(K_AT);
	DEC_CONSTK(K_LEFTBRACKET);
	DEC_CONSTK(K_BACKSLASH);
	DEC_CONSTK(K_RIGHTBRACKET);
	DEC_CONSTK(K_CARET);
	DEC_CONSTK(K_UNDERSCORE);
	DEC_CONSTK(K_BACKQUOTE);
	DEC_CONSTK(K_a);
	DEC_CONSTK(K_b);
	DEC_CONSTK(K_c);
	DEC_CONSTK(K_d);
	DEC_CONSTK(K_e);
	DEC_CONSTK(K_f);
	DEC_CONSTK(K_g);
	DEC_CONSTK(K_h);
	DEC_CONSTK(K_i);
	DEC_CONSTK(K_j);
	DEC_CONSTK(K_k);
	DEC_CONSTK(K_l);
	DEC_CONSTK(K_m);
	DEC_CONSTK(K_n);
	DEC_CONSTK(K_o);
	DEC_CONSTK(K_p);
	DEC_CONSTK(K_q);
	DEC_CONSTK(K_r);
	DEC_CONSTK(K_s);
	DEC_CONSTK(K_t);
	DEC_CONSTK(K_u);
	DEC_CONSTK(K_v);
	DEC_CONSTK(K_w);
	DEC_CONSTK(K_x);
	DEC_CONSTK(K_y);
	DEC_CONSTK(K_z);
	DEC_CONSTK(K_DELETE);

	DEC_CONSTK(K_KP0);
	DEC_CONSTK(K_KP1);
	DEC_CONSTK(K_KP2);
	DEC_CONSTK(K_KP3);
	DEC_CONSTK(K_KP4);
	DEC_CONSTK(K_KP5);
	DEC_CONSTK(K_KP6);
	DEC_CONSTK(K_KP7);
	DEC_CONSTK(K_KP8);
	DEC_CONSTK(K_KP9);
	DEC_CONSTK(K_KP_PERIOD);
	DEC_CONSTK(K_KP_DIVIDE);
	DEC_CONSTK(K_KP_MULTIPLY);
	DEC_CONSTK(K_KP_MINUS);
	DEC_CONSTK(K_KP_PLUS);
	DEC_CONSTK(K_KP_ENTER);
	DEC_CONSTK(K_KP_EQUALS);
	DEC_CONSTK(K_UP);
	DEC_CONSTK(K_DOWN);
	DEC_CONSTK(K_RIGHT);
	DEC_CONSTK(K_LEFT);
	DEC_CONSTK(K_INSERT);
	DEC_CONSTK(K_HOME);
	DEC_CONSTK(K_END);
	DEC_CONSTK(K_PAGEUP);
	DEC_CONSTK(K_PAGEDOWN);
	DEC_CONSTK(K_F1);
	DEC_CONSTK(K_F2);
	DEC_CONSTK(K_F3);
	DEC_CONSTK(K_F4);
	DEC_CONSTK(K_F5);
	DEC_CONSTK(K_F6);
	DEC_CONSTK(K_F7);
	DEC_CONSTK(K_F8);
	DEC_CONSTK(K_F9);
	DEC_CONSTK(K_F10);
	DEC_CONSTK(K_F11);
	DEC_CONSTK(K_F12);
	DEC_CONSTK(K_F13);
	DEC_CONSTK(K_F14);
	DEC_CONSTK(K_F15);

	DEC_CONSTK(K_NUMLOCK);
	DEC_CONSTK(K_CAPSLOCK);
	DEC_CONSTK(K_SCROLLOCK);
	DEC_CONSTK(K_RSHIFT);
	DEC_CONSTK(K_LSHIFT);
	DEC_CONSTK(K_RCTRL);
	DEC_CONSTK(K_LCTRL);
	DEC_CONSTK(K_RALT);
	DEC_CONSTK(K_LALT);
	DEC_CONSTK(K_RMETA);
	DEC_CONSTK(K_LMETA);
	DEC_CONSTK(K_LSUPER);
	DEC_CONSTK(K_RSUPER);
	DEC_CONSTK(K_MODE);

	DEC_CONSTK(K_HELP);
	DEC_CONSTK(K_PRINT);
	DEC_CONSTK(K_SYSREQ);
	DEC_CONSTK(K_BREAK);
	DEC_CONSTK(K_MENU);
	DEC_CONSTK(K_POWER);
	DEC_CONSTK(K_EURO);
	DEC_CONSTK(K_LAST);

	DEC_CONSTN(KMOD_NONE);
	DEC_CONSTN(KMOD_LSHIFT);
	DEC_CONSTN(KMOD_RSHIFT);
	DEC_CONSTN(KMOD_LCTRL);
	DEC_CONSTN(KMOD_RCTRL);
	DEC_CONSTN(KMOD_LALT);
	DEC_CONSTN(KMOD_RALT);
	DEC_CONSTN(KMOD_LMETA);
	DEC_CONSTN(KMOD_RMETA);
	DEC_CONSTN(KMOD_NUM);
	DEC_CONSTN(KMOD_CAPS);
	DEC_CONSTN(KMOD_MODE);

	DEC_CONSTN(KMOD_CTRL);
	DEC_CONSTN(KMOD_SHIFT);
	DEC_CONSTN(KMOD_ALT);
	DEC_CONSTN(KMOD_META);
}



#if 0
/*documentation only*/

    /*DOC*/ static char doc_display[] =
    /*DOC*/    "pygame.constants.display (constants)\n"
    /*DOC*/    "The following constants are used by the display module and Surfaces\n"
    /*DOC*/    "\n"
    /*DOC*/    "HWSURFACE - surface in hardware video memory. (equal to 1)<br>\n"
    /*DOC*/    "RESIZEABLE - display window is resizeable<br>\n"
    /*DOC*/    "ASYNCBLIT - surface blits happen asynchronously (threaded)<br>\n"
    /*DOC*/    "OPENGL - display surface will be controlled by opengl<br>\n"
    /*DOC*/    "OPENGLBLIT - opengl controlled display surface will allow sdl\n"
    /*DOC*/    "blits<br>\n"
    /*DOC*/    "HWPALETTE - display surface has animatable hardware palette\n"
    /*DOC*/    "entries<br>\n"
    /*DOC*/    "DOUBLEBUF - hardware display surface is page flippable<br>\n"
    /*DOC*/    "FULLSCREEN - display surface is fullscreen (nonwindowed)<br>\n"
    /*DOC*/    "RLEACCEL - compile for quick alpha blits, only set in alpha or\n"
    /*DOC*/    "colorkey funcs<br>\n"
    /*DOC*/ ;

    /*DOC*/ static char doc_events[] =
    /*DOC*/    "pygame.constants.events (constants)\n"
    /*DOC*/    "These constants define the various event types\n"
    /*DOC*/    "\n"
    /*DOC*/    "NOEVENT - no event, represents an empty event list, equal to 0<br>\n"
    /*DOC*/    "ACTIVEEVENT - window has gain/lost mouse/keyboard/visiblity focus<br>\n"
    /*DOC*/    "KEYDOWN - keyboard button has been pressed (or down and repeating)<br>\n"
    /*DOC*/    "KEYUP - keyboard button has been released<br>\n"
    /*DOC*/    "MOUSEMOTION - mouse has moved<br>\n"
    /*DOC*/    "MOUSEBUTTONDOWN- mouse button has been pressed<br>\n"
    /*DOC*/    "MOUSEBUTTONUP - mouse button has been released<br>\n"
    /*DOC*/    "JOYAXISMOTION - an opened joystick axis has changed<br>\n"
    /*DOC*/    "JOYBALLMOTION - an opened joystick ball has moved<br>\n"
    /*DOC*/    "JOYHATMOTION - an opened joystick hat has moved<br>\n"
    /*DOC*/    "JOYBUTTONDOWN - an opened joystick button has been pressed<br>\n"
    /*DOC*/    "JOYBUTTONUP - an opened joystick button has been released<br>\n"
    /*DOC*/    "VIDEORESIZE - the display window has been resized by the user<br>\n"
    /*DOC*/    "QUIT - the user has requested the game to quit<br>\n"
    /*DOC*/    "SYSWMEVENT - currently unsupported, system dependant<br>\n"
    /*DOC*/    "USEREVENTS - all user messages are this or higher<br>\n"
    /*DOC*/    "NUMEVENTS - all user messages must be lower than this, equal to 32<br>\n"
    /*DOC*/ ;

    /*DOC*/ static char doc_keyboard[] =
    /*DOC*/    "pygame.constants.keyboard (constants)\n"
    /*DOC*/    "These constants represent the keys on the keyboard.\n"
    /*DOC*/    "\n"
    /*DOC*/    "There are many keyboard constants, they are used to represent\n"
    /*DOC*/    "keys on the keyboard. The following is a list of all keyboard\n"
    /*DOC*/    "constants\n"
    /*DOC*/    "\n"
    /*DOC*/    "<table cellpadding=0 cellspacing=0><tr><td><b>KeyASCII</b></td><td><b>ASCII&nbsp;</b></td><td><b>Common Name</b></td></tr>\n"
    /*DOC*/    "<tr><td>K_BACKSPACE</td><td>\b</td><td><i>backspace</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_TAB</td><td>\t</td><td><i>tab</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_CLEAR</td><td></td><td><i>clear</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_RETURN</td><td>\r</td><td><i>return</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_PAUSE</td><td></td><td><i>pause</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_ESCAPE</td><td>^[</td><td><i>escape</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_SPACE</td><td></td><td><i>space</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_EXCLAIM</td><td>!</td><td><i>exclaim</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_QUOTEDBL</td><td>\"</td><td><i>quotedbl</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_HASH</td><td>#</td><td><i>hash</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_DOLLAR</td><td>$</td><td><i>dollar</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_AMPERSAND</td><td>&</td><td><i>ampersand</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_QUOTE</td><td></td><td><i>quote</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_LEFTPAREN</td><td>(</td><td><i>left parenthesis</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_RIGHTPAREN</td><td>)</td><td><i>right parenthesis</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_ASTERISK</td><td>*</td><td><i>asterisk</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_PLUS</td><td>+</td><td><i>plus sign</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_COMMA</td><td>,</td><td><i>comma</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_MINUS</td><td>-</td><td><i>minus sign</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_PERIOD</td><td>.</td><td><i>period</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_SLASH</td><td>/</td><td><i>forward slash</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_0</td><td>0</td><td><i>0</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_1</td><td>1</td><td><i>1</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_2</td><td>2</td><td><i>2</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_3</td><td>3</td><td><i>3</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_4</td><td>4</td><td><i>4</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_5</td><td>5</td><td><i>5</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_6</td><td>6</td><td><i>6</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_7</td><td>7</td><td><i>7</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_8</td><td>8</td><td><i>8</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_9</td><td>9</td><td><i>9</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_COLON</td><td>:</td><td><i>colon</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_SEMICOLON</td><td>;</td><td><i>semicolon</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_LESS</td><td>&lt;</td><td><i>less-than sign</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_EQUALS</td><td>=</td><td><i>equals sign</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_GREATER</td><td>&gt;</td><td><i>greater-than sign</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_QUESTION</td><td>?</td><td><i>question mark</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_AT</td><td>@</td><td><i>at</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_LEFTBRACKET</td><td>[</td><td><i>left bracket</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_BACKSLASH</td><td>\\</td><td><i>backslash</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_RIGHTBRACKET</td><td>]</td><td><i>right bracket</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_CARET</td><td>^</td><td><i>caret</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_UNDERSCORE</td><td>_</td><td><i>underscore</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_BACKQUOTE</td><td>`</td><td><i>grave</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_a</td><td>a</td><td><i>a</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_b</td><td>b</td><td><i>b</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_c</td><td>c</td><td><i>c</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_d</td><td>d</td><td><i>d</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_e</td><td>e</td><td><i>e</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_f</td><td>f</td><td><i>f</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_g</td><td>g</td><td><i>g</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_h</td><td>h</td><td><i>h</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_i</td><td>i</td><td><i>i</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_j</td><td>j</td><td><i>j</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_k</td><td>k</td><td><i>k</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_l</td><td>l</td><td><i>l</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_m</td><td>m</td><td><i>m</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_n</td><td>n</td><td><i>n</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_o</td><td>o</td><td><i>o</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_p</td><td>p</td><td><i>p</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_q</td><td>q</td><td><i>q</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_r</td><td>r</td><td><i>r</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_s</td><td>s</td><td><i>s</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_t</td><td>t</td><td><i>t</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_u</td><td>u</td><td><i>u</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_v</td><td>v</td><td><i>v</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_w</td><td>w</td><td><i>w</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_x</td><td>x</td><td><i>x</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_y</td><td>y</td><td><i>y</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_z</td><td>z</td><td><i>z</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_DELETE</td><td></td><td><i>delete</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_KP0</td><td></td><td><i>keypad 0</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_KP1</td><td></td><td><i>keypad 1</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_KP2</td><td></td><td><i>keypad 2</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_KP3</td><td></td><td><i>keypad 3</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_KP4</td><td></td><td><i>keypad 4</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_KP5</td><td></td><td><i>keypad 5</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_KP6</td><td></td><td><i>keypad 6</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_KP7</td><td></td><td><i>keypad 7</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_KP8</td><td></td><td><i>keypad 8</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_KP9</td><td></td><td><i>keypad 9</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_KP_PERIOD</td><td>.</td><td><i>keypad period</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_KP_DIVIDE</td><td>/</td><td><i>keypad divide</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_KP_MULTIPLY</td><td>*</td><td><i>keypad multiply</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_KP_MINUS</td><td>-</td><td><i>keypad minus</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_KP_PLUS</td><td>+</td><td><i>keypad plus</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_KP_ENTER</td><td>\r</td><td><i>keypad enter</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_KP_EQUALS</td><td>=</td><td><i>keypad equals</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_UP</td><td></td><td><i>up arrow</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_DOWN</td><td></td><td><i>down arrow</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_RIGHT</td><td></td><td><i>right arrow</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_LEFT</td><td></td><td><i>left arrow</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_INSERT</td><td></td><td><i>insert</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_HOME</td><td></td><td><i>home</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_END</td><td></td><td><i>end</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_PAGEUP</td><td></td><td><i>page up</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_PAGEDOWN</td><td></td><td><i>page down</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_F1</td><td></td><td><i>F1</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_F2</td><td></td><td><i>F2</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_F3</td><td></td><td><i>F3</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_F4</td><td></td><td><i>F4</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_F5</td><td></td><td><i>F5</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_F6</td><td></td><td><i>F6</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_F7</td><td></td><td><i>F7</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_F8</td><td></td><td><i>F8</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_F9</td><td></td><td><i>F9</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_F10</td><td></td><td><i>F10</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_F11</td><td></td><td><i>F11</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_F12</td><td></td><td><i>F12</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_F13</td><td></td><td><i>F13</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_F14</td><td></td><td><i>F14</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_F15</td><td></td><td><i>F15</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_NUMLOCK</td><td></td><td><i>numlock</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_CAPSLOCK</td><td></td><td><i>capslock</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_SCROLLOCK</td><td></td><td><i>scrollock</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_RSHIFT</td><td></td><td><i>right shift</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_LSHIFT</td><td></td><td><i>left shift</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_RCTRL</td><td></td><td><i>right ctrl</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_LCTRL</td><td></td><td><i>left ctrl</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_RALT</td><td></td><td><i>right alt</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_LALT</td><td></td><td><i>left alt</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_RMETA</td><td></td><td><i>right meta</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_LMETA</td><td></td><td><i>left meta</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_LSUPER</td><td></td><td><i>left windows key</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_RSUPER</td><td></td><td><i>right windows key</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_MODE</td><td></td><td><i>mode shift</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_HELP</td><td></td><td><i>help</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_PRINT</td><td></td><td><i>print-screen</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_SYSREQ</td><td></td><td><i>SysRq</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_BREAK</td><td></td><td><i>break</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_MENU</td><td></td><td><i>menu</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_POWER</td><td></td><td><i>power</i></td></tr>\n"
    /*DOC*/    "<tr><td>K_EURO</td><td></td><td><i>euro</i></td></tr></table>\n"
    /*DOC*/ ;

    /*DOC*/ static char doc_modifiers[] =
    /*DOC*/    "pygame.constants.modifiers (constants)\n"
    /*DOC*/    "These constants represent the modifier keys on the keyboard.\n"
    /*DOC*/    "\n"
    /*DOC*/    "Their states are treated slightly differently than normal\n"
    /*DOC*/    "keyboard button states, and you can temporarily set their states.\n"
    /*DOC*/    "\n"
    /*DOC*/    "KMOD_NONE, KMOD_LSHIFT, KMOD_RSHIFT, KMOD_SHIFT, KMOD_CAPS,<br>\n"
    /*DOC*/    "KMOD_LCTRL, KMOD_RCTRL, KMOD_CTRL, KMOD_LALT, KMOD_RALT,<br>\n"
    /*DOC*/    "KMOD_ALT, KMOD_LMETA, KMOD_RMETA, KMOD_META, KMOD_NUM, KMOD_MODE<br>\n"
    /*DOC*/ ;

    /*DOC*/ static char doc_zdeprecated[] =
    /*DOC*/    "pygame.constants.zdepracated (constants)\n"
    /*DOC*/    "The following constants are made available, but generally not needed\n"
    /*DOC*/    "\n"
    /*DOC*/    "The flags labeled as readonly should never be used,\n"
    /*DOC*/    "except when comparing checking flags against Surface.get_flags().\n"
    /*DOC*/    "\n"
    /*DOC*/    "SWSURFACE - not really usable as a surface flag, equates to 0 and\n"
    /*DOC*/    "is always default<br>\n"
    /*DOC*/    "ANYFORMAT - creates a display with in best possible bit depth<br>\n"
    /*DOC*/    "HWACCEL - surface is hardware accelerated, readonly<br>\n"
    /*DOC*/    "SRCCOLORKEY- surface has a colorkey for blits, readonly<br>\n"
    /*DOC*/    "SRCALPHA - surface has alpha enabled, readonly<br>\n"
    /*DOC*/    "RLEACCELOK - surface is rle accelrated but uncompiled, readonly\n"
    /*DOC*/ ;


#endif
