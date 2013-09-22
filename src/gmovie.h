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

/*
 * _movie - movie support for pygame with ffmpeg
 * Author: Tyler Laing
 *
 * This module allows for the loading of, playing, pausing, stopping, and so on
 *  of a video file. Any format supported by ffmpeg is supported by this
 *  video player. Any bugs, please email trinioler@gmail.com :)
 */


#ifndef GMOVIE_H_
#define GMOVIE_H_
#include "_gmovie.h"

/*class methods and internals */
void      _movie_init_internal(PyMovie *self, const char *filename, SDL_Surface *surf);
int       _movie_init         (PyObject *self, PyObject *args, PyObject *kwds);
void      _movie_dealloc      (PyMovie *movie);
PyObject* _movie_repr         (PyMovie *movie);
PyObject* _movie_play         (PyMovie *movie, PyObject* args);
PyObject* _movie_stop         (PyMovie *movie);
PyObject* _movie_pause        (PyMovie *movie);
PyObject* _movie_rewind       (PyMovie *movie, PyObject* args);
PyObject* _movie_resize       (PyMovie *movie, PyObject* args);
PyObject* _movie_seek         (PyMovie *movie, PyObject* args);
PyObject* _movie_easy_seek    (PyMovie *movie, PyObject* args);
PyObject *_movie_shift(PyMovie *movie, PyObject*args);

/* Getters/setters */
PyObject* _movie_get_paused  (PyMovie *movie, void *closure);
PyObject* _movie_get_playing (PyMovie *movie, void *closure);
PyObject* _movie_get_finished(PyMovie *movie,  void *closure);
PyObject* _movie_get_width   (PyMovie *movie, void *closure);
int       _movie_set_width   (PyMovie *movie, PyObject *width, void *closure);
PyObject* _movie_get_height  (PyMovie *movie, void *closure);
int       _movie_set_height  (PyMovie *movie, PyObject *height, void *closure);
PyObject* _movie_get_surface (PyMovie *movie, void *closure);
int       _movie_set_surface (PyObject *movie, PyObject *surface, void *closure);

#endif /*GMOVIE_H_*/
