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
PyObject* _movie_easy_seek    (PyMovie *movie, PyObject* args, PyObject *kwds);
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
