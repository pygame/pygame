#ifndef GMOVIE_H_
#define GMOVIE_H_
#include "_gmovie.h"

/*class methods and internals */
PyMovie*  _movie_init_internal(PyMovie *self, const char *filename, SDL_Surface *surf);
int       _movie_init         (PyObject *self, PyObject *args, PyObject *kwds);
void      _movie_dealloc      (PyMovie *movie);
PyObject* _movie_repr         (PyMovie *movie);
PyObject* _movie_play         (PyMovie *movie, PyObject* args);
PyObject* _movie_stop         (PyMovie *movie);
PyObject* _movie_pause        (PyMovie *movie);
PyObject* _movie_rewind       (PyMovie *movie, PyObject* args);

/* Getters/setters */
PyObject* _movie_get_paused  (PyMovie *movie, void *closure);
PyObject* _movie_get_playing (PyMovie *movie, void *closure);
PyObject* _movie_get_width   (PyMovie *movie, void *closure);
PyObject* _movie_get_height  (PyMovie *movie, void *closure);

#endif /*GMOVIE_H_*/
