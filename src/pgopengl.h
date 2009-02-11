#ifndef _PYGAME_OPENGL_H_
#define _PYGAME_OPENGL_H_

/*
 * This header includes definitions of Opengl functions as pointer types
 * for use with the SDL function SDL_GL_GetProcAddress.
 */

#if defined(_WIN32)
#define GL_APIENTRY __stdcall
#else
#define GL_APIENTRY
#endif

typedef void (GL_APIENTRY *GL_glReadPixels_Func)(int, int, int, int,
    unsigned int, unsigned int, void*);

#endif /* _PYGAME_OPENGL_H_ */
