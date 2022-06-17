#include <SDL.h>

#if !(SDL_VERSION_ATLEAST(2, 0, 5))
/* These functions require SDL 2.0.5 or greater.

  https://wiki.libsdl.org/SDL_SetWindowResizable
*/
void
SDL_SetWindowResizable(SDL_Window *window, SDL_bool resizable)
{
}
int
SDL_GetWindowOpacity(SDL_Window *window, float *opacity)
{
    SDL_SetError("SDL 2.0.5 or greater required");
    return -1;
}
int
SDL_SetWindowOpacity(SDL_Window *window, float opacity)
{
    SDL_SetError("SDL 2.0.5 or greater required");
    return -1;
}

int
SDL_SetWindowModalFor(SDL_Window *modal_window, SDL_Window *parent_window)
{
    SDL_SetError("SDL 2.0.5 or greater required");
    return -1;
}
int
SDL_SetWindowInputFocus(SDL_Window *window)
{
    SDL_SetError("SDL 2.0.5 or greater required");
    return -1;
}
SDL_Surface *
SDL_CreateRGBSurfaceWithFormat(Uint32 flags, int width, int height, int depth,
                               Uint32 format)
{
    SDL_SetError("SDL 2.0.5 or greater required");
    return NULL;
}

#endif /*if !(SDL_VERSION_ATLEAST(2, 0, 5))*/

#if !(SDL_VERSION_ATLEAST(2, 0, 9))
int
SDL_GameControllerRumble(void *gamecontroller, Uint16 low_frequency_rumble,
                         Uint16 high_frequency_rumble, Uint32 duration_ms)
{
    SDL_SetError("SDL 2.0.9 or greater required");
    return -1;
}

#endif /*#if !(SDL_VERSION_ATLEAST(2, 0, 9))*/
