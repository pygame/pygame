/*
 * Compat file specifically for controller (see
 * https://github.com/pygame/pygame/pull/3272) we had trouble using pgcompat.c,
 * because it would be included twice, or it would lead to flakiness on
 * compilers, because of threads...
 */

#include <SDL.h>

int
PG_GameControllerRumble(SDL_GameController *gamecontroller,
                        Uint16 low_frequency_rumble,
                        Uint16 high_frequency_rumble, Uint32 duration_ms)
{
#if SDL_VERSION_ATLEAST(2, 0, 9)
    return SDL_GameControllerRumble(gamecontroller, low_frequency_rumble,
                                    high_frequency_rumble, duration_ms);
#else
    SDL_SetError("pygame built without controller rumble support");
    return -1;
#endif
}
