/*
 * This file should not exist. If you exist in the future and are wondering if
 * this file is cool, it is not.
 */

#include <SDL.h>

/*
 * Compat thing for controller (see https://github.com/pygame/pygame/pull/3270)
 */
int
PG_GameControllerRumble(SDL_GameController *gamecontroller,
                        Uint16 low_frequency_rumble,
                        Uint16 high_frequency_rumble, Uint32 duration_ms)
{
#if SDL_VERSION_ATLEAST(2, 0, 9)
    return SDL_GameControllerRumble(gamecontroller, low_frequency_rumble,
                                    high_frequency_rumble, duration_ms);
#else
    return -1;
#endif
}
