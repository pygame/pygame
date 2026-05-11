#include "SDL.h"
#include "SDL_clipboard.h"

#define PYGAME_SCRAP_FREE_STRING 1

char *pygame_scrap_plaintext_type = "text/plain;charset=utf-8";
char **pygame_scrap_types;

int
pygame_scrap_contains(char *type)
{
    return (strcmp(type, pygame_scrap_plaintext_type) == 0) &&
           SDL_HasClipboardText();
}

char *
pygame_scrap_get(char *type, size_t *count)
{
    char *retval = NULL;
    char *clipboard = NULL;

    if (!pygame_scrap_initialized()) {
        PyErr_SetString(pgExc_SDLError, "scrap system not initialized.");
        return NULL;
    }

    if (strcmp(type, pygame_scrap_plaintext_type) == 0) {
        clipboard = SDL_GetClipboardText();
        if (clipboard != NULL) {
            *count = strlen(clipboard);
            retval = strdup(clipboard);
            SDL_free(clipboard);
            return retval;
        }
    }
    return NULL;
}

char **
pygame_scrap_get_types(void)
{
    if (!pygame_scrap_initialized()) {
        PyErr_SetString(pgExc_SDLError, "scrap system not initialized.");
        return NULL;
    }

    return pygame_scrap_types;
}

int
pygame_scrap_init(void)
{
    /* Phase 1F (Self-Healing 2026-05-05) — previously the
     * SDL_Init return value was discarded; an SDL_Init failure
     * here meant scrap silently proceeded to malloc + flag
     * itself initialised even though the underlying video
     * subsystem wasn't actually up. The caller in scrap.c then
     * had no way to know clipboard ops would fail with cryptic
     * SDL errors. Now we propagate the SDL_Init failure through
     * the function's existing 0-return contract; the caller
     * already does `RAISE(pgExc_SDLError, SDL_GetError())` on
     * the 0 return path, so operators get the real error
     * string instead of silent corruption.
     *
     * Note: this code path is rarely hit in practice because
     * the calling convention requires VIDEO_INIT_CHECK() before
     * pygame_scrap_init, which means SDL_INIT_VIDEO is already
     * up and SDL_Init becomes a no-op. The fix is defensive
     * against the case where the SDL state is somehow lost
     * between VIDEO_INIT_CHECK and this call. */
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        return 0;
    }

    pygame_scrap_types = malloc(sizeof(char *) * 2);
    if (!pygame_scrap_types) {
        return 0;
    }

    pygame_scrap_types[0] = pygame_scrap_plaintext_type;
    pygame_scrap_types[1] = NULL;

    _scrapinitialized = 1;
    return _scrapinitialized;
}

int
pygame_scrap_lost(void)
{
    return 1;
}

int
pygame_scrap_put(char *type, Py_ssize_t srclen, char *src)
{
    if (!pygame_scrap_initialized()) {
        PyErr_SetString(pgExc_SDLError, "scrap system not initialized.");
        return 0;
    }

    if (strcmp(type, pygame_scrap_plaintext_type) == 0) {
        if (SDL_SetClipboardText(src) == 0) {
            return 1;
        }
    }
    return 0;
}
