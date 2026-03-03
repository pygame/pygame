#ifdef PG_SDL3
#include <SDL3/SDL.h>
#include <SDL3/SDL_clipboard.h>
#else
#include <SDL.h>
#include <SDL_clipboard.h>
#endif

#define PYGAME_SCRAP_FREE_STRING 1

char *pygame_scrap_plaintext_type = "text/plain";
char *pygame_scrap_utf8text_type = "text/plain;charset=utf-8";
char **pygame_scrap_types;

int
pygame_scrap_contains(char *type)
{
    return (strcmp(type, pygame_scrap_plaintext_type) == 0 ||
            strcmp(type, pygame_scrap_utf8text_type) == 0) &&
           SDL_HasClipboardText();
}

/* Returns 1 if str is valid ascii, 0 otherwise */
static int
_pg_validate_ascii_str(const char *str)
{
    while (*str) {
        if ((unsigned char)*str > 127) {
            return 0;
        }
        str++;
    }
    return 1;
}

char *
pygame_scrap_get(char *type, size_t *count)
{
    char *retval = NULL;
    char *clipboard = NULL;

    if (count) {
        *count = 0;
    }

    if (!pygame_scrap_initialized()) {
        return RAISE(pgExc_SDLError, "scrap system not initialized.");
    }

    int is_utf8 = strcmp(type, pygame_scrap_utf8text_type) == 0;
    if (is_utf8 || strcmp(type, pygame_scrap_plaintext_type) == 0) {
        clipboard = SDL_GetClipboardText();
        if (clipboard != NULL) {
            if (is_utf8 || _pg_validate_ascii_str(clipboard)) {
                if (count) {
                    *count = strlen(clipboard);
                }
                retval = strdup(clipboard);
            }
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
    SDL_Init(SDL_INIT_VIDEO);

    pygame_scrap_types = malloc(sizeof(char *) * 3);
    if (!pygame_scrap_types) {
        return 0;
    }

    pygame_scrap_types[0] = pygame_scrap_plaintext_type;
    pygame_scrap_types[1] = pygame_scrap_utf8text_type;
    pygame_scrap_types[2] = NULL;

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

    if (strcmp(type, pygame_scrap_plaintext_type) == 0 ||
        strcmp(type, pygame_scrap_utf8text_type) == 0) {
#if SDL_VERSION_ATLEAST(3, 0, 0)
        if (SDL_SetClipboardText(src)) {
#else
        if (SDL_SetClipboardText(src) == 0) {
#endif
            return 1;
        }
    }
    return 0;
}
