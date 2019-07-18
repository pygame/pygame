#include "SDL.h"
#include "SDL_clipboard.h"

char* plaintext_type= "text/plain;charset=utf-8";

int
pygame_scrap_contains(char *type){
    return (strcmp(type, plaintext_type) == 0)
        && SDL_HasClipboardText();
}

char*
pygame_scrap_get(char *type, unsigned long *count){
    char *retval = NULL;
    char *clipboard = NULL;


    if (!pygame_scrap_initialized()) {
        PyErr_SetString(pgExc_SDLError, "scrap system not initialized.");
        return NULL;
    }

    if  (strcmp(type, plaintext_type) == 0){
        printf("type OK\n");
        clipboard = SDL_GetClipboardText();
        if (clipboard != NULL){
            *count=strlen(clipboard);
            retval = strdup(clipboard);
            SDL_free(clipboard);
            return retval;
        }
    }
    return NULL;
}

char**
pygame_scrap_get_types(void){
    char** types;
    types = malloc(sizeof(char *) * 2);
    if (!types)
        return NULL;

    types[0] = strdup(plaintext_type);
    types[1] = NULL;

    return types;
}

int
pygame_scrap_init(void){
    SDL_Init(SDL_INIT_VIDEO);
    _scrapinitialized = 1;
    return _scrapinitialized;
}

int
pygame_scrap_lost(void){
    return 1;
}

int
pygame_scrap_put(char *type, int srclen, char *src){
    if (!pygame_scrap_initialized()) {
        PyErr_SetString(pgExc_SDLError, "scrap system not initialized.");
        return 0;
    }
    if (strcmp(type, plaintext_type) == 0){
        if (SDL_SetClipboardText(src) == 0) {
            return 1;
        }
    }
    return 0;
}
