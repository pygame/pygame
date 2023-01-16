#include "_surface.h"

#include "draw_shared.h"

int
set_at(SDL_Surface *surf, int x, int y, Uint32 color)
{
    SDL_PixelFormat *format = surf->format;
    Uint8 *pixels = (Uint8 *)surf->pixels;
    Uint8 *byte_buf, rgb[4];

    if (x < surf->clip_rect.x || x >= surf->clip_rect.x + surf->clip_rect.w ||
        y < surf->clip_rect.y || y >= surf->clip_rect.y + surf->clip_rect.h)
        return 0;

    switch (format->BytesPerPixel) {
        case 1:
            *((Uint8 *)pixels + y * surf->pitch + x) = (Uint8)color;
            break;
        case 2:
            *((Uint16 *)(pixels + y * surf->pitch) + x) = (Uint16)color;
            break;
        case 4:
            *((Uint32 *)(pixels + y * surf->pitch) + x) = color;
            break;
        default: /*case 3:*/
            SDL_GetRGB(color, format, rgb, rgb + 1, rgb + 2);
            byte_buf = (Uint8 *)(pixels + y * surf->pitch) + x * 3;
#if (SDL_BYTEORDER == SDL_LIL_ENDIAN)
            *(byte_buf + (format->Rshift >> 3)) = rgb[0];
            *(byte_buf + (format->Gshift >> 3)) = rgb[1];
            *(byte_buf + (format->Bshift >> 3)) = rgb[2];
#else
            *(byte_buf + 2 - (format->Rshift >> 3)) = rgb[0];
            *(byte_buf + 2 - (format->Gshift >> 3)) = rgb[1];
            *(byte_buf + 2 - (format->Bshift >> 3)) = rgb[2];
#endif
            break;
    }
    return 1;
}

void
drawhorzline(SDL_Surface *surf, Uint32 color, int x1, int y1, int x2)
{
    Uint8 *pixel, *end;

    pixel = ((Uint8 *)surf->pixels) + surf->pitch * y1;
    end = pixel + x2 * surf->format->BytesPerPixel;
    pixel += x1 * surf->format->BytesPerPixel;
    switch (surf->format->BytesPerPixel) {
        case 1:
            for (; pixel <= end; ++pixel) {
                *pixel = (Uint8)color;
            }
            break;
        case 2:
            for (; pixel <= end; pixel += 2) {
                *(Uint16 *)pixel = (Uint16)color;
            }
            break;
        case 3:
#if SDL_BYTEORDER == SDL_BIG_ENDIAN
            color <<= 8;
#endif
            for (; pixel <= end; pixel += 3) {
                memcpy(pixel, &color, 3 * sizeof(Uint8));
            }
            break;
        default: /*case 4*/
            for (; pixel <= end; pixel += 4) {
                *(Uint32 *)pixel = color;
            }
            break;
    }
}

static void
drawhorzlineclip(SDL_Surface *surf, Uint32 color, int x1, int y1, int x2)
{
    if (y1 < surf->clip_rect.y || y1 >= surf->clip_rect.y + surf->clip_rect.h)
        return;

    if (x2 < x1) {
        int temp = x1;
        x1 = x2;
        x2 = temp;
    }

    x1 = MAX(x1, surf->clip_rect.x);
    x2 = MIN(x2, surf->clip_rect.x + surf->clip_rect.w - 1);

    if (x2 < surf->clip_rect.x || x1 >= surf->clip_rect.x + surf->clip_rect.w)
        return;

    if (x1 == x2) {
        set_at(surf, x1, y1, color);
        return;
    }
    drawhorzline(surf, color, x1, y1, x2);
}

void
draw_rect(SDL_Surface *surf, int x1, int y1, int x2, int y2, int width,
          Uint32 color)
{
    int i;
    for (i = 0; i < width; i++) {
        drawhorzlineclip(surf, color, x1, y1 + i, x2);
        drawhorzlineclip(surf, color, x1, y2 - i, x2);
    }
    for (i = 0; i < (y2 - y1) - 2 * width + 1; i++) {
        drawhorzlineclip(surf, color, x1, y1 + width + i, x1 + width - 1);
        drawhorzlineclip(surf, color, x2 - width + 1, y1 + width + i, x2);
    }
}
