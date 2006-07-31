/*
    pygame - Python Game Library
    Copyright (C) 2006 Rene Dudfield, Marcus von Appen

    Originally put in the public domain by Sam Lantinga.

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Library General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Library General Public License for more details.

    You should have received a copy of the GNU Library General Public
    License along with this library; if not, write to the Free
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

#include <Python.h>

/* Handle clipboard text and data in arbitrary formats */

/**
 * Type conversion macro for the scrap formats.
 */
#define PYGAME_SCRAP_TYPE(A, B, C, D) \
    ((int) ((A<<24) | (B<<16) | (C<<8) | (D<<0)))

/**
 * Currently supported pygame scrap types.
 * 
 * PYGAME_SCRAP_TEXT should be used for text.
 * PYGAME_SCRAP_BMP should be used for arbitrary data in a string format.
 * 
 * PYGAME_SCRAP_BMP is only supported by pygame window instances. For
 * interchangeable data that should be used by other applications, use
 * PYGAME_SCRAP_TEXT.
 */
#define PYGAME_SCRAP_TEXT PYGAME_SCRAP_TYPE('T', 'E', 'X', 'T')
#define PYGAME_SCRAP_BMP PYGAME_SCRAP_TYPE('B', 'M', 'P', ' ')

/**
 * Macro for initialization checks.
 */
#define PYGAME_SCRAP_INIT_CHECK() \
    if(!pygame_scrap_initialized()) \
        return (PyErr_SetString (PyExc_SDLError, \
                                 "scrap system not initialized."), NULL)

/**
 * \brief Checks, whether the pygame scrap module was initialized.
 *
 * \return 1 if the modules was initialized, 0 otherwise.
 */
extern int
pygame_scrap_initialized (void);

/**
 * \brief Initializes the pygame scrap module internals. Call this before any
 *        other method.
 *
 * \return 1 on successful initialization, 0 otherwise.
 */
extern int
pygame_init_scrap (void);

/**
 * \brief Checks, whether the pygame window lost the clipboard focus or not.
 *
 * \return 1 if the window lost the focus, 0 otherwise.
 */
extern int
pygame_lost_scrap (void);

/**
 * \brief Places content of a specific type into the clipboard.
 *
 * \note The clipboard implementations are usually global for all
 *       applications or a specific content type (QNX). Thus,
 *       any previous content, be it from another window or of a different
 *       format will be replaced by the passed content. This also applies
 *       to other windows, that access the clipboard in any way, so that
 *       it is not guaranteed that the placed content will last until
 *       termination of the pygame application.
 *
 * \param type The type of the content. Should be a valid value of
 *             PYGAME_SCRAP_TEXT, PYGAME_SCRAP_BMP.
 * \param srclen The length of the content.
 * \param src The NULL terminated content.
 * \return 1, if the content could be successfully pasted into the clipboard,
 *         0 otherwise.
 */
extern int
pygame_put_scrap (int type, int srclen, char *src);

/**
 * \brief Gets the current content from the clipboard.
 *
 * \note The received content does not need to be the content previously
 *       placed in the clipboard using pygame_put_scrap(). See the 
 *       pygame_put_scrap() notes for more details.
 *
 * \param The type of the content to receive.
 * \return The content or NULL in case of an error or if no content of the
 *         specified type was available.
 */
extern char*
pygame_get_scrap (int type);
