/*
  pygame - Python Game Library
  Copyright (C) 2006, 2007 Rene Dudfield, Marcus von Appen

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

#ifndef _SCRAP_H_
#define _SCRAP_H_

#include <SDL.h>

/**
 * The supported scrap clipboard types.
 * 
 * This is only relevant in a X11 environment, which supports mouse
 * selections as well. For Win32 and MacOS environments the default
 * clipboard is used, no matter what value is passed.
 */
typedef enum
{
    SCRAP_SELECTION,
    SCRAP_CLIPBOARD
} ScrapType;

/**
 * Predefined supported scrap types.
 */
#define SCRAP_FORMAT_TEXT "text/plain"
#define SCRAP_FORMAT_BMP "image/bmp"
#define SCRAP_FORMAT_PPM "image/ppm"
#define SCRAP_FORMAT_PBM "image/pbm"

/**
 * \brief Initializes the scrap module internals. Call this before any
 * other method.
 *
 * \return 1 on successful initialization, 0 otherwise.
 */
int
pyg_scrap_init (void);

/**
 * \brief Checks, whether the scrap module was initialized.
 *
 * \return 1 if the module was initialized, 0 otherwise.
 */
int
pyg_scrap_was_init (void);

/**
 * \brief Releases the internals of the scrap module.
 */
void
pyg_scrap_quit (void);

/**
 * \brief Checks whether content for the specified scrap type is currently
 * available in the clipboard.
 *
 * \param type The type to check for.
 * \return 1, if there is content, 0 otherwise. If an error occured, -1 is
 * returned.
 */
int
pyg_scrap_contains (char *type);

/**
 * \brief Checks, whether the window lost the clipboard focus or not.
 *
 * \return 1 if the window lost the focus, 0 otherwise.
 */
 int
pyg_scrap_lost (void);

ScrapType
pyg_scrap_get_mode (void);

ScrapType
pyg_scrap_set_mode (ScrapType mode);

/**
 * \brief Gets the current content from the clipboard.
 *
 * \note The received content does not need to be the content previously
 *       placed in the clipboard using pyg_scrap_put(). See the 
 *       pyg_scrap_put() notes for more details.
 *
 * \param type The type of the content to receive.
 * \param data Pointer to the storage location for the content.
 * \param size The size of the returned content.
 * \return 1, if the content could be retrieved, 0 if there was no content.
 * If an error occured, -1 is returned.
 */
int
pyg_scrap_get (char *type, char **data, unsigned int *size);

/**
 * \brief Places content of a specific type into the clipboard.
 *
 * \note For X11 the following notes are important: The following types
 *       are reserved for internal usage and thus will throw an error on
 *       setting them: "TIMESTAMP", "TARGETS", "SDL_SELECTION".
 *       Setting SCRAP_FORMAT_TEXT ("text/plain") will also automatically
 *       set the X11 types "STRING" (XA_STRING), "TEXT" and "UTF8_STRING".
 *
 *       For Win32 the following notes are important: Setting
 *       SCRAP_FORMAT_TEXT ("text/plain") will also automatically set
 *       the Win32 type "TEXT" (CF_TEXT).
 *
 * \param type The type of the content.
 * \param data The NULL terminated content.
 * \param size The length of the content.
 * \return 1, if the content could be successfully pasted into the clipboard,
 *         -1 if an error occured.
 */
int
pyg_scrap_put (char *type, char *data, unsigned int size);

/**
 * \brief Gets the currently available content types from the clipboard.
 *
 * \param types Pointer to the storage location for the types.
 * \return 1, if types could be retrieved, 0 if the clipboard is empty.
 * If an error occured, -1 is returned.
 */
int
pyg_scrap_get_types (char** types);

#endif /* _SCRAP_H_ */
