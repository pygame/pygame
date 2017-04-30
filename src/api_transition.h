/* Temporary compatibilty definitions while the Pygame C API has the "pg" prefix added.
 * This is included at the top of _pygame.h; remove when done.
 */
#ifndef API_TRANSITION_H
#define API_TRANSITION_H
#warning Transitional include until Pygame C API is fully updated

#ifdef __GNUC__
#include "base_api_transition.h"
#endif

#endif
