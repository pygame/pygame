/* Temporary compatibilty definitions while the Pygame C API has the "pg" prefix added.
 * This is included at the top of _pygame.h; remove when done.
 */
#ifndef API_TRANSITION_H
#define API_TRANSITION_H
#ifdef __GNUC__
#warning Transitional include until Pygame C API is fully updated
#endif

#include "base_api_transition.h"
#include "rect_api_transition.h"
#include "rwobject_api_transition.h"
#include "color_api_transition.h"
#include "surflock_api_transition.h"
#include "bufferproxy_api_transition.h"
#include "surface_api_transition.h"
#include "event_api_transition.h"
#include "display_api_transition.h"
#include "mixer_api_transition.h"

#endif
