/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2022 Sam Lantinga <slouken@libsdl.org>

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/

/* Touch gestures were removed from SDL3, so this is the SDL2 implementation
 * copied in here, and tweaked a little. */

#ifndef INCL_SDL_GESTURE_H
#define INCL_SDL_GESTURE_H

#if !defined(SDL_MAJOR_VERSION)
#error Please include SDL.h before including this header.
#elif SDL_MAJOR_VERSION < 2
#error This header requires SDL2 or later.
#elif SDL_MAJOR_VERSION == 2
/* building against SDL2? Just use the built-in SDL2 implementation. */
#define Gesture_Init() (0)
#define Gesture_Quit()
#define Gesture_ID SDL_GestureID
#define Gesture_LoadDollarTemplates SDL_LoadDollarTemplates
#define Gesture_RecordGesture SDL_RecordGesture
#define Gesture_SaveAllDollarTemplates SDL_SaveAllDollarTemplates
#define Gesture_SaveDollarTemplate SDL_SaveDollarTemplate
#define GESTURE_DOLLARGESTURE SDL_DOLLARGESTURE
#define GESTURE_DOLLARRECORD SDL_DOLLARRECORD
#define GESTURE_MULTIGESTURE SDL_MULTIGESTURE
#define Gesture_MultiGestureEvent SDL_MultiGestureEvent
#define Gesture_DollarGestureEvent SDL_DollarGestureEvent
#else

/* Set up for C function definitions, even when using C++ */
#ifdef __cplusplus
extern "C" {
#endif

typedef Sint64 Gesture_ID;

/* events... */

/* generally you shouldn't hardcode event type numbers--and doubly so in
   the reserved range!--but these match SDL2 and SDL3 promises to preserve
   these values to help sdl2-compat. */
#define GESTURE_DOLLARGESTURE 0x800
#define GESTURE_DOLLARRECORD 0x801
#define GESTURE_MULTIGESTURE 0x802

typedef struct Gesture_MultiGestureEvent {
    Uint32 type;
    Uint32 reserved;
    Uint64 timestamp;
    SDL_TouchID touchID;
    float dTheta;
    float dDist;
    float x;
    float y;
    Uint16 numFingers;
    Uint16 padding;
} Gesture_MultiGestureEvent;

typedef struct Gesture_DollarGestureEvent {
    Uint32 type;
    Uint32 reserved;
    Uint64 timestamp;
    SDL_TouchID touchID;
    Gesture_ID gestureId;
    Uint32 numFingers;
    float error;
    float x;
    float y;
} Gesture_DollarGestureEvent;

/* Function prototypes */

/**
 * Call this once, AFTER SDL_Init, to set up the Gesture API.
 *
 * \returns 0 on success, -1 on error. Call SDL_GetError() for specifics.
 */
extern int SDLCALL
Gesture_Init(void);

/**
 * Call this once, BEFORE SDL_Quit, to clean up the Gesture API.
 */
extern void SDLCALL
Gesture_Quit(void);

/**
 * Begin recording a gesture on a specified touch device or all touch devices.
 *
 * If the parameter `touchID` is -1 (i.e., all devices), this function will
 * always return 1, regardless of whether there actually are any devices.
 *
 * \param touchID the touch device id, or -1 for all touch devices
 * \returns 1 on success or 0 if the specified device could not be found.
 */
extern int SDLCALL
Gesture_RecordGesture(SDL_TouchID touchID);

/**
 * Save all currently loaded Dollar Gesture templates.
 *
 * \param dst a SDL_IOStream to save to
 * \returns the number of saved templates on success or 0 on failure; call
 *          SDL_GetError() for more information.
 *
 * \since This function is available since SDL 2.0.0.
 *
 * \sa Gesture_LoadDollarTemplates
 * \sa Gesture_SaveDollarTemplate
 */
extern int SDLCALL
Gesture_SaveAllDollarTemplates(SDL_IOStream *dst);

/**
 * Save a currently loaded Dollar Gesture template.
 *
 * \param gestureId a gesture id
 * \param dst a SDL_IOStream to save to
 * \returns 1 on success or 0 on failure; call SDL_GetError() for more
 *          information.
 *
 * \since This function is available since SDL 2.0.0.
 *
 * \sa SDL_LoadDollarTemplates
 * \sa SDL_SaveAllDollarTemplates
 */
extern int SDLCALL
Gesture_SaveDollarTemplate(Gesture_ID gestureId, SDL_IOStream *dst);

/**
 * Load Dollar Gesture templates from a file.
 *
 * \param touchID a touch id
 * \param src a SDL_IOStream to load from
 * \returns the number of loaded templates on success or a negative error code
 *          (or 0) on failure; call SDL_GetError() for more information.
 *
 * \since This function is available since SDL 2.0.0.
 *
 * \sa SDL_SaveAllDollarTemplates
 * \sa SDL_SaveDollarTemplate
 */
extern int SDLCALL
Gesture_LoadDollarTemplates(SDL_TouchID touchID, SDL_IOStream *src);

/* Ends C function definitions when using C++ */
#ifdef __cplusplus
}
#endif

#if defined(SDL_GESTURE_IMPLEMENTATION)

#define GESTURE_MAX_DOLLAR_PATH_SIZE 1024
#define GESTURE_DOLLARNPOINTS 64
#define GESTURE_DOLLARSIZE 256
#define GESTURE_PHI 0.618033989

typedef struct {
    float length;
    int numPoints;
    SDL_FPoint p[GESTURE_MAX_DOLLAR_PATH_SIZE];
} GestureDollarPath;

typedef struct {
    SDL_FPoint path[GESTURE_DOLLARNPOINTS];
    Uint64 hash;
} GestureDollarTemplate;

typedef struct {
    SDL_TouchID touchID;
    SDL_FPoint centroid;
    GestureDollarPath dollarPath;
    Uint16 numDownFingers;
    int numDollarTemplates;
    GestureDollarTemplate *dollarTemplate;
    bool recording;
} GestureTouch;

static GestureTouch *GestureTouches = NULL;
static int GestureNumTouches = 0;
static bool GestureRecordAll = false;

static void
GestureProcessEvent(const SDL_Event *event);

static bool SDLCALL
GestureEventWatch(void *userdata, SDL_Event *event)
{
    GestureProcessEvent(event);
    return true;
}

int
Gesture_Init(void)
{
    Gesture_Quit();
    SDL_AddEventWatch(GestureEventWatch, NULL);
    return 0;
}

static GestureTouch *
GestureAddTouch(const SDL_TouchID touchID)
{
    GestureTouch *gestureTouch = (GestureTouch *)SDL_realloc(
        GestureTouches, (GestureNumTouches + 1) * sizeof(GestureTouch));
    if (gestureTouch == NULL) {
        SDL_OutOfMemory();
        return NULL;
    }

    GestureTouches = gestureTouch;
    SDL_zero(GestureTouches[GestureNumTouches]);
    GestureTouches[GestureNumTouches].touchID = touchID;
    return &GestureTouches[GestureNumTouches++];
}

/* pygame specific patch:
 * This has been commented out to silence compiler warnings of this function
 * being unused. */
#if 0
static int
GestureDelTouch(const SDL_TouchID touchID)
{
    int i;
    for (i = 0; i < GestureNumTouches; i++) {
        if (GestureTouches[i].touchID == touchID) {
            break;
        }
    }

    if (i == GestureNumTouches) {
        /* not found */
        return -1;
    }

    SDL_free(GestureTouches[i].dollarTemplate);
    SDL_zero(GestureTouches[i]);

    GestureNumTouches--;
    if (i != GestureNumTouches) {
        SDL_copyp(&GestureTouches[i], &GestureTouches[GestureNumTouches]);
    }
    return 0;
}
#endif

static GestureTouch *
GestureGetTouch(const SDL_TouchID touchID)
{
    int i;
    for (i = 0; i < GestureNumTouches; i++) {
        /* printf("%i ?= %i\n",GestureTouches[i].touchID,touchID); */
        if (GestureTouches[i].touchID == touchID) {
            return &GestureTouches[i];
        }
    }
    return NULL;
}

int
Gesture_RecordGesture(SDL_TouchID touchID)
{
    SDL_TouchID *devices;
    int i;

    devices = SDL_GetTouchDevices(NULL);
    if (devices) {
        /* make sure we know about all the devices SDL3 knows about, since we
         * aren't connected as tightly as we were in SDL2. */
        for (i = 0; devices[i]; i++) {
            if (!GestureGetTouch(devices[i])) {
                GestureAddTouch(devices[i]);
            }
        }
        SDL_free(devices);
    }

    if (touchID != 0) {
        GestureRecordAll = true; /* !!! FIXME: this is never set back to false
                                    anywhere, that's probably a bug. */
        for (i = 0; i < GestureNumTouches; i++) {
            GestureTouches[i].recording = true;
        }
    }
    else {
        GestureTouch *touch = GestureGetTouch(touchID);
        if (!touch) {
            return 0; /* bogus touchid */
        }
        touch->recording = true;
    }

    return 1;
}

void
Gesture_Quit(void)
{
    SDL_RemoveEventWatch(GestureEventWatch, NULL);
    SDL_free(GestureTouches);
    GestureTouches = NULL;
    GestureNumTouches = 0;
    GestureRecordAll = false;
}

static Uint64
GestureHashDollar(SDL_FPoint *points)
{
    Uint64 hash = 5381;
    int i;
    for (i = 0; i < GESTURE_DOLLARNPOINTS; i++) {
        hash = ((hash << 5) + hash) + (Uint64)points[i].x;
        hash = ((hash << 5) + hash) + (Uint64)points[i].y;
    }
    return hash;
}

static int
GestureSaveTemplate(GestureDollarTemplate *templ, SDL_IOStream *dst)
{
    const size_t bytes = sizeof(templ->path[0]) * GESTURE_DOLLARNPOINTS;

    if (dst == NULL) {
        return 0;
    }

    /* No Longer storing the Hash, rehash on load */
    /* if (SDL_IOWrite(dst, &(templ->hash), sizeof(templ->hash)) !=
     * sizeof(templ->hash)) return 0; */

#if SDL_BYTEORDER == SDL_LIL_ENDIAN
    if (SDL_WriteIO(dst, templ->path, bytes) != bytes) {
        return 0;
    }
#else
    {
        GestureDollarTemplate copy = *templ;
        SDL_FPoint *p = copy.path;
        int i;
        for (i = 0; i < GESTURE_DOLLARNPOINTS; i++, p++) {
            p->x = SDL_SwapFloatLE(p->x);
            p->y = SDL_SwapFloatLE(p->y);
        }

        if (SDL_WriteIO(dst, copy.path, bytes) != bytes) {
            return 0;
        }
    }
#endif

    return 1;
}

SDL_DECLSPEC int SDLCALL
Gesture_SaveAllDollarTemplates(SDL_IOStream *dst)
{
    int i, j, rtrn = 0;
    for (i = 0; i < GestureNumTouches; i++) {
        GestureTouch *touch = &GestureTouches[i];
        for (j = 0; j < touch->numDollarTemplates; j++) {
            rtrn += GestureSaveTemplate(&touch->dollarTemplate[j], dst);
        }
    }
    return rtrn;
}

SDL_DECLSPEC int SDLCALL
Gesture_SaveDollarTemplate(Gesture_ID gestureId, SDL_IOStream *dst)
{
    int i, j;
    for (i = 0; i < GestureNumTouches; i++) {
        GestureTouch *touch = &GestureTouches[i];
        for (j = 0; j < touch->numDollarTemplates; j++) {
            if (touch->dollarTemplate[j].hash == (Uint64)gestureId) {
                return GestureSaveTemplate(&touch->dollarTemplate[j], dst);
            }
        }
    }
    return SDL_SetError("Unknown gestureId");
}

/* path is an already sampled set of points
Returns the index of the gesture on success, or -1 */
static int
GestureAddDollar_one(GestureTouch *inTouch, SDL_FPoint *path)
{
    GestureDollarTemplate *dollarTemplate;
    GestureDollarTemplate *templ;
    int index;

    index = inTouch->numDollarTemplates;
    dollarTemplate = (GestureDollarTemplate *)SDL_realloc(
        inTouch->dollarTemplate, (index + 1) * sizeof(GestureDollarTemplate));
    if (dollarTemplate == NULL) {
        return SDL_OutOfMemory();
    }
    inTouch->dollarTemplate = dollarTemplate;

    templ = &inTouch->dollarTemplate[index];
    SDL_memcpy(templ->path, path, GESTURE_DOLLARNPOINTS * sizeof(SDL_FPoint));
    templ->hash = GestureHashDollar(templ->path);
    inTouch->numDollarTemplates++;

    return index;
}

static int
GestureAddDollar(GestureTouch *inTouch, SDL_FPoint *path)
{
    int index = -1;
    int i = 0;
    if (inTouch == NULL) {
        if (GestureNumTouches == 0) {
            return SDL_SetError("no gesture touch devices registered");
        }
        for (i = 0; i < GestureNumTouches; i++) {
            inTouch = &GestureTouches[i];
            index = GestureAddDollar_one(inTouch, path);
            if (index < 0) {
                return -1;
            }
        }
        /* Use the index of the last one added. */
        return index;
    }
    return GestureAddDollar_one(inTouch, path);
}

static void
SDL_RemoveDollarTemplate_one(GestureTouch *inTouch, int index)
{
    if (index < inTouch->numDollarTemplates - 1) {
        SDL_memmove(&inTouch->dollarTemplate[index],
                    &inTouch->dollarTemplate[index + 1],
                    (inTouch->numDollarTemplates - 1 - index) *
                        sizeof(GestureDollarTemplate));
    }
    if (inTouch->numDollarTemplates > 1) {
        inTouch->dollarTemplate = SDL_realloc(
            inTouch->dollarTemplate,
            (inTouch->numDollarTemplates - 1) * sizeof(GestureDollarTemplate));
    }
    else {
        SDL_free(inTouch->dollarTemplate);
        inTouch->dollarTemplate = NULL;
    }
    --inTouch->numDollarTemplates;
}

int
SDL_RemoveDollarTemplate(Gesture_ID gestureId)
{
    int i, j, ret = 0;
    for (i = 0; i < GestureNumTouches; i++) {
        GestureTouch *touch = &GestureTouches[i];
        for (j = 0; j < touch->numDollarTemplates; j++) {
            if (touch->dollarTemplate[j].hash == (Uint64)gestureId) {
                SDL_RemoveDollarTemplate_one(touch, j);
                ret = 1;
            }
        }
    }
    return ret;
}

void
SDL_RemoveAllDollarTemplates(void)
{
    int i;
    for (i = 0; i < GestureNumTouches; i++) {
        GestureTouch *touch = &GestureTouches[i];
        SDL_free(touch->dollarTemplate);
        touch->dollarTemplate = NULL;
        touch->numDollarTemplates = 0;
    }
}

SDL_DECLSPEC int SDLCALL
Gesture_LoadDollarTemplates(SDL_TouchID touchID, SDL_IOStream *src)
{
    int i, loaded = 0;
    GestureTouch *touch = NULL;
    if (src == NULL) {
        return 0;
    }
    if (touchID >= 0) {
        for (i = 0; i < GestureNumTouches; i++) {
            if (GestureTouches[i].touchID == touchID) {
                touch = &GestureTouches[i];
            }
        }
        if (touch == NULL) {
            return SDL_SetError("given touch id not found");
        }
    }

    while (1) {
        GestureDollarTemplate templ;
        const size_t bytes = sizeof(templ.path[0]) * GESTURE_DOLLARNPOINTS;

        if (SDL_ReadIO(src, templ.path, bytes) < bytes) {
            if (loaded == 0) {
                return SDL_SetError(
                    "could not read any dollar gesture from rwops");
            }
            break;
        }

#if SDL_BYTEORDER != SDL_LIL_ENDIAN
        for (i = 0; i < GESTURE_DOLLARNPOINTS; i++) {
            SDL_FPoint *p = &templ.path[i];
            p->x = SDL_SwapFloatLE(p->x);
            p->y = SDL_SwapFloatLE(p->y);
        }
#endif

        if (touchID >= 0) {
            /* printf("Adding loaded gesture to 1 touch\n"); */
            if (GestureAddDollar(touch, templ.path) >= 0) {
                loaded++;
            }
        }
        else {
            /* printf("Adding to: %i touches\n",GestureNumTouches); */
            for (i = 0; i < GestureNumTouches; i++) {
                touch = &GestureTouches[i];
                /* printf("Adding loaded gesture to + touches\n"); */
                /* TODO: What if this fails? */
                GestureAddDollar(touch, templ.path);
            }
            loaded++;
        }
    }

    return loaded;
}

static float
GestureDollarDifference(SDL_FPoint *points, SDL_FPoint *templ, float ang)
{
    /*  SDL_FPoint p[GESTURE_DOLLARNPOINTS]; */
    float dist = 0;
    SDL_FPoint p;
    int i;
    for (i = 0; i < GESTURE_DOLLARNPOINTS; i++) {
        p.x = points[i].x * SDL_cosf(ang) - points[i].y * SDL_sinf(ang);
        p.y = points[i].x * SDL_sinf(ang) + points[i].y * SDL_cosf(ang);
        dist += SDL_sqrtf((p.x - templ[i].x) * (p.x - templ[i].x) +
                          (p.y - templ[i].y) * (p.y - templ[i].y));
    }
    return dist / GESTURE_DOLLARNPOINTS;
}

static float
GestureBestDollarDifference(SDL_FPoint *points, SDL_FPoint *templ)
{
    /*------------BEGIN DOLLAR BLACKBOX------------------
      -TRANSLATED DIRECTLY FROM PSUDEO-CODE AVAILABLE AT-
      -"http://depts.washington.edu/aimgroup/proj/dollar/"
    */
    double ta = -SDL_PI_D / 4;
    double tb = SDL_PI_D / 4;
    double dt = SDL_PI_D / 90;
    float x1 = (float)(GESTURE_PHI * ta + (1 - GESTURE_PHI) * tb);
    float f1 = GestureDollarDifference(points, templ, x1);
    float x2 = (float)((1 - GESTURE_PHI) * ta + GESTURE_PHI * tb);
    float f2 = GestureDollarDifference(points, templ, x2);
    while (SDL_fabs(ta - tb) > dt) {
        if (f1 < f2) {
            tb = x2;
            x2 = x1;
            f2 = f1;
            x1 = (float)(GESTURE_PHI * ta + (1 - GESTURE_PHI) * tb);
            f1 = GestureDollarDifference(points, templ, x1);
        }
        else {
            ta = x1;
            x1 = x2;
            f1 = f2;
            x2 = (float)((1 - GESTURE_PHI) * ta + GESTURE_PHI * tb);
            f2 = GestureDollarDifference(points, templ, x2);
        }
    }
    /*
      if (f1 <= f2)
          printf("Min angle (x1): %f\n",x1);
      else if (f1 >  f2)
          printf("Min angle (x2): %f\n",x2);
    */
    return SDL_min(f1, f2);
}

/* `path` contains raw points, plus (possibly) the calculated length */
static int
GestureDollarNormalize(const GestureDollarPath *path, SDL_FPoint *points,
                       bool is_recording)
{
    int i;
    float interval;
    float dist;
    int numPoints = 0;
    SDL_FPoint centroid;
    float xmin, xmax, ymin, ymax;
    float ang;
    float w, h;
    float length = path->length;

    /* Calculate length if it hasn't already been done */
    if (length <= 0) {
        for (i = 1; i < path->numPoints; i++) {
            const float dx = path->p[i].x - path->p[i - 1].x;
            const float dy = path->p[i].y - path->p[i - 1].y;
            length += SDL_sqrtf(dx * dx + dy * dy);
        }
    }

    /* Resample */
    interval = length / (GESTURE_DOLLARNPOINTS - 1);
    dist = interval;

    centroid.x = 0;
    centroid.y = 0;

    /* printf("(%f,%f)\n",path->p[path->numPoints-1].x,path->p[path->numPoints-1].y);
     */
    for (i = 1; i < path->numPoints; i++) {
        const float d = SDL_sqrtf((path->p[i - 1].x - path->p[i].x) *
                                      (path->p[i - 1].x - path->p[i].x) +
                                  (path->p[i - 1].y - path->p[i].y) *
                                      (path->p[i - 1].y - path->p[i].y));
        /* printf("d = %f dist = %f/%f\n",d,dist,interval); */
        while (dist + d > interval) {
            points[numPoints].x =
                path->p[i - 1].x +
                ((interval - dist) / d) * (path->p[i].x - path->p[i - 1].x);
            points[numPoints].y =
                path->p[i - 1].y +
                ((interval - dist) / d) * (path->p[i].y - path->p[i - 1].y);
            centroid.x += points[numPoints].x;
            centroid.y += points[numPoints].y;
            numPoints++;

            dist -= interval;
        }
        dist += d;
    }
    if (numPoints < GESTURE_DOLLARNPOINTS - 1) {
        if (is_recording) {
            SDL_SetError("ERROR: NumPoints = %i", numPoints);
        }
        return 0;
    }
    /* copy the last point */
    points[GESTURE_DOLLARNPOINTS - 1] = path->p[path->numPoints - 1];
    numPoints = GESTURE_DOLLARNPOINTS;

    centroid.x /= numPoints;
    centroid.y /= numPoints;

    /* printf("Centroid (%f,%f)",centroid.x,centroid.y); */
    /* Rotate Points so point 0 is left of centroid and solve for the bounding
     * box */
    xmin = centroid.x;
    xmax = centroid.x;
    ymin = centroid.y;
    ymax = centroid.y;

    ang = SDL_atan2f(centroid.y - points[0].y, centroid.x - points[0].x);

    for (i = 0; i < numPoints; i++) {
        const float px = points[i].x;
        const float py = points[i].y;
        points[i].x = (px - centroid.x) * SDL_cosf(ang) -
                      (py - centroid.y) * SDL_sinf(ang) + centroid.x;
        points[i].y = (px - centroid.x) * SDL_sinf(ang) +
                      (py - centroid.y) * SDL_cosf(ang) + centroid.y;

        if (points[i].x < xmin) {
            xmin = points[i].x;
        }
        if (points[i].x > xmax) {
            xmax = points[i].x;
        }
        if (points[i].y < ymin) {
            ymin = points[i].y;
        }
        if (points[i].y > ymax) {
            ymax = points[i].y;
        }
    }

    /* Scale points to GESTURE_DOLLARSIZE, and translate to the origin */
    w = xmax - xmin;
    h = ymax - ymin;

    for (i = 0; i < numPoints; i++) {
        points[i].x = (points[i].x - centroid.x) * GESTURE_DOLLARSIZE / w;
        points[i].y = (points[i].y - centroid.y) * GESTURE_DOLLARSIZE / h;
    }
    return numPoints;
}

static float
GestureDollarRecognize(const GestureDollarPath *path, int *bestTempl,
                       GestureTouch *touch)
{
    SDL_FPoint points[GESTURE_DOLLARNPOINTS];
    int i;
    float bestDiff = 10000;

    SDL_memset(points, 0, sizeof(points));

    GestureDollarNormalize(path, points, false);

    /* PrintPath(points); */
    *bestTempl = -1;
    for (i = 0; i < touch->numDollarTemplates; i++) {
        const float diff =
            GestureBestDollarDifference(points, touch->dollarTemplate[i].path);
        if (diff < bestDiff) {
            bestDiff = diff;
            *bestTempl = i;
        }
    }
    return bestDiff;
}

static void
GestureSendMulti(GestureTouch *touch, float dTheta, float dDist)
{
    if (SDL_EventEnabled(GESTURE_MULTIGESTURE)) {
        Gesture_MultiGestureEvent mgesture;
        mgesture.type = GESTURE_MULTIGESTURE;
        mgesture.timestamp = 0;
        mgesture.touchID = touch->touchID;
        mgesture.x = touch->centroid.x;
        mgesture.y = touch->centroid.y;
        mgesture.dTheta = dTheta;
        mgesture.dDist = dDist;
        mgesture.numFingers = touch->numDownFingers;
        SDL_PushEvent((SDL_Event *)&mgesture);
    }
}

static void
GestureSendDollar(GestureTouch *touch, Gesture_ID gestureId, float error)
{
    if (SDL_EventEnabled(GESTURE_DOLLARGESTURE)) {
        Gesture_DollarGestureEvent dgesture;
        dgesture.type = GESTURE_DOLLARGESTURE;
        dgesture.timestamp = 0;
        dgesture.touchID = touch->touchID;
        dgesture.x = touch->centroid.x;
        dgesture.y = touch->centroid.y;
        dgesture.gestureId = gestureId;
        dgesture.error = error;
        /* A finger came up to trigger this event. */
        dgesture.numFingers = touch->numDownFingers + 1;
        SDL_PushEvent((SDL_Event *)&dgesture);
    }
}

static void
GestureSendDollarRecord(GestureTouch *touch, Gesture_ID gestureId)
{
    if (SDL_EventEnabled(GESTURE_DOLLARRECORD)) {
        Gesture_DollarGestureEvent dgesture;
        dgesture.type = GESTURE_DOLLARRECORD;
        dgesture.timestamp = 0;
        dgesture.touchID = touch->touchID;
        dgesture.gestureId = gestureId;
        SDL_PushEvent((SDL_Event *)&dgesture);
    }
}

static void
GestureProcessEvent(const SDL_Event *event)
{
    float x, y;
    int index;
    int i;
    float pathDx, pathDy;
    SDL_FPoint lastP;
    SDL_FPoint lastCentroid;
    float lDist;
    float Dist;
    float dtheta;
    float dDist;

    if (event->type == SDL_EVENT_FINGER_MOTION ||
        event->type == SDL_EVENT_FINGER_DOWN ||
        event->type == SDL_EVENT_FINGER_UP) {
        GestureTouch *inTouch = GestureGetTouch(event->tfinger.touchID);

        if (inTouch == NULL) { /* we maybe didn't see this one before. */
            inTouch = GestureAddTouch(event->tfinger.touchID);
            if (!inTouch) {
                return; /* oh well. */
            }
        }

        x = event->tfinger.x;
        y = event->tfinger.y;

        /* Finger Up */
        if (event->type == SDL_EVENT_FINGER_UP) {
            SDL_FPoint path[GESTURE_DOLLARNPOINTS];
            inTouch->numDownFingers--;

            if (inTouch->recording) {
                inTouch->recording = false;
                GestureDollarNormalize(&inTouch->dollarPath, path, true);
                /* PrintPath(path); */
                if (GestureRecordAll) {
                    index = GestureAddDollar(NULL, path);
                    for (i = 0; i < GestureNumTouches; i++) {
                        GestureTouches[i].recording = false;
                    }
                }
                else {
                    index = GestureAddDollar(inTouch, path);
                }

                if (index >= 0) {
                    Gesture_ID gestureId =
                        (Gesture_ID)inTouch->dollarTemplate[index].hash;
                    GestureSendDollarRecord(inTouch, gestureId);
                }
                else {
                    GestureSendDollarRecord(inTouch, -1);
                }
            }
            else {
                int bestTempl = -1;
                const float error = GestureDollarRecognize(
                    &inTouch->dollarPath, &bestTempl, inTouch);
                if (bestTempl >= 0) {
                    /* Send Event */
                    Gesture_ID gestureId =
                        (Gesture_ID)inTouch->dollarTemplate[bestTempl].hash;
                    GestureSendDollar(inTouch, gestureId, error);
                    /* printf ("%s\n",);("Dollar error: %f\n",error); */
                }
            }

            /* inTouch->gestureLast[j] =
             * inTouch->gestureLast[inTouch->numDownFingers]; */
            if (inTouch->numDownFingers > 0) {
                inTouch->centroid.x =
                    (inTouch->centroid.x * (inTouch->numDownFingers + 1) - x) /
                    inTouch->numDownFingers;
                inTouch->centroid.y =
                    (inTouch->centroid.y * (inTouch->numDownFingers + 1) - y) /
                    inTouch->numDownFingers;
            }
        }
        else if (event->type == SDL_EVENT_FINGER_CANCELED) {
            inTouch->numDownFingers--;

            if (inTouch->recording) {
                inTouch->recording = false;
                if (GestureRecordAll) {
                    for (i = 0; i < GestureNumTouches; i++) {
                        GestureTouches[i].recording = false;
                    }
                }
            }

            if (inTouch->numDownFingers > 0) {
                inTouch->centroid.x =
                    (inTouch->centroid.x * (inTouch->numDownFingers + 1) - x) /
                    inTouch->numDownFingers;
                inTouch->centroid.y =
                    (inTouch->centroid.y * (inTouch->numDownFingers + 1) - y) /
                    inTouch->numDownFingers;
            }
        }
        else if (event->type == SDL_EVENT_FINGER_MOTION) {
            const float dx = event->tfinger.dx;
            const float dy = event->tfinger.dy;
            GestureDollarPath *path = &inTouch->dollarPath;
            if (path->numPoints < GESTURE_MAX_DOLLAR_PATH_SIZE) {
                path->p[path->numPoints].x = inTouch->centroid.x;
                path->p[path->numPoints].y = inTouch->centroid.y;
                pathDx = (path->p[path->numPoints].x -
                          path->p[path->numPoints - 1].x);
                pathDy = (path->p[path->numPoints].y -
                          path->p[path->numPoints - 1].y);
                path->length +=
                    (float)SDL_sqrt(pathDx * pathDx + pathDy * pathDy);
                path->numPoints++;
            }

            lastP.x = x - dx;
            lastP.y = y - dy;
            lastCentroid = inTouch->centroid;

            inTouch->centroid.x += dx / inTouch->numDownFingers;
            inTouch->centroid.y += dy / inTouch->numDownFingers;
            /* printf("Centrid :
             * (%f,%f)\n",inTouch->centroid.x,inTouch->centroid.y); */
            if (inTouch->numDownFingers > 1) {
                SDL_FPoint lv; /* Vector from centroid to last x,y position */
                SDL_FPoint
                    v; /* Vector from centroid to current x,y position */
                /* lv = inTouch->gestureLast[j].cv; */
                lv.x = lastP.x - lastCentroid.x;
                lv.y = lastP.y - lastCentroid.y;
                lDist = SDL_sqrtf(lv.x * lv.x + lv.y * lv.y);
                /* printf("lDist = %f\n",lDist); */
                v.x = x - inTouch->centroid.x;
                v.y = y - inTouch->centroid.y;
                /* inTouch->gestureLast[j].cv = v; */
                Dist = SDL_sqrtf(v.x * v.x + v.y * v.y);
                /* SDL_cosf(dTheta) = (v . lv)/(|v| * |lv|) */

                /* Normalize Vectors to simplify angle calculation */
                lv.x /= lDist;
                lv.y /= lDist;
                v.x /= Dist;
                v.y /= Dist;
                dtheta = SDL_atan2f(lv.x * v.y - lv.y * v.x,
                                    lv.x * v.x + lv.y * v.y);

                dDist = (Dist - lDist);
                if (lDist == 0) {
                    /* To avoid impossible values */
                    dDist = 0;
                    dtheta = 0;
                }

                /* inTouch->gestureLast[j].dDist = dDist;
                inTouch->gestureLast[j].dtheta = dtheta;

                printf("dDist = %f, dTheta = %f\n",dDist,dtheta);
                gdtheta = gdtheta*.9 + dtheta*.1;
                gdDist  =  gdDist*.9 +  dDist*.1
                knob.r += dDist/numDownFingers;
                knob.ang += dtheta;
                printf("thetaSum = %f, distSum = %f\n",gdtheta,gdDist);
                printf("id: %i dTheta = %f, dDist = %f\n",j,dtheta,dDist); */
                GestureSendMulti(inTouch, dtheta, dDist);
            }
            else {
                /* inTouch->gestureLast[j].dDist = 0;
                inTouch->gestureLast[j].dtheta = 0;
                inTouch->gestureLast[j].cv.x = 0;
                inTouch->gestureLast[j].cv.y = 0; */
            }
            /* inTouch->gestureLast[j].f.p.x = x;
            inTouch->gestureLast[j].f.p.y = y;
            break;
            pressure? */
        }
        else if (event->type == SDL_EVENT_FINGER_DOWN) {
            inTouch->numDownFingers++;
            inTouch->centroid.x =
                (inTouch->centroid.x * (inTouch->numDownFingers - 1) + x) /
                inTouch->numDownFingers;
            inTouch->centroid.y =
                (inTouch->centroid.y * (inTouch->numDownFingers - 1) + y) /
                inTouch->numDownFingers;
            /* printf("Finger Down: (%f,%f). Centroid: (%f,%f\n",x,y,
                 inTouch->centroid.x,inTouch->centroid.y); */

            inTouch->dollarPath.length = 0;
            inTouch->dollarPath.p[0].x = x;
            inTouch->dollarPath.p[0].y = y;
            inTouch->dollarPath.numPoints = 1;
        }
    }
}

#endif /* defined(SDL_GESTURE_IMPLEMENTATION) */
#endif /* SDL version > 2 */
#endif /* INCL_SDL_GESTURE_H */

/* vi: set sts=4 ts=4 sw=4 expandtab: */
