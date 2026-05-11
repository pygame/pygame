# Structure: C Headers — `src_c/include/` + `src_c/*.h`

**Last reviewed:** 2026-04-05  

---

## Overview

pygame's C headers serve three purposes:
1. **Internal headers** — shared definitions within the pygame C codebase
2. **Public API headers** — define the Slot API for third-party C extensions
3. **Compatibility headers** — abstract compiler/platform differences

---

## `src_c/include/` — Public Headers (for external C extensions)

These are installed alongside pygame and can be used by third-party Cython/C extensions that extend pygame.

| Header | Purpose |
|---|---|
| `pygame.h` | Main public header. Just includes `_pygame.h`. Use this in external code |
| `_pygame.h` | Core definitions: Slot API macros, all module import macros, type definitions |
| `pgplatform.h` | Platform/compiler abstraction: `PG_INLINE`, `PG_FORCEINLINE`, `WIN32`, SIMD guards |
| `pgcompat.h` | SDL1/SDL2 compatibility macros (mostly SDL2-only now) |
| `pgimport.h` | Auto-generated: `import_pygame_*()` macros for each module |
| `pygame_bufferproxy.h` | BufferProxy type/slot declarations |
| `pygame_font.h` | Font type/slot declarations |
| `pygame_freetype.h` | FreeType font type/slot declarations |
| `pygame_mask.h` | Mask type/slot declarations |
| `pygame_mixer.h` | Mixer Sound/Channel type/slot declarations |
| `bitmask.h` | bitmask_t type and bitmask operation declarations |

---

## `_pygame.h` — The Master Header

The most important file in the entire codebase. Defines:

### Version Macros
```c
#define PG_MAJOR_VERSION 2
#define PG_MINOR_VERSION 6
#define PG_PATCH_VERSION 1
#define PG_VERSION_ATLEAST(MAJOR, MINOR, PATCH)  // comparison macro
```

### `pg_buffer` Type
```c
typedef void (*pybuffer_releaseproc)(Py_buffer *);
typedef struct pg_bufferinfo_s {
    Py_buffer view;
    PyObject *consumer;          // Borrowed reference
    pybuffer_releaseproc release_buffer;
} pg_buffer;
```
Extended `Py_buffer` that adds a per-instance release callback. Used throughout buffer protocol implementation.

### Slot API Macros (Core Mechanism)

```c
// How a module stores its API:
#define PYGAMEAPI_LOCAL_ENTRY "__api__"

// How another module retrieves a slot:
#define PYGAMEAPI_GET_SLOT(module, index) \
    (((void **)PyCapsule_GetPointer(      \
        PyObject_GetAttrString(           \
            PyImport_ImportModule("pygame." #module), \
            PYGAMEAPI_LOCAL_ENTRY),       \
        NULL))[index])

// Macro for each type:
// (Defined differently when PYGAMEAPI_*_INTERNAL is set)
#ifndef PYGAMEAPI_SURFACE_INTERNAL
#define pgSurface_Type    ((PyTypeObject*)PYGAMEAPI_GET_SLOT(surface, 0))
#define pgSurface_New(s)  (*(pgSurfaceObject*(*)(SDL_Surface*))PYGAMEAPI_GET_SLOT(surface, 1))(s)
#define pgSurface_Check(x) (PyObject_IsInstance((x), (PyObject*)&pgSurface_Type))
// ... etc for all exported symbols
#endif
```

### `import_pygame_*()` Macros

Each module defines an `import_pygame_module()` macro that retrieves and validates the Capsule:

```c
// Defined in pgimport.h (auto-generated):
#define import_pygame_base()    // Retrieve and validate pygame.base API capsule
#define import_pygame_surface() // Retrieve and validate pygame.surface API capsule
// ... one per module
```

---

## `src_c/*.h` — Internal Headers

| Header | Purpose |
|---|---|
| `pygame.h` | Just re-includes `_pygame.h` (for single include in top-level files) |
| `pgplatform.h` | Internal platform macros (MIN, MAX, ABS, warning macros) |
| `pgcompat.h` | Internal SDL1/SDL2 compatibility |
| `pgopengl.h` | OpenGL attribute constants for `display.gl_set_attribute()` |
| `pgarrinter.h` | numpy array interface (`PyArrayInterface`) struct definition |
| `pgbufferproxy.h` | Internal BufferProxy declarations |
| `surface.h` | Surface blend mode constants, pixel macros, blit function declarations |
| `_blit_info.h` | Blit info struct used between alphablit.c and surface.c |
| `_surface.h` | Internal Surface struct (pgSurfaceObject) definition |
| `simd_blitters.h` | SIMD function pointer declarations |
| `scale.h` | Scale function declarations (MMX/SSE scale paths) |
| `mask.h` | Mask type internal declarations |
| `mixer.h` | Mixer internal macros (`MIXER_INIT_CHECK`) |
| `camera.h` | Camera module internal declarations |
| `font.h` | Font module internal declarations |
| `freetype.h` | FreeType module internal declarations |
| `palette.h` | Palette utility functions |
| `pgcompat.h` | SDL compatibility shims |
| `sse2neon.h` | SSE2 → ARM NEON translation (large header, ~10k lines) |

---

## `pgimport.h` — Auto-Generated Import Macros

This file is typically auto-generated during the build process or maintained manually. It contains the `import_pygame_*()` function-like macros for each pygame module.

Pattern:
```c
#define import_pygame_base() \
    { \
        PyObject *_mod = PyImport_ImportModule("pygame.base"); \
        if (!_mod) return NULL; \
        PyObject *_api = PyObject_GetAttrString(_mod, PYGAMEAPI_LOCAL_ENTRY); \
        Py_DECREF(_mod); \
        if (!_api || !PyCapsule_CheckExact(_api)) { \
            PyErr_SetString(PyExc_ImportError, "missing pygame.base API"); \
            Py_XDECREF(_api); \
            return NULL; \
        } \
        pg_base_slots = (void **)PyCapsule_GetPointer(_api, NULL); \
        Py_DECREF(_api); \
    }
```

---

## `src_c/doc/*.h` — Docstring Headers

Auto-generated headers containing C string arrays for Python docstrings:
```c
// display_doc.h example:
#define DOC_PYGAMEDISPLAY \
    "pygame module to control the display window and screen\n"

#define DOC_PYGAMEDISPLAYINIT \
    "init() -> None\n" \
    "Initialize the display module\n"
```

Generated from the RST documentation in `docs/` via a build script. This ensures the `pygame.display.__doc__` strings match the actual documentation. The `src_c/doc/README.rst` explains the generation process.

Do not edit these headers manually — edit the RST source and regenerate.

---

## Key Type Definitions (in `_pygame.h`)

```c
// Forward declarations of all pygame Python types:
typedef struct pgSurfaceObject pgSurfaceObject;
typedef struct pgEventObject pgEventObject;
typedef struct pgRectObject pgRectObject;
typedef struct pgColorObject pgColorObject;
// ... etc.

// Each concrete definition is in its module's .c file
```

---

## Notes

- The header inclusion model: each `.c` file includes `pygame.h` (which includes `_pygame.h`) in the top-level file only. Other files in the same module include just `_pygame.h`. This is SDL's recommended pattern for extension modules.
- `pgopengl.h` contains GL attribute enum values that duplicate what's in PyOpenGL — they're defined here independently so pygame doesn't require PyOpenGL at build time.
- `sse2neon.h` is a third-party header from the ARM Intrinsics project. It's large (~10k lines) and provides imperfect but functional translation of SSE2 intrinsics to NEON. Real ARM optimization should use native NEON (Phase 2C).
