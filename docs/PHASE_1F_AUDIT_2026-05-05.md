# Phase 1F — Self-Healing Patterns Audit (2026-05-05)

**Auditor:** Runa Gridweaver Freyjasdottir, executing on Volmarr's behalf during the autonomous-mode cross-project housekeeping pass.
**Branch:** `development`
**Status:** **CLOSED — 2 defects found and fixed.**

The Phase 1F scope was two concrete sub-items per `TASK_pygame_viking_edition.md` lines 477-479:

1. SDL_Init failure recovery paths.
2. Error context enrichment (SDL_GetError propagation).

This document records the findings + the fixes that landed in the same commit.

---

## Audit method

Surveyed every `SDL_Init` and `SDL_InitSubSystem` call in `src_c/`:

| File | Line | Pattern | Verdict |
|---|---|---|---|
| `base.c` | 352 | `SDL_Init(EVENTTHREAD|TIMER|NOPARACHUTE)` — return captured to `pg_sdl_was_init` but never warned-on | ⚠ DEFECT |
| `base.c` | 355 | Same as 352 (non-thread branch) | ⚠ DEFECT |
| `display.c` | 248 | `SDL_InitSubSystem(VIDEO)` → RAISE(pgExc_SDLError, SDL_GetError()) on failure | ✓ CLEAN |
| `joystick.c` | 45 | Same RAISE pattern | ✓ CLEAN |
| `mixer.c` | 447 | Same RAISE pattern | ✓ CLEAN |
| `scrap_sdl2.c` | 53 | `SDL_Init(VIDEO)` — **return value discarded entirely** | ⚠ DEFECT |
| `time.c` | 263, 318, 380, 427, 548 | `SDL_InitSubSystem(TIMER)` → PyErr_SetString(pgExc_SDLError, SDL_GetError()) on failure | ✓ CLEAN (5 sites) |

**Summary:** 8 of 10 SDL_Init / SDL_InitSubSystem call sites already had clean recovery paths. Two sites swallowed errors.

---

## Defect 1 — `scrap_sdl2.c:53` — return value discarded

### Before

```c
int
pygame_scrap_init(void)
{
    SDL_Init(SDL_INIT_VIDEO);

    pygame_scrap_types = malloc(sizeof(char *) * 2);
    if (!pygame_scrap_types) {
        return 0;
    }
    ...
    _scrapinitialized = 1;
    return _scrapinitialized;
}
```

### Problem

`SDL_Init(SDL_INIT_VIDEO)` ignored the return value. If SDL_Init failed (rare but possible — e.g. driver not present), `pygame_scrap_init` proceeded to malloc + flag itself initialized even though the underlying video subsystem wasn't actually up.

The caller at `scrap.c:119` already had the right pattern:

```c
if (!pygame_scrap_init())
    return RAISE(pgExc_SDLError, SDL_GetError());
```

— but it never triggered because `pygame_scrap_init` returned 1 even on failure. Operators saw cryptic clipboard failures later instead of a clear "scrap init failed" diagnostic.

### Fix

Check the SDL_Init return; return 0 on failure so the caller's existing RAISE path surfaces the SDL error. Plus a comment block explaining why this code path is rare in practice (VIDEO_INIT_CHECK runs before pygame_scrap_init in the call sequence) but the fix is defensive.

```c
int
pygame_scrap_init(void)
{
    /* Phase 1F (Self-Healing 2026-05-05) — see commit body. */
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        return 0;
    }

    pygame_scrap_types = malloc(sizeof(char *) * 2);
    ...
}
```

Risk: minimal. The new return-0 branch is reached only when SDL_Init returns non-zero, which was already a quietly-broken state.

---

## Defect 2 — `base.c:352-356` — top-level init failure swallowed

### Before

```c
#if defined(WITH_THREAD) && !defined(MS_WIN32) && defined(SDL_INIT_EVENTTHREAD)
    pg_sdl_was_init = SDL_Init(SDL_INIT_EVENTTHREAD | SDL_INIT_TIMER |
                               SDL_INIT_NOPARACHUTE) == 0;
#else
    pg_sdl_was_init = SDL_Init(SDL_INIT_TIMER | SDL_INIT_NOPARACHUTE) == 0;
#endif
```

### Problem

`pg_sdl_was_init` captured the success/failure as a boolean used by the atexit handler to decide whether to call `SDL_Quit`. But the SDL error message itself was lost — when SDL_Init genuinely failed, operators saw no diagnostic at startup. Each submodule (display, joystick, mixer, time) lazily retries `SDL_InitSubSystem` for its specific subsystem, so the call wasn't fatal — but the lack of a top-level warning meant operators only saw cryptic per-submodule errors much later.

### Fix

Snapshot `SDL_GetError()` into a stack-local buffer before any further SDL call could clobber it, then emit a `PyExc_RuntimeWarning` so operators see the failure at startup. The atexit-handler logic at line 386 is unchanged — it still gates `SDL_Quit` on `pg_sdl_was_init`.

```c
    if (!pg_sdl_was_init) {
        char sdl_err_snapshot[512];
        const char *sdl_err = SDL_GetError();
        if (sdl_err && sdl_err[0]) {
            SDL_strlcpy(sdl_err_snapshot, sdl_err,
                        sizeof(sdl_err_snapshot));
        }
        else {
            SDL_strlcpy(sdl_err_snapshot, "(no SDL error message available)",
                        sizeof(sdl_err_snapshot));
        }
        PyErr_WarnEx(PyExc_RuntimeWarning, sdl_err_snapshot, 1);
        PyErr_Clear();
    }
```

Risk: minimal. The warning is non-fatal. `PyErr_WarnEx` returns -1 if the warning is configured as an error (operator opt-in via `python -W error`); the call's return is intentionally discarded + the error state cleared via `PyErr_Clear()` so startup can continue exactly as before. An operator who DOES want startup-fail-on-warning gets that semantics by their own filter config — Phase 1F doesn't change pygame's default behavior.

---

## Error-context enrichment audit

Every site that does pass error context through SDL_GetError uses one of two patterns:

- `RAISE(pgExc_SDLError, SDL_GetError())` — for module init paths that propagate to a Python caller (display.c, joystick.c, mixer.c).
- `PyErr_SetString(pgExc_SDLError, SDL_GetError())` — for callbacks where the C code can't directly raise (time.c).

Both patterns capture the SDL error string at call site. There's a subtle thread-safety note: `SDL_GetError` is thread-local on SDL2, so as long as the captured string is consumed before the next SDL call on the same thread, the message survives. All audited sites consume the string immediately into the Python exception or warning system — no race.

**Verdict:** error-context propagation is already well-covered across the codebase. No additional changes needed.

---

## Closeout

Phase 1F is closed at `development` HEAD. Two real defects fixed:

1. `scrap_sdl2.c` SDL_Init return value now propagated through the existing 0-return contract.
2. `base.c` top-level SDL_Init failure now surfaces a `RuntimeWarning` with the SDL error string instead of being silently swallowed.

Together these close the "operator can't tell what went wrong" failure mode that survived Phase 1A-1E. Both are defensive fixes against rare init failures (most operator setups have SDL_Init succeed unconditionally), so the change is high-leverage diagnostic improvement at low behavioral risk.

The next phase per `TASK_pygame_viking_edition.md` is Phase 2 (cross-platform expansion). Phase 1 (defensive C surface) is done.
