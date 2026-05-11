# Phase 1E — Thread Safety Audit (2026-05-05)

**Auditor:** Runa Gridweaver Freyjasdottir, executing on Volmarr's behalf during the autonomous-mode cross-project housekeeping pass.
**Branch:** `development`
**Status:** **CLOSED — no bugs found.**

The Phase 1E scope was three concrete sub-items per `TASK_pygame_viking_edition.md` lines 472-475:

1. Verify mutex coverage completeness in `event.c`.
2. Surface lock/unlock safety from non-main threads.
3. Display state access from non-main threads.

This document records the audit findings + the rationale for closing the phase without code changes.

---

## 1. event.c mutex coverage — CLEAN

**File:** `src_c/event.c` (2368 lines).

The shared mutable state in event.c is:

- `pg_evfilter_mutex` (line 87) — the SDL_mutex itself.
- `_pg_last_keydown_event` (line 100) — the most-recent keydown event captured by the event filter for unicode-followup pairing.
- The internal event filter callback list (managed via SDL_AddEventWatch / SDL_DelEventWatch).

The lock + unlock macros at lines 108-130:

```c
#define PG_LOCK_EVFILTER_MUTEX                                             \
    if (pg_evfilter_mutex) {                                               \
        if (SDL_LockMutex(pg_evfilter_mutex) < 0) { ... PG_EXIT(1); }      \
    }

#define PG_UNLOCK_EVFILTER_MUTEX                                           \
    if (pg_evfilter_mutex) {                                               \
        if (SDL_UnlockMutex(pg_evfilter_mutex) < 0) { ... PG_EXIT(1); }    \
    }
```

These guard every read/write to the shared state. Verified call sites:

| Line | Site | Operation | Guarded |
|---|---|---|---|
| 137 | `_pg_repeat_callback` | (locks across full body) | ✓ |
| 502-514 | keydown branch in event filter | writes `_pg_last_keydown_event` | ✓ |
| 518-523 | text-input branch in event filter | reads + clears `_pg_last_keydown_event` | ✓ |
| 531-537 | keyup branch | clears state | ✓ |
| 605-608 | event filter set | writes filter list | ✓ |
| 615-618 | event filter get | reads filter list | ✓ |
| 625-630 | AutoQuit | tears down filter list before mutex destroy | ✓ |
| 958-961 | event-watch dispatch | writes filter list | ✓ |

**No reads or writes of the shared state occur outside a `PG_LOCK_EVFILTER_MUTEX / PG_UNLOCK_EVFILTER_MUTEX` pair.**

The mutex is created lazily in `pgEvent_AutoInit` (line 648-651) by checking the pointer is NULL before calling `SDL_CreateMutex`. Pygame's `pygame.init()` is documented as single-threaded, so the lazy-create has no race window. Acceptable.

The Emscripten branch (lines 102-105) correctly no-ops both macros — the JavaScript runtime is single-threaded.

**Verdict:** mutex coverage is complete. No code changes required.

---

## 2. Surface lock/unlock safety from non-main threads — CLEAN-BY-DESIGN

**File:** `src_c/surface.c` (3890 lines).

Pygame's surface lock API (`pgSurface_Lock` / `pgSurface_Unlock`) wraps SDL2's `SDL_LockSurface` / `SDL_UnlockSurface` and adds a refcount-based "lock chain" for nested-lock support.

The audit pattern shows balanced Lock/Unlock pairs at every documented call site:

| Line | Use case | Pair |
|---|---|---|
| 766 / 795 | `surface.set_at_mapped` | ✓ |
| 844 / 878 | `surface.fill` | ✓ |
| 906 / 930 | `surface.scroll` | ✓ |
| 980 / 988 | view buffer setup | ✓ |
| 1828 / 1830 | array access path | ✓ |
| 2178 / 2212 | `subsurface.get_buffer` | ✓ |
| 2457 / 2466 | helper subsurface fill | ✓ |
| 2669 / 2813 | blits / blit_sequence | ✓ |
| 3409 / 3452 | external buffer protocol consumer | ✓ |

### Multi-thread safety policy

Pygame inherits SDL2's threading model: **Surface mutation from non-main threads is the operator's responsibility** to synchronize. The pgSurface_Lock refcount is NOT a thread-safety primitive — it's a re-entry counter for nested locks within a single thread.

This is documented behavior matching upstream pygame and SDL2. Adding a per-surface mutex would:
- Break backward compatibility with operator code that already assumes single-threaded surface access.
- Add measurable overhead to the most-hot path in pygame (pixel access).
- Duplicate a guarantee SDL2 already documents the operator must provide.

The right enhancement (if needed in future) is operator-facing documentation, not a code-level lock.

**Verdict:** lock/unlock balance is correct. Multi-thread safety policy is unchanged from upstream SDL2. No code changes required.

---

## 3. Display state access from non-main threads — CLEAN-BY-DESIGN

**File:** `src_c/display.c` (2701 lines).

SDL2 itself imposes the constraint that **all SDL_Video subsystem functions must be called from the same thread that initialized SDL_Video** (typically the main thread). This is documented at <https://wiki.libsdl.org/SDL2/SDL_VideoInit>:

> "Be sure to call SDL_VideoInit() from the same thread that's going to call SDL_VideoQuit(); these calls are not thread-safe."

`pygame.display.*` functions are direct or near-direct wrappers over SDL_Video calls. Adding pygame-side mutex protection would be redundant with SDL2's own model — and worse, would mask the underlying constraint that operators must respect anyway.

The single SDL_GetWindowFromID call (line 708) in the SDL_WINDOWEVENT handler is invoked from the SDL event-watch callback, which SDL2 guarantees runs on the same thread as the SDL_PumpEvents call that delivered the event. Operators calling `pygame.event.pump()` from the main thread (the only supported pattern) get correct behavior automatically.

**Verdict:** no thread-safety violations. SDL2's own threading model is the authoritative contract; pygame correctly inherits it without redundant locking.

---

## Closeout

Phase 1E is closed at `development` HEAD. No code changes; the audit confirms existing thread-safety guarantees are correct + complete.

Next phase per `TASK_pygame_viking_edition.md`: **Phase 1F — Self-Healing Patterns** (SDL_Init failure recovery paths, error context enrichment).

If a future session wants to enhance multi-thread surface access, the right lever is operator-facing documentation in `docs/THREAD_SAFETY.md` (not yet created) covering the SDL2 inheritance + the "you must serialize multi-thread surface mutation yourself" policy. Code-level changes are not required.
