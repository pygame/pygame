# pygame Viking Edition — Test Coverage Map

**Last Updated:** 2026-04-05  

---

## Test Suite Overview

The pygame test suite lives in `test/`. It uses Python's `unittest` framework with a custom parallel test runner (`test/run_tests.py`).

```
test/
├── run_tests.py          # Parallel test runner
├── util/                 # Test utilities
│   ├── __init__.py
│   └── ...               # Helpers for display mock, tags, etc.
├── pygame_test/          # Test support package
│   └── __init__.py
└── *_test.py             # Per-module test files
```

---

## Module → Test File Mapping

| Module | Primary Test File | Secondary Tests |
|---|---|---|
| `pygame.base` | `base_test.py` | — |
| `pygame.display` | `display_test.py` | `pixelarray_test.py` (display-dependent) |
| `pygame.surface` | `surface_test.py` | `pixelarray_test.py`, `surfarray_test.py` |
| `pygame.event` | `event_test.py` | — |
| `pygame.draw` | `draw_test.py` | `draw_py_test.py` (pure Python comparison) |
| `pygame.transform` | `transform_test.py` | — |
| `pygame.rect` | `rect_test.py` | — |
| `pygame.color` | `color_test.py` | — |
| `pygame.image` | `image_test.py` | — |
| `pygame.font` | `font_test.py` | — |
| `pygame.freetype` | `freetype_test.py` | — |
| `pygame.mixer` | `mixer_test.py` | — |
| `pygame.mixer.music` | `mixer_music_test.py` | — |
| `pygame.key` | `key_test.py` | — |
| `pygame.mouse` | `mouse_test.py` | — |
| `pygame.joystick` | `joystick_test.py` | — |
| `pygame.time` | `time_test.py` | — |
| `pygame.mask` | `mask_test.py` | — |
| `pygame.math` | `math_test.py` | — |
| `pygame.pixelarray` | `pixelarray_test.py` | — |
| `pygame.pixelcopy` | `pixelcopy_test.py` | — |
| `pygame.surfarray` | `surfarray_test.py` | — |
| `pygame.sprite` | `sprite_test.py` | — |
| `pygame.scrap` | `scrap_test.py` | — |
| `pygame.camera` | `camera_test.py` | — |
| `pygame.midi` | `midi_test.py` | — |
| `pygame.gfxdraw` | `gfxdraw_test.py` | — |
| `pygame.bufferproxy` | `bufferproxy_test.py` | — |
| `pygame.cursors` | `cursors_test.py` | — |
| `pygame.version` | `version_test.py` | — |
| `pygame._sdl2.video` | `_sdl2_video_test.py` | — |
| `pygame._sdl2.audio` | `_sdl2_audio_test.py` | — |
| `pygame._sdl2.controller` | `controller_test.py` | — |
| `pygame.fastevent` | `fastevent_test.py` | — |
| `pygame.sysfont` | `sysfont_test.py` | — |
| `pygame.colordict` | `colordict_test.py` | — |
| `pygame.locals` | `locals_test.py` | — |
| `pygame.pkgdata` | `pkgdata_test.py` | — |
| `pygame.threads` | `threads_test.py` | — |

---

## Running Tests

```bash
# All tests (parallel):
python test/run_tests.py

# With pytest:
python -m pytest test/

# Single module:
python -m pytest test/surface_test.py

# With verbose output:
python -m pytest test/ -v

# Skip tests that require display:
python -m pytest test/ -m "not interactive"
```

---

## Test Infrastructure

### Display Requirements

Many tests require an actual display (SDL2 window). Tests that need this:
- Use `pygame.display.init()` / `set_mode()` in setUp
- Are tagged with display requirements via the test framework
- May be skipped in headless CI environments

### Tag System

The test runner supports tags to skip/include test categories:
- `display` — needs a display window
- `audio` — needs audio hardware
- `network` — needs network (rare)
- `slow` — time-intensive tests
- `interactive` — requires user interaction

### `util/` Helpers

- Display surface mock for tests that don't need real rendering
- Random seed fixtures for reproducible fuzzing
- Platform-specific skip decorators

---

## Coverage Gaps (as of 2026-04-05)

### Known Untested or Under-Tested Areas

| Area | Gap | Priority |
|---|---|---|
| `simd_blitters_avx2.c` | No explicit SIMD path tests (only end-to-end blit tests) | Medium |
| `scrap.c` | Platform-specific paths hard to test in CI (clipboard access) | Low |
| `camera_v4l2.c` | Requires physical camera; often skipped | Medium |
| `midi.py` | Requires portmidi + physical MIDI device | Low |
| `scale_mmx*.c` | MMX paths — no explicit tests, covered by transform tests | Low |
| `freetype/ft_cache.c` | Cache eviction behavior untested | Medium |
| `event.c` unicode tracking | Edge cases with 15+ simultaneous keys | Low |
| `display.c` GL context | OpenGL path needs GPU; often untested in CI | High |
| `_sdl2.audio` | Capture mode largely untested | Medium |
| `newbuffer.c` | Legacy array interface paths | Low |

### Phase 0C Target — Test Gaps to Fix (Future Work)

1. **SIMD unit tests** — test SSE2/AVX2 blit paths explicitly with known pixel values
2. **FreeType cache tests** — fill cache beyond capacity, verify eviction
3. **Event unicode edge cases** — synthesize 15 simultaneous key events
4. **Platform clipboard tests** — mock clipboard or use SDL2 fallback
5. **OpenGL smoke tests** — minimal GL init/render/quit cycle

---

## Test Writing Guide (for Viking Edition additions)

When adding new features (Phases 3-7), each new module needs tests:

```python
# Template for a new test file:
import unittest
import pygame
import pygame.new_module  # your new module

class NewModuleTestCase(unittest.TestCase):
    def setUp(self):
        pygame.init()

    def tearDown(self):
        pygame.quit()

    def test_basic_init(self):
        pygame.new_module.init()
        self.assertTrue(pygame.new_module.get_init())

    def test_basic_quit(self):
        pygame.new_module.init()
        pygame.new_module.quit()
        self.assertFalse(pygame.new_module.get_init())

    # ... more tests

if __name__ == "__main__":
    unittest.main()
```

**Test file naming:** `test/<module_name>_test.py`

**Register in runner:** Add to `test/run_tests.py`'s test discovery list if not auto-discovered.
