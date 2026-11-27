# Structure: SDL2 Cython Wrappers — `src_c/cython/pygame/_sdl2/`

**Type:** Cython + Python modules  
**Compiled to:** `pygame._sdl2.video`, `pygame._sdl2.audio`, `pygame._sdl2.controller`, `pygame._sdl2.mixer`  
**Files:** `video.pyx`, `video.pxd`, `audio.pyx`, `audio.pxd`, `controller.pyx`, `controller.pxd`, `mixer.pyx`, `mixer.pxd`, `sdl2.pxd`  
**Last reviewed:** 2026-04-05  

---

## Purpose

`pygame._sdl2` exposes **SDL2-specific APIs** that weren't available in pygame's SDL1.2 era design. These are "modern" pygame APIs — not backward compatible with old code but significantly more capable:

- **`video`**: Hardware-accelerated `Window`, `Renderer`, `Texture`, `Image` objects
- **`audio`**: Direct SDL2 audio device management (beyond SDL_mixer)
- **`controller`**: SDL2 game controller API with named button/axis mappings
- **`mixer`**: Alternative SDL2 mixer bindings

---

## `pygame._sdl2.video` — Hardware Renderer

This is the **modern rendering path** for pygame. Unlike `pygame.display` + Surface (which uses a software-blit-to-texture approach), `_sdl2.video` gives direct access to SDL2's hardware-accelerated renderer.

### Key Difference

```
Classic pygame path:
  Surface (CPU RAM) → SDL_UpdateTexture → SDL_RenderCopy → GPU → Screen
  (every frame, uploads all pixels to GPU)

_sdl2.video path:
  Texture (GPU VRAM) → SDL_RenderCopy → GPU → Screen
  (pixels stay on GPU; only updated when image changes)
```

### `Window`

```python
window = pygame._sdl2.video.Window(
    title="My Game",
    size=(800, 600),
    position=(x, y),           # or WINDOWPOS_CENTERED
    flags=0,                   # SDL window flags
    fullscreen=False,
    fullscreen_desktop=False,
    resizable=False,
    borderless=False,
    always_on_top=False,
)

window.title          # Get/set title
window.size           # Get/set (w, h)
window.position       # Get/set (x, y)
window.opacity        # Get/set float 0.0-1.0
window.grab           # Get/set mouse grab
window.resizable      # Get/set
window.borderless     # Get/set
window.id             # Window ID (for events)
window.display_index  # Monitor index

window.get_surface()        # Get classic Surface (for pygame.draw etc.)
window.flip()               # Update window (surface path)
window.set_icon(surface)
window.maximize()
window.minimize()
window.restore()
window.focus()
window.hide() / show()
window.set_modal_for(parent)
window.destroy()

# Class method:
Window.from_display_module()  # Get Window from pygame.display.set_mode()
```

### `Renderer`

```python
renderer = pygame._sdl2.video.Renderer(
    window,
    index=-1,               # -1 = auto-select driver
    accelerated=-1,         # -1 = prefer hardware
    vsync=False,
    target_texture=False    # Allow render-to-texture
)

renderer.draw_color = (r, g, b, a)    # Drawing color (for primitives)
renderer.blend_mode                    # SDL blend mode
renderer.logical_size = (w, h)         # Logical resolution (auto-scaled)
renderer.scale = (x, y)               # Manual scale
renderer.viewport = Rect              # Viewport clip rect
renderer.target = None / Texture      # Render target (None = screen)

# Drawing:
renderer.clear()                       # Fill with draw_color
renderer.present()                     # Flip to screen
renderer.draw_point((x, y))
renderer.draw_line((x1,y1), (x2,y2))
renderer.draw_rect(Rect)
renderer.fill_rect(Rect)
renderer.draw_triangle((x1,y1),(x2,y2),(x3,y3))  # outline
renderer.fill_triangle(...)            # filled
renderer.draw_quad(...)
renderer.fill_quad(...)

# Blitting:
renderer.blit(texture, dest, src, angle, origin, flip_x, flip_y)

# Texture creation:
renderer.to_surface(surface=None, area=None)  # Read back from GPU to CPU

# Class method:
Renderer.from_window(window)
```

### `Texture`

```python
# From Surface:
texture = pygame._sdl2.video.Texture.from_surface(renderer, surface)

# Blank texture:
texture = pygame._sdl2.video.Texture(
    renderer, size, depth=0, static=False, streaming=False, target=False
)

texture.width, texture.height
texture.alpha = 255          # Global alpha 0-255
texture.blend_mode           # SDL blend mode
texture.color = (r, g, b)   # Color modulation (multiplied with texture)

texture.draw(srcrect=None, dstrect=None, angle=0, origin=None, flip_x=False, flip_y=False)
texture.draw_triangle(...)
texture.draw_quad(...)
texture.update(surface, area=None)  # Update texture from surface (re-upload pixels)
texture.get_rect()
```

### `Image` — High-Level Texture Wrapper

```python
image = pygame._sdl2.video.Image(texture_or_image, srcrect=None)
image.draw(srcrect, dstrect)

# Modifiable defaults:
image.angle = 0
image.origin = None
image.flip_x, image.flip_y = False, False
image.color = (255, 255, 255)
image.alpha = 255
image.blend_mode = 0
```

---

## `pygame._sdl2.audio` — SDL2 Audio Devices

Direct access to SDL2's audio subsystem — independent of SDL_mixer.

```python
pygame._sdl2.audio.get_num_audio_devices(capture=False)  # Number of output/input devices
pygame._sdl2.audio.get_audio_device_name(index, capture=False)  # Device name

# Open an audio device:
device = pygame._sdl2.audio.AudioDevice(
    devicename=None,     # None = default device
    iscapture=False,     # True = microphone input
    frequency=44100,
    audioformat=pygame._sdl2.audio.AUDIO_S16SYS,
    numchannels=2,
    chunksize=512,
    allowed_changes=pygame._sdl2.audio.AUDIO_ALLOW_ANY_CHANGE,
)

device.pause(False)     # Unpause (start playing)
device.pause(True)      # Pause

# SoundStream:
stream = pygame._sdl2.audio.SoundStream(device)
stream.queue(audio_bytes)  # Queue raw PCM bytes for playback

# AudioBuffer:
buf = pygame._sdl2.audio.AudioBuffer(device, size)  # Allocate PCM buffer
```

Use cases:
- Capturing microphone input for voice or AI speech
- Low-latency audio streaming
- Multiple audio device management (speakers + headset simultaneously)
- Direct PCM synthesis without going through SDL_mixer

---

## `pygame._sdl2.controller` — Game Controller

SDL2's game controller API provides named button/axis mappings for XInput/DInput/PS controllers.

```python
pygame._sdl2.controller.init()
pygame._sdl2.controller.quit()
pygame._sdl2.controller.get_count()     # Number of connected controllers
pygame._sdl2.controller.is_controller(joystick_index)  # Has a known mapping?
pygame._sdl2.controller.name_forindex(joystick_index)  # Controller name
pygame._sdl2.controller.add_mapping(mapping_string)    # Add SDL2 gamepad mapping

ctrl = pygame._sdl2.controller.Controller(joystick_index)
ctrl.get_init()
ctrl.attached()     # Still connected?
ctrl.get_axis(axis_id)   # -32768 to 32767 (NOT -1.0 to 1.0 like Joystick!)
ctrl.get_button(button_id)  # 0 or 1
ctrl.get_hat(hat_id)        # (x, y)
ctrl.rumble(low_frequency, high_frequency, duration_ms)
ctrl.rumble_triggers(left, right, duration_ms)  # Trigger haptics (if supported)
ctrl.stop_rumble()
ctrl.as_joystick()   # Get underlying pygame.joystick.Joystick object
ctrl.id              # Device index
ctrl.name            # Controller name
```

### Axis Constants

| Constant | Description |
|---|---|
| `AXIS_LEFTX` | Left stick X axis |
| `AXIS_LEFTY` | Left stick Y axis |
| `AXIS_RIGHTX` | Right stick X axis |
| `AXIS_RIGHTY` | Right stick Y axis |
| `AXIS_TRIGGERLEFT` | Left trigger (0 to 32767) |
| `AXIS_TRIGGERRIGHT` | Right trigger |

### Button Constants

`BUTTON_A`, `BUTTON_B`, `BUTTON_X`, `BUTTON_Y`, `BUTTON_BACK`, `BUTTON_GUIDE`, `BUTTON_START`, `BUTTON_LEFTSTICK`, `BUTTON_RIGHTSTICK`, `BUTTON_LEFTSHOULDER`, `BUTTON_RIGHTSHOULDER`, `BUTTON_DPAD_UP/DOWN/LEFT/RIGHT`

---

## `sdl2.pxd` — SDL2 C Declarations

Cython declaration file (`.pxd`) that exposes SDL2 C types to all other Cython files:

```cython
# sdl2.pxd declares:
cdef extern from "SDL.h":
    ctypedef struct SDL_Window
    ctypedef struct SDL_Renderer
    ctypedef struct SDL_Texture
    ctypedef struct SDL_AudioDeviceID
    # ... etc.
```

This allows the `.pyx` files to use SDL2 types without redundant C declarations.

---

## `.pxd` Files — Cython Type Declarations

Each module has a paired `.pxd` file declaring the extension types for use by other Cython modules:

- `video.pxd` — declares `Window`, `Renderer`, `Texture`, `Image` cdef classes
- `audio.pxd` — declares `AudioDevice`, `AudioBuffer`, `SoundStream`
- `controller.pxd` — declares `Controller`
- `mixer.pxd` — declares SDL2 mixer types

---

## Comparison: Classic vs _sdl2 API

| Feature | Classic (`pygame.display`) | `_sdl2.video` |
|---|---|---|
| Rendering | Software blit to texture each frame | Direct GPU texture operations |
| Pixel access | Full (Surface is CPU RAM) | Limited (Texture is GPU VRAM) |
| Performance | Good for 2D sprites | Better for texture-heavy rendering |
| GPU features | None | Hardware scaling, rotation, blend |
| Render-to-texture | No | Yes (`target=True` Texture) |
| Multiple windows | No | Yes (multiple `Window` objects) |
| Backward compat | Full | Breaks old code |

---

## Known Quirks / Notes

- `_sdl2.video` and `pygame.display.set_mode()` can coexist but share SDL2's renderer. Use `Window.from_display_module()` to get the Window object for the display-created window.
- `Texture` pixels live on GPU — `texture.update(surface)` uploads from CPU to GPU, and `renderer.to_surface()` downloads from GPU to CPU. Both are expensive operations — avoid in hot paths.
- `Controller.get_axis()` returns integers in range `-32768` to `32767` (NOT floats like `Joystick.get_axis()`). Normalize: `value / 32767.0` for float range.
- `AudioDevice` in capture mode requires microphone permissions on macOS and Windows. On macOS, add `NSMicrophoneUsageDescription` to Info.plist if bundling with py2app.
- The `_sdl2` package is explicitly "unstable" in upstream pygame — its API may change between pygame minor versions. For Viking Edition, we will stabilize and document the API.
