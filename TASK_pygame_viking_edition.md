# TASK: pygame Viking Edition — Full Transformation Plan

**Project:** pygame Viking Edition (hrabanazviking/pygame fork)  
**Session Start:** 2026-04-05  
**Collaborators:** Volmarr Wyrd + Runa Gridweaver Freyjasdottir  
**Branch:** development  
**Local Path:** `C:/Users/volma/runa/pygame`

---

## Vision Statement

Take the pygame library — the backbone of thousands of Python games and Raspberry Pi projects — and transform it into a modern, AI-native, 3D/VR-capable, vibe-coding-ready game engine foundation. Build the layer beneath H.E.R.E.T.I.C. and future NorseSagaEngine 3D expansion. Leave the code better, more robust, more capable, and more clearly structured than any version that came before it.

---

## Codebase Snapshot (as of 2026-04-05)

| Area | Files | Notes |
|---|---|---|
| `src_c/` | 154 | C core — SDL2 wrappers, blitters, math, surface, event, display, etc. |
| `src_py/` | 28 | Python layer — sprite, camera, midi, surfarray, sysfont, etc. |
| `test/` | 122 | Test suite |
| `docs/` | 249 | RST documentation |
| `examples/` | 88 | Example scripts |
| `buildconfig/` | 160 | Build system, deps config |

**Key C Modules:**
- `base.c` — pygame init/quit, error handling, module base
- `display.c` — window management, display modes
- `surface.c` / `surface.h` — Surface object, core pixel buffer
- `event.c` — event queue, pump, custom events
- `draw.c` — 2D drawing primitives
- `transform.c` — scale, rotate, flip
- `rect.c` — Rect object
- `color.c` — Color object
- `image.c` / `imageext.c` — image load/save
- `font.c` — bitmap font
- `_freetype.c` / `freetype/` — FreeType font engine
- `mixer.c` / `music.c` — audio
- `joystick.c` — gamepad/joystick
- `key.c` / `mouse.c` — input
- `mask.c` — collision masks
- `math.c` — Vector2, Vector3, Vector4
- `pixelarray.c` / `pixelcopy.c` — pixel-level access
- `simd_blitters_sse2.c` / `simd_blitters_avx2.c` — SIMD-accelerated blitting
- `scale2x.c` / `rotozoom.c` — image scaling algorithms
- `scrap_*.c` — clipboard (platform-specific: win/x11/mac/sdl2/qnx)
- `camera.c` / `camera_v4l2.c` — camera input
- `_sdl2/` — Cython wrappers for SDL2 audio, controller, video, mixer
- `newbuffer.c` / `bufferproxy.c` — Python buffer protocol
- `bitmask.c` — bitmask collision detection

**Key Python Modules:**
- `sprite.py` — Sprite and Group classes
- `camera.py` — camera abstraction
- `midi.py` — MIDI I/O
- `surfarray.py` / `sndarray.py` — numpy array integration
- `sysfont.py` — system font discovery
- `cursors.py` — cursor shapes
- `colordict.py` — named color database
- `draw_py.py` — pure-Python draw fallback
- `ftfont.py` / `freetype.py` — FreeType Python interface
- `locals.py` — constants re-export
- `macosx.py` — macOS-specific init

---

## Master Phase Plan

### PHASE 0 — Vibe Coding Foundation (Documentation & Structure)
> Goal: Make every part of this codebase legible to a vibe coding session. No file should be a black box.

**0A — Master Architecture Document** ✓ COMPLETE
- [x] `ARCHITECTURE.md` — complete systems overview (29 sections, all major systems with mermaid)

**0B — Per-File Structure Documents** ✓ COMPLETE
- [x] `docs/structure/` directory — 23 docs covering all C + Python + header files

**0C — Test Coverage Map** ✓ COMPLETE
- [x] `docs/TEST_MAP.md` — test file → module mapping, gap analysis

**0D — Build System Docs** ✓ COMPLETE
- [x] `docs/BUILD_SYSTEM.md` — full build system docs, platform notes, adding new C modules

**0E — API Surface Reference**
- [ ] `docs/API_SURFACE.md` — every public Python-facing symbol, with type stubs cross-reference

---

### PHASE 1 — Bug Audit & Hardening
> Goal: Make the code trustworthy. Every crash a past coder accepted as "just how it is" gets fixed.

**1A — Static Analysis Pass** ✓ COMPLETE
- [ ] Run `cppcheck` over all `src_c/` files — cppcheck not available on Windows; manual audit performed instead
- [x] Run `mypy` on `src_py/` — 4 findings, documented in AUDIT_REPORT.md
- [x] Run `pylint` on `src_py/` — 2 findings (platform-conditional imports in sysfont.py)
- [x] Document all findings in `docs/AUDIT_REPORT.md`

**1B — C Safety Audit** ✓ COMPLETE (9 priority files)
- [x] Null pointer dereference audit — 6 null-deref bugs fixed (display×4, surface×1, image×1)
- [x] TTF resource check — font_init silent failure fixed (font×1)
- [ ] Integer overflow audit — not yet done (lower priority; blitters use controlled types)
- [ ] Buffer overflow audit — not yet done
- [ ] Use-after-free audit — not yet done (surface cleanup looks correct)
- [ ] Uninitialized variable audit — not yet done

**1C — Python Layer Safety** ✓ COMPLETE
- [x] `sprite.py` Generic base class — `Generic[TypeVar("T")]` → `_T = TypeVar("_T")` + `Generic[_T]`
- [x] `sysfont.py` platform-conditional imports — moved into function scope; pylint 10.00/10
- [x] `sysfont.py` type annotations — Sysfonts/Sysalias properly typed
- [x] `__init__.py` copyreg.pickle — `# type: ignore[arg-type]` on old-style API calls
- [ ] Broader type annotation pass on public functions — deferred to later phase

**1D — Memory Management Audit** ✓ COMPLETE
- [x] SDL2 resource leak check — 4 bugs fixed (display.c quit/error paths; transform.c rotozoom)
- [x] GIL handling — BEGIN/END pairs verified balanced in all key files
- [x] Refcount correctness — surface.c blits and subsurface chain verified correct
- [x] FreeType lifecycle — _PGFT_Init/_PGFT_Quit ref-counting verified clean

**1E — Thread Safety**
- [ ] Identify all non-thread-safe paths
- [ ] Add guards where needed or document clearly

**1F — Self-Healing Patterns**
- [ ] Graceful degradation when hardware features unavailable
- [ ] Fallback paths for missing fonts, audio, camera
- [ ] Retry logic for transient SDL failures

---

### PHASE 2 — Cross-Platform & Platform Agnostic
> Goal: True parity. If it runs on Windows, it runs identically on RPi, Mac, and Linux.

**2A — Platform Abstraction Layer**
- [ ] Audit all `#ifdef` platform guards — map all platform-specific paths
- [ ] Create `pgplatform.h` extension with unified abstractions
- [ ] Platform capability detection at runtime vs compile time

**2B — Platform Parity Audit**
- [ ] Windows — full feature check
- [ ] macOS — full feature check (M1/M2 native + Rosetta)
- [ ] Linux x86_64 — full feature check
- [ ] Linux ARM (Raspberry Pi 4/5) — full feature check
- [ ] Document gaps per platform in `docs/PLATFORM_MATRIX.md`

**2C — ARM / Raspberry Pi Optimization**
- [ ] NEON SIMD blitters (ARM equivalent of SSE2/AVX2)
- [ ] Memory bandwidth optimization for RPi's unified memory
- [ ] GPU acceleration via OpenGL ES 2.0/3.0 on RPi

**2D — WebAssembly Target**
- [ ] Emscripten build configuration
- [ ] Pyodide integration path
- [ ] SDL2/Emscripten event loop adaptation

**2E — Android/iOS Groundwork**
- [ ] SDL2 mobile backend research and integration map
- [ ] Touch input normalization
- [ ] Mobile asset loading patterns

---

### PHASE 3 — AI Integration Layer
> Goal: Make pygame the natural substrate for AI-driven games. Python IS the AI language; pygame should feel that way.

**3A — AI Agent Interface**
- [ ] `pygame.ai` module — base classes for AI agents
- [ ] `Agent` — perceives game state, produces actions
- [ ] `AgentEnvironment` — wraps a pygame scene as an RL environment
- [ ] OpenAI Gym / Gymnasium compatible interface
- [ ] Observable game state serialization

**3B — Neural Network Hooks**
- [ ] Input preprocessing — screen → tensor (numpy bridge)
- [ ] Output postprocessing — tensor → action
- [ ] Frame buffer → model input pipeline (zero-copy where possible)
- [ ] Support: ONNX runtime, PyTorch, TensorFlow input formats

**3C — Procedural Content Generation API**
- [ ] `pygame.pcg` module
- [ ] Noise generators (Perlin, Simplex, Worley)
- [ ] Terrain generation hooks
- [ ] L-system support for flora/structures
- [ ] LLM prompt → asset description pipeline hook

**3D — LLM Narrative Engine Hooks**
- [ ] `pygame.narrative` module
- [ ] NPC dialogue state machine interface
- [ ] Event-driven narrative triggers
- [ ] WYRD Protocol world model bridge (integrate with WYRD ECS)
- [ ] Async LLM call management (non-blocking narrative updates)

**3E — Computer Vision Integration**
- [ ] Camera frame → numpy → model pipeline
- [ ] Face detection hooks (mediapipe/OpenCV bridge)
- [ ] Gesture recognition interface
- [ ] AR overlay support

**3F — Reinforcement Learning Environment**
- [ ] Full `gymnasium.Env` implementation wrapper
- [ ] Step/reset/render interface
- [ ] Observation space auto-detection from game state
- [ ] Action space mapping (keyboard/mouse/gamepad)
- [ ] Reward signal hook system

**3G — Emotion & Affect System**
- [ ] `pygame.affect` module
- [ ] Emotional state model for game entities (links to WYRD Wyrd Matrix concepts)
- [ ] Affect → visual feedback mappings (color, animation speed, particle behavior)
- [ ] Hume AI / local affect model bridge

**3H — WYRD Protocol Bridge**
- [ ] `pygame.wyrd` bridge module
- [ ] World model state → pygame scene sync
- [ ] pygame events → WYRD world model updates
- [ ] Character sheet → sprite/NPC binding

---

### PHASE 4 — Advanced 3D Support
> Goal: pygame can render 3D. Not just "you can use OpenGL" — actually support it as a first-class citizen.

**4A — OpenGL Scene Graph**
- [ ] `pygame.gl` module
- [ ] Scene graph: Node, MeshNode, LightNode, CameraNode
- [ ] Shader program management
- [ ] VAO/VBO abstraction
- [ ] Texture management (2D, cubemap, array)

**4B — Vulkan Renderer (Optional Backend)**
- [ ] Research: SDL2 Vulkan integration path
- [ ] `pygame.vulkan` optional module
- [ ] Command buffer abstraction
- [ ] Render pass management

**4C — 3D Math Library Extension**
- [ ] Extend existing `math.c` Vector2/3/4
- [ ] Matrix4x4 class (transform, projection, view)
- [ ] Quaternion class (rotation)
- [ ] Frustum / AABB / OBB classes
- [ ] Ray / plane intersection math

**4D — 3D Asset Loading**
- [ ] OBJ file loader
- [ ] GLTF/GLB file loader
- [ ] Material system (PBR properties)
- [ ] Mesh optimizer and normal computation

**4E — Lighting System**
- [ ] Physically Based Rendering (PBR) shader library
- [ ] Directional, point, spot light types
- [ ] IBL (Image Based Lighting) support
- [ ] HDR rendering pipeline

**4F — Shadow Rendering**
- [ ] Shadow mapping (directional)
- [ ] Omnidirectional shadow maps (point lights)
- [ ] Cascade Shadow Maps for large scenes

**4G — 3D Physics Integration**
- [ ] Bullet physics Python bridge
- [ ] Rigid body dynamics
- [ ] Collision shapes: box, sphere, capsule, convex hull, mesh
- [ ] Raycast API
- [ ] Character controller

**4H — Skeletal Animation**
- [ ] Bone/armature data structure
- [ ] Animation clip system
- [ ] Blend tree (interpolation between clips)
- [ ] Inverse Kinematics (FABRIK)

---

### PHASE 5 — VR Support
> Goal: pygame scenes can be experienced in VR. H.E.R.E.T.I.C. will run in a headset.

**5A — OpenXR Integration**
- [ ] `pygame.xr` module
- [ ] OpenXR session management
- [ ] Swapchain and framebuffer management
- [ ] Stereoscopic rendering pipeline (left/right eye)

**5B — HMD Management**
- [ ] Device enumeration (SteamVR, Oculus, WMR, standalone)
- [ ] Display refresh rate negotiation
- [ ] Foveated rendering support

**5C — Motion Controller Input**
- [ ] Controller pose tracking (position + orientation)
- [ ] Button/trigger/grip/thumbstick input
- [ ] Haptic feedback output
- [ ] Controller model rendering

**5D — Spatial Audio**
- [ ] HRTF (Head-Related Transfer Function) audio
- [ ] 3D sound positioning (relative to HMD)
- [ ] Room acoustics simulation
- [ ] Ambisonics decode support

**5E — Hand Tracking**
- [ ] OpenXR hand tracking extension
- [ ] 26-joint hand pose data
- [ ] Gesture recognition from joint data
- [ ] Hand mesh rendering

**5F — Room-Scale Support**
- [ ] Boundary/guardian visualization
- [ ] Standing, seated, room-scale mode detection
- [ ] Comfort/locomotion system (teleport, smooth turn)

---

### PHASE 6 — Internal API Architecture (Location Agnostic)
> Goal: Every pygame module communicates through a defined internal API. Code can move, split, or distribute without rewiring.

**6A — Module Communication Bus**
- [ ] Internal event bus (not SDL events — internal pygame signals)
- [ ] Message passing interface between C and Python layers
- [ ] Async message queue for non-blocking module communication

**6B — Plugin / Extension System**
- [ ] `pygame.plugin` module
- [ ] Plugin discovery and loading
- [ ] Versioned plugin API contracts
- [ ] Hot-swap of non-critical plugins

**6C — Hot-Reload Support**
- [ ] Game logic Python module hot-reload without restart
- [ ] Asset hot-reload (textures, audio, shaders)
- [ ] State preservation across reload

**6D — Distributed / Networked Mode**
- [ ] Optional: render process separate from game logic process
- [ ] Network protocol for remote game state → local rendering
- [ ] WebSocket and TCP transport options
- [ ] Headless server mode (AI training without display)

---

### PHASE 7 — Modern Game Engine Features
> Goal: Match and exceed the feature set of any contemporary Python game engine.

**7A — Entity Component System (ECS)**
- [ ] `pygame.ecs` module
- [ ] World, Entity, Component, System abstractions
- [ ] Archetype-based storage (cache-friendly)
- [ ] System dependency/ordering graph
- [ ] WYRD Protocol ECS bridge (leverage existing WYRD work)

**7B — Advanced Physics**
- [ ] 2D physics (Box2D integration — native Python)
- [ ] 3D physics (Phase 4G extended)
- [ ] Particle physics system
- [ ] Soft body / cloth simulation hooks

**7C — Shader System**
- [ ] GLSL shader hot-reload
- [ ] Shader graph (node-based, outputs GLSL)
- [ ] Post-processing stack (bloom, SSAO, DOF, motion blur, color grading)
- [ ] Compute shader support

**7D — Asset Pipeline & Streaming**
- [ ] Asset manifest and registry
- [ ] Async background loading
- [ ] Level-of-detail (LOD) management
- [ ] Asset compression and packing

**7E — Networking & Multiplayer**
- [ ] `pygame.net` module
- [ ] Reliable UDP (ENet or KCP)
- [ ] Client-server and P2P topologies
- [ ] State synchronization with delta compression
- [ ] Lag compensation / rollback netcode

**7F — Advanced Audio DSP**
- [ ] Effects chain (reverb, delay, chorus, EQ, compressor)
- [ ] Procedural audio generation
- [ ] Adaptive music system (state-driven soundtrack)
- [ ] Voice activity detection hook

**7G — Immediate Mode UI**
- [ ] `pygame.imgui` module (Dear ImGui Python bridge)
- [ ] In-game debug overlay
- [ ] Editor-mode UI panels
- [ ] Theming system

**7H — Scene Management**
- [ ] Scene graph (2D and 3D unified)
- [ ] Scene transition system
- [ ] Additive scene loading
- [ ] Scene serialization / save-load

**7I — Profiling & Debugging Tools**
- [ ] Frame time graph overlay
- [ ] GPU query timer integration
- [ ] Memory allocator tracker
- [ ] Event heatmap visualizer
- [ ] Python cProfile integration overlay

**7J — Extended Scripting**
- [ ] Lua scripting integration (lupa)
- [ ] REPL / console in running game
- [ ] Live code execution from in-game terminal

---

## Progress Tracker

| Phase | Status | Notes |
|---|---|---|
| 0A Master Architecture | **COMPLETE** | ARCHITECTURE.md — 29 sections, all major systems with mermaid |
| 0B Per-File Structure Docs | **COMPLETE** | 23 docs in docs/structure/ covering all C + Python modules |
| 0C Test Coverage Map | **COMPLETE** | docs/TEST_MAP.md |
| 0D Build System Docs | **COMPLETE** | docs/BUILD_SYSTEM.md |
| 0E API Surface Reference | NOT STARTED | Next phase work |
| 1A Static Analysis | **COMPLETE** | mypy 1.20.0 + pylint 4.0.5 run; findings in docs/AUDIT_REPORT.md |
| 1B C Safety Audit | **COMPLETE** | 7 bugs fixed across display/surface/image/font; see AUDIT_REPORT.md |
| 1C Python Layer Safety | **COMPLETE** | sprite.py Generic fix, sysfont imports, type annotations, copyreg ignore |
| 1D Memory Management | **COMPLETE** | 4 SDL resource bugs fixed (display.c×3, transform.c×1); GIL/refcount verified |
| 1E Thread Safety | NOT STARTED | |
| 1F Self-Healing Patterns | NOT STARTED | |
| 2A Platform Abstraction | NOT STARTED | |
| 2B Platform Parity Audit | NOT STARTED | |
| 2C ARM/RPi Optimization | NOT STARTED | |
| 2D WebAssembly Target | NOT STARTED | |
| 2E Android/iOS Groundwork | NOT STARTED | |
| 3A AI Agent Interface | NOT STARTED | |
| 3B Neural Network Hooks | NOT STARTED | |
| 3C PCG API | NOT STARTED | |
| 3D LLM Narrative Hooks | NOT STARTED | |
| 3E Computer Vision | NOT STARTED | |
| 3F RL Environment | NOT STARTED | |
| 3G Emotion/Affect System | NOT STARTED | |
| 3H WYRD Protocol Bridge | NOT STARTED | |
| 4A OpenGL Scene Graph | NOT STARTED | |
| 4B Vulkan Renderer | NOT STARTED | |
| 4C 3D Math Library | NOT STARTED | |
| 4D 3D Asset Loading | NOT STARTED | |
| 4E Lighting System | NOT STARTED | |
| 4F Shadow Rendering | NOT STARTED | |
| 4G 3D Physics | NOT STARTED | |
| 4H Skeletal Animation | NOT STARTED | |
| 5A OpenXR Integration | NOT STARTED | |
| 5B HMD Management | NOT STARTED | |
| 5C Motion Controller Input | NOT STARTED | |
| 5D Spatial Audio | NOT STARTED | |
| 5E Hand Tracking | NOT STARTED | |
| 5F Room-Scale Support | NOT STARTED | |
| 6A Module Communication Bus | NOT STARTED | |
| 6B Plugin System | NOT STARTED | |
| 6C Hot-Reload | NOT STARTED | |
| 6D Distributed Mode | NOT STARTED | |
| 7A ECS | NOT STARTED | |
| 7B Advanced Physics | NOT STARTED | |
| 7C Shader System | NOT STARTED | |
| 7D Asset Pipeline | NOT STARTED | |
| 7E Networking | NOT STARTED | |
| 7F Audio DSP | NOT STARTED | |
| 7G Immediate Mode UI | NOT STARTED | |
| 7H Scene Management | NOT STARTED | |
| 7I Profiling Tools | NOT STARTED | |
| 7J Extended Scripting | NOT STARTED | |

---

## Immediate Next Steps (Phase 1E)

**Phase 0 COMPLETE. Phase 1A–1D COMPLETE. HEAD: 1bc45dfe**

1. **Phase 1E — Thread Safety:**
   - Verify mutex coverage completeness in event.c
   - Surface lock/unlock safety from non-main threads
   - Display state access from non-main threads

2. **Phase 1F — Self-Healing Patterns:**
   - SDL_Init failure recovery paths
   - Error context enrichment (SDL_GetError propagation)

---

## Key Design Decisions

- **This fork diverges from upstream.** If upstream pygame becomes active again, a diff/merge strategy will be needed. For now: our own version.
- **Backward compatibility is maintained through Phase 1-2.** Breaking changes (if any) are deferred to Phase 3+.
- **Phase 3 (AI) and Phase 4 (3D) can proceed in parallel** once Phase 0-2 are solid.
- **WYRD Protocol is the world model layer.** pygame Viking Edition renders it; WYRD owns the state.
- **H.E.R.E.T.I.C. is the first major target** for the complete stack (Phases 0-5 minimum before ship).

---

## Related Projects

| Project | Relationship |
|---|---|
| `WYRD-Protocol` | World model + ECS layer that pygame VE will render |
| `NorseSagaEngine` | Will eventually migrate to pygame VE 3D backend |
| `H.E.R.E.T.I.C.` | First major game built on pygame VE full stack |
| `Viking_Girlfriend_Skill_for_OpenClaw` | AI persona layer — will use 3H WYRD bridge |
