if plat.startswith('emscripten')  # single build

base_files = ['base.c', 'bitmask.c', 'rotozoom.c', 'SDL_gfx/SDL_gfxPrimitives.c']
cython_files = [
    'cython/pygame/_sdl2/audio.pyx',
    'cython/pygame/_sdl2/mixer.pyx',
    'cython/pygame/_sdl2/sdl2.pyx',
    'cython/pygame/_sdl2/video.pyx',
]

# make one big shared build on emscripten
pygame = py.extension_module(
    'base',
    base_files + cython_files,
    c_args: ['-DBUILD_STATIC=1'],
    dependencies: pg_base_deps + [sdl_image_dep, sdl_mixer_dep, sdl_ttf_dep, freetype_dep],
    install: true,
    subdir: pg,
)

else  # regular build

# first the "required" modules

base = py.extension_module(
    'base',
    'base.c',
    c_args: warnings_error,
    dependencies: pg_base_deps,
    install: true,
    subdir: pg,
)

color = py.extension_module(
    'color',
    'color.c',
    c_args: warnings_error,
    dependencies: pg_base_deps,
    install: true,
    subdir: pg,
)

constants = py.extension_module(
    'constants',
    'constants.c',
    c_args: warnings_error,
    dependencies: pg_base_deps,
    install: true,
    subdir: pg,
)

display = py.extension_module(
    'display',
    'display.c',
    c_args: warnings_error,
    dependencies: pg_base_deps,
    install: true,
    subdir: pg,
)

event = py.extension_module(
    'event',
    'event.c',
    c_args: warnings_error,
    dependencies: pg_base_deps,
    install: true,
    subdir: pg,
)

key = py.extension_module(
    'key',
    'key.c',
    c_args: warnings_error,
    dependencies: pg_base_deps,
    install: true,
    subdir: pg,
)

mouse = py.extension_module(
    'mouse',
    'mouse.c',
    c_args: warnings_error,
    dependencies: pg_base_deps,
    install: true,
    subdir: pg,
)

rect = py.extension_module(
    'rect',
    ['rect.c', 'pgcompat_rect.c'],
    c_args: warnings_error,
    dependencies: pg_base_deps,
    install: true,
    subdir: pg,
)

rwobject = py.extension_module(
    'rwobject',
    'rwobject.c',
    c_args: warnings_error,
    dependencies: pg_base_deps,
    install: true,
    subdir: pg,
)

simd_blitters_avx2 = static_library(
    'simd_blitters_avx2',
    'simd_blitters_avx2.c',
    dependencies: pg_base_deps,
    c_args: simd_avx2_flags + warnings_error,
)

simd_blitters_sse2 = static_library(
    'simd_blitters_sse2',
    'simd_blitters_sse2.c',
    dependencies: pg_base_deps,
    c_args: simd_sse2_neon_flags + warnings_error,
)

simd_surface_fill_avx2 = static_library(
    'simd_surface_fill_avx2',
    'simd_surface_fill_avx2.c',
    dependencies: pg_base_deps,
    c_args: simd_avx2_flags + warnings_error,
)

simd_surface_fill_sse2 = static_library(
    'simd_surface_fill_sse2',
    'simd_surface_fill_sse2.c',
    dependencies: pg_base_deps,
    c_args: simd_sse2_neon_flags + warnings_error,
)

surface = py.extension_module(
    'surface',
    [
        'surface.c',
        'alphablit.c',
        'surface_fill.c',
    ],
    c_args: warnings_error,
    link_with: [
        simd_blitters_avx2,
        simd_blitters_sse2,
        simd_surface_fill_avx2,
        simd_surface_fill_sse2,
    ],
    dependencies: pg_base_deps,
    install: true,
    subdir: pg,
)

surflock = py.extension_module(
    'surflock',
    'surflock.c',
    c_args: warnings_error,
    dependencies: pg_base_deps,
    install: true,
    subdir: pg,
)

time = py.extension_module(
    'time',
    'time.c',
    c_args: warnings_error,
    dependencies: pg_base_deps,
    install: true,
    subdir: pg,
)

joystick = py.extension_module(
    'joystick',
    'joystick.c',
    c_args: warnings_error,
    dependencies: pg_base_deps,
    install: true,
    subdir: pg,
)

draw = py.extension_module(
    'draw',
    'draw.c',
    c_args: warnings_error,
    dependencies: pg_base_deps,
    install: true,
    subdir: pg,
)

image = py.extension_module(
    'image',
    'image.c',
    c_args: warnings_error,
    dependencies: pg_base_deps,
    install: true,
    subdir: pg,
)

simd_transform_avx2 = static_library(
    'simd_transform_avx2',
    'simd_transform_avx2.c',
    dependencies: pg_base_deps,
    c_args: simd_avx2_flags + warnings_error,
)

simd_transform_sse2 = static_library(
    'simd_transform_sse2',
    'simd_transform_sse2.c',
    dependencies: pg_base_deps,
    c_args: simd_sse2_neon_flags + warnings_error,
)

transform_sources = ['transform.c', 'rotozoom.c', 'scale2x.c']
transform_objs = []

if (
    cc.get_argument_syntax() == 'msvc'
    and host_machine.cpu_family().startswith('x86')
)
    if host_machine.cpu_family() == 'x86'
        transform_objs += '../buildconfig/obj/win32/scale_mmx.obj'
    else
        transform_objs += '../buildconfig/obj/win64/scale_mmx.obj'
    endif
else
    transform_sources += 'scale_mmx.c'
endif

transform = py.extension_module(
    'transform',
    transform_sources,
    c_args: warnings_error,
    link_with: [simd_transform_avx2, simd_transform_sse2],
    objects: transform_objs,
    dependencies: pg_base_deps,
    install: true,
    subdir: pg,
)

mask = py.extension_module(
    'mask',
    ['mask.c', 'bitmask.c'],
    c_args: warnings_error + warnings_temp_mask,
    dependencies: pg_base_deps,
    install: true,
    subdir: pg,
)

bufferproxy = py.extension_module(
    'bufferproxy',
    'bufferproxy.c',
    c_args: warnings_error,
    dependencies: pg_base_deps,
    install: true,
    subdir: pg,
)

pixelarray = py.extension_module(
    'pixelarray',
    'pixelarray.c',
    c_args: warnings_error,
    dependencies: pg_base_deps,
    install: true,
    subdir: pg,
)

math = py.extension_module(
    'math',
    'math.c',
    c_args: warnings_error,
    dependencies: pg_base_deps,
    install: true,
    subdir: pg,
)

pixelcopy = py.extension_module(
    'pixelcopy',
    'pixelcopy.c',
    c_args: warnings_error,
    dependencies: pg_base_deps,
    install: true,
    subdir: pg,
)

newbuffer = py.extension_module(
    'newbuffer',
    'newbuffer.c',
    c_args: warnings_error,
    dependencies: pg_base_deps,
    install: true,
    subdir: pg,
)

# new/experimental/uncommon stuff, but built by default
system = py.extension_module(
    'system',
    'system.c',
    c_args: warnings_error,
    dependencies: pg_base_deps,
    install: true,
    subdir: pg,
)

geometry = py.extension_module(
    'geometry',
    'geometry.c',
    c_args: warnings_error,
    dependencies: pg_base_deps,
    install: true,
    subdir: pg,
)

window = py.extension_module(
    'window',
    'window.c',
    c_args: warnings_error,
    dependencies: pg_base_deps,
    install: true,
    subdir: pg,
)

# TODO: support SDL3
if sdl_api != 3
_render = py.extension_module(
    '_render',
    'render.c',
    c_args: warnings_error,
    dependencies: pg_base_deps,
    install: true,
    subdir: pg,
)
endif

# TODO: support SDL3
if sdl_api != 3
gfxdraw = py.extension_module(
    'gfxdraw',
    ['gfxdraw.c', 'SDL_gfx/SDL_gfxPrimitives.c'],
    # c_args: warnings_error,
    dependencies: pg_base_deps,
    install: true,
    subdir: pg,
)
endif

# pygame._sdl2
# TODO: support SDL3
if sdl_api != 3
subdir('_sdl2')
endif

# pygame._camera
pg_camera_sources = ['_camera.c']
pg_camera_link = []
if plat == 'win'
    pg_camera_sources += 'camera_windows.c'
    # TODO: should this link logic be improved/made meson-ey?
    pg_camera_link = []
    foreach link_arg : [
        '-lMfplat',
        '-lMf',
        '-lMfuuid',
        '-lMfreadwrite',
        '-lOle32',
    ]
        if cc.has_link_argument(link_arg)
            pg_camera_link += link_arg
        endif
    endforeach
elif plat == 'linux'
    pg_camera_sources += 'camera_v4l2.c'
endif

_camera = py.extension_module(
    '_camera',
    pg_camera_sources,
    c_args: warnings_error,
    dependencies: pg_base_deps,
    link_args: pg_camera_link,
    install: true,
    subdir: pg,
)

# pygame.scrap
pg_scrap_link = [] # TODO: should this link logic be improved/made meson-ey?
if plat == 'win'
    pg_scrap_link += ['-luser32', '-lgdi32']
endif

scrap = py.extension_module(
    'scrap',
    'scrap.c',
    c_args: warnings_error,
    dependencies: pg_base_deps,
    link_args: pg_scrap_link,
    install: true,
    subdir: pg,
)

# optional modules

if sdl_image_dep.found()
    imageext = py.extension_module(
        'imageext',
        'imageext.c',
        c_args: warnings_error,
        dependencies: pg_base_deps + sdl_image_dep,
        install: true,
        subdir: pg,
    )
endif

if sdl_ttf_dep.found()
    font = py.extension_module(
        'font',
        'font.c',
        c_args: warnings_error,
        dependencies: pg_base_deps + sdl_ttf_dep,
        install: true,
        subdir: pg,
    )
endif

# TODO: support SDL3
if sdl_api != 3

if sdl_mixer_dep.found()
    mixer = py.extension_module(
        'mixer',
        'mixer.c',
        c_args: warnings_error,
        dependencies: pg_base_deps + sdl_mixer_dep,
        install: true,
        subdir: pg,
    )

    mixer_music = py.extension_module(
        'mixer_music',
        'music.c',
        c_args: warnings_error,
        dependencies: pg_base_deps + sdl_mixer_dep,
        install: true,
        subdir: pg,
    )
endif

endif

if freetype_dep.found()
    _freetype = py.extension_module(
        '_freetype',
        [
            'freetype/ft_cache.c',
            'freetype/ft_wrap.c',
            'freetype/ft_render.c',
            'freetype/ft_render_cb.c',
            'freetype/ft_layout.c',
            'freetype/ft_unicode.c',
            '_freetype.c',
        ],
        c_args: warnings_error + warnings_temp_freetype,
        dependencies: pg_base_deps + freetype_dep,
        install: true,
        subdir: pg,
    )
endif

if portmidi_dep.found()
    pypm = py.extension_module(
        'pypm',
        fs.is_file('pypm.c') ? 'pypm.c' : 'cython/pygame/pypm.pyx',
        # c_args: warnings_error,
        dependencies: pg_base_deps + portmidi_deps,
        install: true,
        subdir: pg,
    )
endif

endif # regular build
