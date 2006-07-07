#!/usr/bin/env python

'''Constants and enums for all SDL submodules.
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id: $'

import sys

# enum SDLKey {

#  The keyboard syms have been cleverly chosen to map to ASCII 
SDLK_UNKNOWN            = 0
SDLK_FIRST              = 0
SDLK_BACKSPACE          = 8
SDLK_TAB                = 9
SDLK_CLEAR              = 12
SDLK_RETURN             = 13
SDLK_PAUSE              = 19
SDLK_ESCAPE             = 27
SDLK_SPACE              = 32
SDLK_EXCLAIM            = 33
SDLK_QUOTEDBL           = 34
SDLK_HASH               = 35
SDLK_DOLLAR             = 36
SDLK_AMPERSAND          = 38
SDLK_QUOTE              = 39
SDLK_LEFTPAREN          = 40
SDLK_RIGHTPAREN         = 41
SDLK_ASTERISK           = 42
SDLK_PLUS               = 43
SDLK_COMMA              = 44
SDLK_MINUS              = 45
SDLK_PERIOD             = 46
SDLK_SLASH              = 47
SDLK_0                  = 48
SDLK_1                  = 49
SDLK_2                  = 50
SDLK_3                  = 51
SDLK_4                  = 52
SDLK_5                  = 53
SDLK_6                  = 54
SDLK_7                  = 55
SDLK_8                  = 56
SDLK_9                  = 57
SDLK_COLON              = 58
SDLK_SEMICOLON          = 59
SDLK_LESS               = 60
SDLK_EQUALS             = 61
SDLK_GREATER            = 62
SDLK_QUESTION           = 63
SDLK_AT                 = 64

#  Skip uppercase letters

SDLK_LEFTBRACKET        = 91
SDLK_BACKSLASH          = 92
SDLK_RIGHTBRACKET       = 93
SDLK_CARET              = 94
SDLK_UNDERSCORE         = 95
SDLK_BACKQUOTE          = 96
SDLK_a                  = 97
SDLK_b                  = 98
SDLK_c                  = 99
SDLK_d                  = 100
SDLK_e                  = 101
SDLK_f                  = 102
SDLK_g                  = 103
SDLK_h                  = 104
SDLK_i                  = 105
SDLK_j                  = 106
SDLK_k                  = 107
SDLK_l                  = 108
SDLK_m                  = 109
SDLK_n                  = 110
SDLK_o                  = 111
SDLK_p                  = 112
SDLK_q                  = 113
SDLK_r                  = 114
SDLK_s                  = 115
SDLK_t                  = 116
SDLK_u                  = 117
SDLK_v                  = 118
SDLK_w                  = 119
SDLK_x                  = 120
SDLK_y                  = 121
SDLK_z                  = 122
SDLK_DELETE             = 127
#  End of ASCII mapped keysyms 

#  International keyboard syms 
SDLK_WORLD_0            = 160          #  0xA0 
SDLK_WORLD_1            = 161
SDLK_WORLD_2            = 162
SDLK_WORLD_3            = 163
SDLK_WORLD_4            = 164
SDLK_WORLD_5            = 165
SDLK_WORLD_6            = 166
SDLK_WORLD_7            = 167
SDLK_WORLD_8            = 168
SDLK_WORLD_9            = 169
SDLK_WORLD_10           = 170
SDLK_WORLD_11           = 171
SDLK_WORLD_12           = 172
SDLK_WORLD_13           = 173
SDLK_WORLD_14           = 174
SDLK_WORLD_15           = 175
SDLK_WORLD_16           = 176
SDLK_WORLD_17           = 177
SDLK_WORLD_18           = 178
SDLK_WORLD_19           = 179
SDLK_WORLD_20           = 180
SDLK_WORLD_21           = 181
SDLK_WORLD_22           = 182
SDLK_WORLD_23           = 183
SDLK_WORLD_24           = 184
SDLK_WORLD_25           = 185
SDLK_WORLD_26           = 186
SDLK_WORLD_27           = 187
SDLK_WORLD_28           = 188
SDLK_WORLD_29           = 189
SDLK_WORLD_30           = 190
SDLK_WORLD_31           = 191
SDLK_WORLD_32           = 192
SDLK_WORLD_33           = 193
SDLK_WORLD_34           = 194
SDLK_WORLD_35           = 195
SDLK_WORLD_36           = 196
SDLK_WORLD_37           = 197
SDLK_WORLD_38           = 198
SDLK_WORLD_39           = 199
SDLK_WORLD_40           = 200
SDLK_WORLD_41           = 201
SDLK_WORLD_42           = 202
SDLK_WORLD_43           = 203
SDLK_WORLD_44           = 204
SDLK_WORLD_45           = 205
SDLK_WORLD_46           = 206
SDLK_WORLD_47           = 207
SDLK_WORLD_48           = 208
SDLK_WORLD_49           = 209
SDLK_WORLD_50           = 210
SDLK_WORLD_51           = 211
SDLK_WORLD_52           = 212
SDLK_WORLD_53           = 213
SDLK_WORLD_54           = 214
SDLK_WORLD_55           = 215
SDLK_WORLD_56           = 216
SDLK_WORLD_57           = 217
SDLK_WORLD_58           = 218
SDLK_WORLD_59           = 219
SDLK_WORLD_60           = 220
SDLK_WORLD_61           = 221
SDLK_WORLD_62           = 222
SDLK_WORLD_63           = 223
SDLK_WORLD_64           = 224
SDLK_WORLD_65           = 225
SDLK_WORLD_66           = 226
SDLK_WORLD_67           = 227
SDLK_WORLD_68           = 228
SDLK_WORLD_69           = 229
SDLK_WORLD_70           = 230
SDLK_WORLD_71           = 231
SDLK_WORLD_72           = 232
SDLK_WORLD_73           = 233
SDLK_WORLD_74           = 234
SDLK_WORLD_75           = 235
SDLK_WORLD_76           = 236
SDLK_WORLD_77           = 237
SDLK_WORLD_78           = 238
SDLK_WORLD_79           = 239
SDLK_WORLD_80           = 240
SDLK_WORLD_81           = 241
SDLK_WORLD_82           = 242
SDLK_WORLD_83           = 243
SDLK_WORLD_84           = 244
SDLK_WORLD_85           = 245
SDLK_WORLD_86           = 246
SDLK_WORLD_87           = 247
SDLK_WORLD_88           = 248
SDLK_WORLD_89           = 249
SDLK_WORLD_90           = 250
SDLK_WORLD_91           = 251
SDLK_WORLD_92           = 252
SDLK_WORLD_93           = 253
SDLK_WORLD_94           = 254
SDLK_WORLD_95           = 255          #  0xFF 

#  Numeric keypad 
SDLK_KP0                = 256
SDLK_KP1                = 257
SDLK_KP2                = 258
SDLK_KP3                = 259
SDLK_KP4                = 260
SDLK_KP5                = 261
SDLK_KP6                = 262
SDLK_KP7                = 263
SDLK_KP8                = 264
SDLK_KP9                = 265
SDLK_KP_PERIOD          = 266
SDLK_KP_DIVIDE          = 267
SDLK_KP_MULTIPLY        = 268
SDLK_KP_MINUS           = 269
SDLK_KP_PLUS            = 270
SDLK_KP_ENTER           = 271
SDLK_KP_EQUALS          = 272

#  Arrows + Home/End pad 
SDLK_UP                 = 273
SDLK_DOWN               = 274
SDLK_RIGHT              = 275
SDLK_LEFT               = 276
SDLK_INSERT             = 277
SDLK_HOME               = 278
SDLK_END                = 279
SDLK_PAGEUP             = 280
SDLK_PAGEDOWN           = 281

#  Function keys 
SDLK_F1                 = 282
SDLK_F2                 = 283
SDLK_F3                 = 284
SDLK_F4                 = 285
SDLK_F5                 = 286
SDLK_F6                 = 287
SDLK_F7                 = 288
SDLK_F8                 = 289
SDLK_F9                 = 290
SDLK_F10                = 291
SDLK_F11                = 292
SDLK_F12                = 293
SDLK_F13                = 294
SDLK_F14                = 295
SDLK_F15                = 296

#  Key state modifier keys 
SDLK_NUMLOCK            = 300
SDLK_CAPSLOCK           = 301
SDLK_SCROLLOCK          = 302
SDLK_RSHIFT             = 303
SDLK_LSHIFT             = 304
SDLK_RCTRL              = 305
SDLK_LCTRL              = 306
SDLK_RALT               = 307
SDLK_LALT               = 308
SDLK_RMETA              = 309
SDLK_LMETA              = 310
SDLK_LSUPER             = 311          #  Left "Windows" key 
SDLK_RSUPER             = 312          #  Right "Windows" key 
SDLK_MODE               = 313          #  "Alt Gr" key 
SDLK_COMPOSE            = 314          #  Multi-key compose key 

#  Miscellaneous function keys 
SDLK_HELP               = 315
SDLK_PRINT              = 316
SDLK_SYSREQ             = 317
SDLK_BREAK              = 318
SDLK_MENU               = 319
SDLK_POWER              = 320          #  Power Macintosh power key 
SDLK_EURO               = 321          #  Some european keyboards 
SDLK_UNDO               = 322          #  Atari keyboard has Undo 

SDLK_LAST               = 323          #  Keep me updated please.

# end of enum SDLKey

# enum SDLMod

KMOD_NONE       = 0x0000
KMOD_LSHIFT     = 0x0001
KMOD_RSHIFT     = 0x0002
KMOD_LCTRL      = 0x0040
KMOD_RCTRL      = 0x0080
KMOD_LALT       = 0x0100
KMOD_RALT       = 0x0200
KMOD_LMETA      = 0x0400
KMOD_RMETA      = 0x0800
KMOD_NUM        = 0x1000
KMOD_CAPS       = 0x2000
KMOD_MODE       = 0x4000
KMOD_RESERVED   = 0x8000

# end of enum SDLMod

KMOD_CTRL   = KMOD_LCTRL    | KMOD_RCTRL
KMOD_SHIFT  = KMOD_LSHIFT   | KMOD_RSHIFT
KMOD_ALT    = KMOD_LALT     | KMOD_RALT
KMOD_META   = KMOD_LMETA    | KMOD_RMETA

#BEGIN GENERATED CONSTANTS; see support/make_constants.py

#Constants from SDL_mouse.h:
SDL_BUTTON_LEFT = 0x00000001
SDL_BUTTON_MIDDLE = 0x00000002
SDL_BUTTON_RIGHT = 0x00000003
SDL_BUTTON_WHEELUP = 0x00000004
SDL_BUTTON_WHEELDOWN = 0x00000005

#Constants from SDL_version.h:
SDL_MAJOR_VERSION = 0x00000001
SDL_MINOR_VERSION = 0x00000002
SDL_PATCHLEVEL = 0x0000000a

#Constants from SDL.h:
SDL_INIT_TIMER = 0x00000001
SDL_INIT_AUDIO = 0x00000010
SDL_INIT_VIDEO = 0x00000020
SDL_INIT_CDROM = 0x00000100
SDL_INIT_JOYSTICK = 0x00000200
SDL_INIT_NOPARACHUTE = 0x00100000
SDL_INIT_EVENTTHREAD = 0x01000000
SDL_INIT_EVERYTHING = 0x0000ffff

#Constants from SDL_mutex.h:
SDL_MUTEX_TIMEDOUT = 0x00000001

#Constants from SDL_video.h:
SDL_ALPHA_OPAQUE = 0x000000ff
SDL_ALPHA_TRANSPARENT = 0x00000000
SDL_SWSURFACE = 0x00000000
SDL_HWSURFACE = 0x00000001
SDL_ASYNCBLIT = 0x00000004
SDL_ANYFORMAT = 0x10000000
SDL_HWPALETTE = 0x20000000
SDL_DOUBLEBUF = 0x40000000
SDL_FULLSCREEN = 0x80000000
SDL_OPENGL = 0x00000002
SDL_OPENGLBLIT = 0x0000000a
SDL_RESIZABLE = 0x00000010
SDL_NOFRAME = 0x00000020
SDL_HWACCEL = 0x00000100
SDL_SRCCOLORKEY = 0x00001000
SDL_RLEACCELOK = 0x00002000
SDL_RLEACCEL = 0x00004000
SDL_SRCALPHA = 0x00010000
SDL_PREALLOC = 0x01000000
SDL_YV12_OVERLAY = 0x32315659
SDL_IYUV_OVERLAY = 0x56555949
SDL_YUY2_OVERLAY = 0x32595559
SDL_UYVY_OVERLAY = 0x59565955
SDL_YVYU_OVERLAY = 0x55595659
SDL_LOGPAL = 0x00000001
SDL_PHYSPAL = 0x00000002

#Constants from SDL_name.h:
NeedFunctionPrototypes = 0x00000001

#Constants from SDL_endian.h:
SDL_LIL_ENDIAN = 0x000004d2
SDL_BIG_ENDIAN = 0x000010e1

#Constants from SDL_audio.h:
AUDIO_U8 = 0x00000008
AUDIO_S8 = 0x00008008
AUDIO_U16LSB = 0x00000010
AUDIO_S16LSB = 0x00008010
AUDIO_U16MSB = 0x00001010
AUDIO_S16MSB = 0x00009010
SDL_MIX_MAXVOLUME = 0x00000080

#Constants from begin_code.h:
NULL = 0x00000000

#Constants from SDL_cdrom.h:
SDL_MAX_TRACKS = 0x00000063
SDL_AUDIO_TRACK = 0x00000000
SDL_DATA_TRACK = 0x00000004
CD_FPS = 0x0000004b

#Constants from SDL_events.h:
SDL_RELEASED = 0x00000000
SDL_PRESSED = 0x00000001
SDL_ALLEVENTS = 0xffffffff
SDL_IGNORE = 0x00000000
SDL_DISABLE = 0x00000000
SDL_ENABLE = 0x00000001

#Constants from SDL_active.h:
SDL_APPMOUSEFOCUS = 0x00000001
SDL_APPINPUTFOCUS = 0x00000002
SDL_APPACTIVE = 0x00000004

#Constants from SDL_joystick.h:
SDL_HAT_CENTERED = 0x00000000
SDL_HAT_UP = 0x00000001
SDL_HAT_RIGHT = 0x00000002
SDL_HAT_DOWN = 0x00000004
SDL_HAT_LEFT = 0x00000008

#Constants from SDL_keyboard.h:
SDL_ALL_HOTKEYS = 0xffffffff
SDL_DEFAULT_REPEAT_DELAY = 0x000001f4
SDL_DEFAULT_REPEAT_INTERVAL = 0x0000001e

#Constants from SDL_rwops.h:
RW_SEEK_SET = 0x00000000
RW_SEEK_CUR = 0x00000001
RW_SEEK_END = 0x00000002

#Constants from SDL_timer.h:
SDL_TIMESLICE = 0x0000000a
TIMER_RESOLUTION = 0x0000000a
#END GENERATED CONSTANTS

# From SDL_audio.h (inserted manually)

# enum SDL_audiostatus
(SDL_AUDIO_STOPPED,
    SDL_AUDIO_PLAYING,
    SDL_AUDIO_PAUSED) = range(3)

if sys.byteorder == 'little':
    AUDIO_U16SYS = AUDIO_U16LSB
    AUDIO_S16SYS = AUDIO_S16LSB
else:
    AUDIO_U16SYS = AUDIO_U16MSB
    AUDIO_S16SYS = AUDIO_S16MSB
AUDIO_U16 = AUDIO_U16LSB
AUDIO_S16 = AUDIO_S16LSB

# From SDL_cdrom.h (inserted manually)
# enum CDstatus
(CD_TRAYEMPTY,
    CD_STOPPED,
    CD_PLAYING,
    CD_PAUSED) = range(4)
CD_ERROR = -1

# From SDL_events.h (inserted manually)
# enum SDL_EventType
(SDL_NOEVENT,
    SDL_ACTIVEEVENT,
    SDL_KEYDOWN,
    SDL_KEYUP,
    SDL_MOUSEMOTION,
    SDL_MOUSEBUTTONDOWN,
    SDL_MOUSEBUTTONUP,
    SDL_JOYAXISMOTION,
    SDL_JOYBALLMOTION,
    SDL_JOYHATMOTION,
    SDL_JOYBUTTONDOWN,
    SDL_JOYBUTTONUP,
    SDL_QUIT,
    SDL_SYSWMEVENT,
    SDL_EVENT_RESERVEDA,
    SDL_EVENT_RESERVEDB,
    SDL_VIDEORESIZE,
    SDL_VIDEOEXPOSE,
    SDL_EVENT_RESERVED2,
    SDL_EVENT_RESERVED3,
    SDL_EVENT_RESERVED4,
    SDL_EVENT_RESERVED5,
    SDL_EVENT_RESERVED6,
    SDL_EVENT_RESERVED7) = range(24)
SDL_USEREVENT = 24
SDL_NUMEVENTS = 32

def SDL_EVENTMASK(x):
    '''Used for predefining event masks.'''
    return 1 << x

# enum SDL_EventMask
SDL_ACTIVEEVENTMASK     = SDL_EVENTMASK(SDL_ACTIVEEVENT)
SDL_KEYDOWNMASK         = SDL_EVENTMASK(SDL_KEYDOWN)
SDL_KEYUPMASK           = SDL_EVENTMASK(SDL_KEYUP)
SDL_KEYEVENTMASK        = SDL_KEYUPMASK | \
                          SDL_KEYDOWNMASK
SDL_MOUSEMOTIONMASK     = SDL_EVENTMASK(SDL_MOUSEMOTION)
SDL_MOUSEBUTTONDOWNMASK = SDL_EVENTMASK(SDL_MOUSEBUTTONDOWN)
SDL_MOUSEBUTTONUPMASK   = SDL_EVENTMASK(SDL_MOUSEBUTTONUP)
SDL_MOUSEEVENTMASK      = SDL_MOUSEMOTIONMASK | \
                          SDL_MOUSEBUTTONDOWNMASK | \
                          SDL_MOUSEBUTTONUPMASK
SDL_JOYAXISMOTIONMASK   = SDL_EVENTMASK(SDL_JOYAXISMOTION)
SDL_JOYBALLMOTIONMASK   = SDL_EVENTMASK(SDL_JOYBALLMOTION)
SDL_JOYHATMOTIONMASK    = SDL_EVENTMASK(SDL_JOYHATMOTION)
SDL_JOYBUTTONDOWNMASK   = SDL_EVENTMASK(SDL_JOYBUTTONDOWN)
SDL_JOYBUTTONUPMASK     = SDL_EVENTMASK(SDL_JOYBUTTONUP)
SDL_JOYEVENTMASK        = SDL_JOYAXISMOTIONMASK | \
                          SDL_JOYBALLMOTIONMASK | \
                          SDL_JOYHATMOTIONMASK | \
                          SDL_JOYBUTTONDOWNMASK | \
                          SDL_JOYBUTTONUPMASK
SDL_QUITMASK            = SDL_EVENTMASK(SDL_QUIT)
SDL_SYSWMEVENTMASK      = SDL_EVENTMASK(SDL_SYSWMEVENT)
SDL_VIDEORESIZEMASK     = SDL_EVENTMASK(SDL_VIDEORESIZE)
SDL_VIDEOEXPOSEMASK     = SDL_EVENTMASK(SDL_VIDEOEXPOSE)

# enum SDL_eventaction
(SDL_ADDEVENT,
    SDL_PEEKEVENT,
    SDL_GETEVENT) = range(3)

#From SDL_joystick.h (inserted manually)
SDL_HAT_RIGHTUP = SDL_HAT_RIGHT | SDL_HAT_UP
SDL_HAT_RIGHTDOWN = SDL_HAT_RIGHT | SDL_HAT_DOWN
SDL_HAT_LEFTUP = SDL_HAT_LEFT | SDL_HAT_UP
SDL_HAT_LEFTDOWN = SDL_HAT_LEFT | SDL_HAT_DOWN

# From SDL_video.h (inserted manually)
# enum SDL_GLattr
(SDL_GL_RED_SIZE,
    SDL_GL_GREEN_SIZE,
    SDL_GL_BLUE_SIZE,
    SDL_GL_ALPHA_SIZE,
    SDL_GL_BUFFER_SIZE,
    SDL_GL_DOUBLEBUFFER,
    SDL_GL_DEPTH_SIZE,
    SDL_GL_STENCIL_SIZE,
    SDL_GL_ACCUM_RED_SIZE,
    SDL_GL_ACCUM_GREEN_SIZE,
    SDL_GL_ACCUM_BLUE_SIZE,
    SDL_GL_ACCUM_ALPHA_SIZE,
    SDL_GL_STEREO,
    SDL_GL_MULTISAMPLEBUFFERS,
    SDL_GL_MULTISAMPLESAMPLES,
    SDL_GL_ACCELERATED_VISUAL,
    SDL_GL_SWAP_CONTROL) = range(17)

# enum SDL_GrabMode
(SDL_GRAB_QUERY,
    SDL_GRAB_OFF,
    SDL_GRAB_ON) = range(-1,2)

# From SDL_ttf.h (inserted manually)
TTF_STYLE_NORMAL    = 0x00
TTF_STYLE_BOLD      = 0x01
TTF_STYLE_ITALIC    = 0x02
TTF_STYLE_UNDERLINE = 0x04

# From SDL_mixer.h (inserted manually)
MIX_CHANNELS            = 8
MIX_DEFAULT_FREQUENCY   = 22050
MIX_MAX_VOLUME          = 128
MIX_CHANNEL_POST        = -2
MIX_EFFECTSMAXSPEED     = 'MIX_EFFECTSMAXSPEED'
MIX_DEFAULT_CHANNELS    = 2

if sys.byteorder == 'little':
    MIX_DEFAULT_FORMAT  = AUDIO_S16LSB 
else:
    MIX_DEFAULT_FORMAT  = AUDIO_S16MSB

# enum Mix_Fading
(MIX_NO_FADING,
    MIX_FADING_OUT,
    MIX_FADING_IN) = range(3)

# enum Mix_MusicType
(MUS_NONE,
    MUS_CMD,
    MUS_WAV,
    MUS_MOD,
    MUS_MID,
    MUS_OGG,
    MUS_MP3) = range(7)

# From SDL_sound.h (inserted manually):
# enum Sound_SampleFlags
SOUND_SAMPLEFLAG_NONE       = 0
SOUND_SAMPLEFLAG_CANSEEK    = 1
SOUND_SAMPLEFLAG_EOF        = 1 << 29
SOUND_SAMPLEFLAG_ERROR      = 1 << 30
SOUND_SAMPLEFLAG_EGAIN      = 1 << 31
