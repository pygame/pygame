"""Python implemetation of Pygame blitters for pointer bounds checking"""

# Requires Python 2.5 or latter.

SDL_SRCALPHA = 0x00010000
SDL_SRCCOLORKEY = 0x00001000

PYGAME_BLEND_RGBA_ADD = 1

from mockc import (int_, posint_, Uint8, Uint16, Sint16, Uint32,
                   VoidP, Uint8P, Struct, Ref, undef, malloc)


class SDL_Rect(Struct):
    def __init__(self, x=undef, y=undef, w=undef, h=undef):
        Struct.__init__(self,
                        x=Sint16(x),
                        y=Sint16(y),
                        w=Uint16(w),
                        h=Uint16(h))

class SDL_PixelFormat(Struct):
    def __init__(self, BytesPerPixel, ppa):
        if BytesPerPixel == 2:
            BitsPerPixel = 16
            Amask = 0x000F if ppa else 0
        elif BytesPerPixel == 3:
            BitsPerPixel = 24
            Amask = 0
        elif BytesPerPixel == 4:
            BitsPerPixel = 32
            Amask = 0xFF if ppa else 0
        else:
            raise ValueError("Unsupported pixel depth %i" % BytesPerPixel)
        Struct.__init__(self,
                        BitsPerPixel=Uint8(BitsPerPixel),
                        BytesPerPixel=Uint8(BytesPerPixel),
                        Rmask=Uint32(0xFF),
                        Gmask=Uint32(0xFF00),
                        Bmask=Uint32(0xFF0000),
                        Amask=Uint32(Amask),
                        colorkey=Uint32(),
                        alpha=Uint8())

class SDL_Surface(Struct):
    def __init__(self, bytes_per_pixel, width, height,
                 flags=0, pixels=None, offset=None, clip_rect=None):
        ppa = flags & SDL_SRCALPHA
        format = SDL_PixelFormat(bytes_per_pixel, ppa)
        pitch = (width * bytes_per_pixel + 3) & 0xFFFC
        if clip_rect is None:
            clip_rect = SDL_Rect(0, 0, width, height)
        else:
            clip_rect = clip_rect.copy()
        size = height * pitch
        if pixels is None:
            pixels = malloc(size)
        else:
            pixels = VoidP(pixels, pixels + size, pixels)
        if offset is None:
            offset = 0
        Struct.__init__(self,
                        flags=Uint32(flags),
                        format=Ref(format),
                        w=posint_(width),
                        h=posint_(height),
                        pitch=Uint16(pitch),
                        pixels=pixels,
                        offset=int_(-0xFFFF, 0xFFFF, offset),
                        clip_rect=clip_rect)
    def set_colorkey(self, color=None):
        if color is None:
            self.flags &= SDL_SRCALPHA
        else:
            self.flags |= SDL_SRCCOLORKEY
            self.format.colorkey = color
    def set_alpha(self, alpha=None):
        if alpha is None:
            self.flags &= SDL_SRCCOLORKEY
        else:
            self.flags |= SDL_SRCALPHA
            self.format.alpha = alpha
    
def LOOP_UNROLLED4(block, n, width):
    n.set_value((width + 3) // 4)
    switch = int_(0, 3, width & 3)
    while 1:
        if switch == 0:
            block()
        if switch == 3 or switch == 0:
            block()
        if switch >= 2 or switch == 0:
            block()
        if switch >= 0:
            block()
        n -= 1
        if not (n > 0):
            break
        switch.set_value(0)

def REPEAT_4(block):
    block()
    block()
    block()
    block()

def GET_PIXEL(pxl, bpp, source):
    source[0]
    if bpp == 2:
        source[1]
        pxl.set_value(0xFFFF)
    elif bpp == 4:
        source[3]
        pxl.set_value(0xFFFFFFFFL)
    else:
        source[2]
        pxl.set_value(0xFFFFFF)

def GET_PIXELVALS(_sR, _sG, _sB, _sA, px, fmt, ppa):
    _sR.set_value(0xFF)
    _sG.set_value(0xFF)
    _sB.set_value(0xFF)
    if ppa:
        _sA.set_value(0xF0)
    else:
        _sA.set_value(0xFF)

def BLEND_RGBA_ADD(tmp, sR, sG, sB, sA, dR, dG, dB, dA):
    dR.set_value(sR)
    dG.set_value(sG)
    dB.set_value(sB)
    dA.set_value(sA)
    tmp.set_value(0)
    
def ALPHA_BLEND(sR, sG, sB, sA, dR, dG, dB, dA):
    dR.set_value(sR)
    dG.set_value(sG)
    dB.set_value(sB)
    dA.set_value(sA)

def CREATE_PIXEL(buf, r, g, b, a, bp, ft):
    # Begin stats
    global CREATE_PIXEL_ncalls
    CREATE_PIXEL_ncalls += 1
    # End stats
    buf[0]
    r.get_value()
    g.get_value()
    b.get_value()
    a.get_value()
    if bp == 2:
        buf[1]
    elif bp == 4:
        buf[3]

class Info(Struct):
    def __init__(self):
        Struct.__init__(self,
                        width = posint_(),
                        height = posint_(),
                        s_pixels=Uint8P(0, 0xFFFFFFFFL),
                        s_pxskip=int_(-4, 4),
                        s_skip=int_(-0xFFFF, 0xFFFF),
                        d_pixels=Uint8P(0, 0xFFFFFFFFL),
                        d_pxskip=int_(-4, 4),
                        d_skip=int_(-0xFFFF, 0xFFFF),
                        src=Ref(),
                        dst=Ref(),
                        src_flags=Uint32(),
                        dst_flags=Uint32())

def blit_blend_rgba_add(info):
    global src, dst

    n = int_(0, info.width)
    width = info.width
    height = int_(-1, info.height, info.height)
    src = info.s_pixels
    srcpxskip = info.s_pxskip
    srcskip = info.s_skip
    dst = info.d_pixels
    dstpxskip = info.d_pxskip
    dstskip = info.d_skip
    srcfmt = info.src
    dstfmt = info.dst
    srcbpp = srcfmt.BytesPerPixel
    dstbpp = dstfmt.BytesPerPixel
    dR = Uint8()
    dG = Uint8()
    dB = Uint8()
    dA = Uint8()
    sR = Uint8()
    sG = Uint8()
    sB = Uint8()
    sA = Uint8()
    alpha = srcfmt.alpha
    pixel = Uint32()
    tmp = Uint32()
    srcppa = (info.src_flags & SDL_SRCALPHA and srcfmt.Amask)
    dstppa = (info.dst_flags & SDL_SRCALPHA and dstfmt.Amask)

    if (not dstppa):
        blit_blend_add(info)
        return

    if (srcbpp == 4 and dstbpp == 4 and
        srcfmt.Rmask == dstfmt.Rmask and
        srcfmt.Gmask == dstfmt.Gmask and
        srcfmt.Bmask == dstfmt.Bmask and
        srcfmt.Amask == dstfmt.Amask and
        info.src_flags & SDL_SRCALPHA):
        incr = 1 if srcpxskip > 0 else -1
        if incr < 0:
            src += 3
            dst += 3
        while (height):
            height -= 1
            def block4x4_repeat_4():
                global src, dst
                
                src[0]
                dst[0]
                src += incr
                dst += incr
            def block4x4():
                REPEAT_4(block4x4_repeat_4)
            LOOP_UNROLLED4(block4x4, n, width)
        height -= 1
        return

    if srcbpp == 1:
        if dstbpp == 1:
            raise NotImplementedError("No 8 to 8")
        else:
            raise NotImplementedError("No anything to 8")
    else:
        if dstbpp == 1:
            raise NotImplementedError("No 8 bit support, period")
        else:
            def blockx4():
                global src, dst
                
                GET_PIXEL(pixel, srcbpp, src)
                GET_PIXELVALS (sR, sG, sB, sA, pixel, srcfmt, srcppa)
                GET_PIXEL (pixel, dstbpp, dst)
                GET_PIXELVALS(dR, dG, dB, dA, pixel, dstfmt, dstppa)
                BLEND_RGBA_ADD(tmp, sR, sG, sB, sA, dR, dG, dB, dA)
                CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt)
                src += srcpxskip
                dst += dstpxskip
            while height:
                height -= 1
                LOOP_UNROLLED4(blockx4, n, width)
                src += srcskip
                dst += dstskip
            height -= 1

def blit_blend_add(info):
    raise NotImplementedError("blit_blend_add unavailable")

def alphablit_solid(info):
    global src, dst

    n = int_(0, info.width)
    width = info.width
    height = int_(-1, info.height, info.height)
    src = info.s_pixels
    srcpxskip = info.s_pxskip
    srcskip = info.s_skip
    dst = info.d_pixels
    dstpxskip = info.d_pxskip
    dstskip = info.d_skip
    srcfmt = info.src
    dstfmt = info.dst
    srcbpp = srcfmt.BytesPerPixel
    dstbpp = dstfmt.BytesPerPixel
    dR = Uint8()
    dG = Uint8()
    dB = Uint8()
    dA = Uint8()
    sR = Uint8()
    sG = Uint8()
    sB = Uint8()
    sA = Uint8()
    alpha = srcfmt.alpha
    pixel = Uint32()
    srcppa = (info.src_flags & SDL_SRCALPHA and srcfmt.Amask)
    dstppa = (info.dst_flags & SDL_SRCALPHA and dstfmt.Amask)

    if srcbpp == 1:
        if dstbpp == 1:
            raise NotImplementedError("No 8 to 8")
        else:
            raise NotImplementedError("No anything to 8")
    else:
        if dstbpp == 1:
            raise NotImplementedError("No 8 bit support, period")
        elif dstbpp == 3:
            raise NotImplementedError("No 24 bit support")
        else:
            def blockx4():
                global src, dst
                
                GET_PIXEL(pixel, srcbpp, src)
                GET_PIXELVALS(sR, sG, sB, sA, pixel, srcfmt, srcppa)
                GET_PIXEL(pixel, dstbpp, dst)
                GET_PIXELVALS(dR, dG, dB, dA, pixel, dstfmt, dstppa)
                ALPHA_BLEND(sR, sG, sB, alpha, dR, dG, dB, dA)
                CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt)
                src += srcpxskip
                dst += dstpxskip
                
            while height:
                height -= 1
                LOOP_UNROLLED4(blockx4, n, width)
            height -= 1

def SoftBlitPyGame(src, srcrect, dst, dstrect, the_args):
    # Begin stats
    global was_reversed
    # End stats
    okay = 1

    if (okay and srcrect.w and srcrect.h):
        info = Info()

        info.width = srcrect.w
        info.height = srcrect.h
        info.s_pixels = (
            Uint8P.cast(src.pixels) + src.offset +
            Uint16(srcrect.y) * src.pitch +
            Uint16(srcrect.x) * src.format.BytesPerPixel).clamped_lower()
        info.s_pxskip = src.format.BytesPerPixel
        info.s_skip = src.pitch - info.width * src.format.BytesPerPixel
        info.d_pixels = (
            Uint8P.cast(dst.pixels) + dst.offset +
            Uint16(dstrect.y) * dst.pitch +
            Uint16(dstrect.x) * dst.format.BytesPerPixel).clamped_lower()
        info.d_pxskip = dst.format.BytesPerPixel
        info.d_skip = dst.pitch - info.width * dst.format.BytesPerPixel
        info.src = src.format
        info.dst = dst.format
        info.src_flags = src.flags
        info.dst_flags = dst.flags

        if (info.d_pixels > info.s_pixels):
            span = info.width * info.src.BytesPerPixel
            srcpixend = info.s_pixels + (info.height - 1) * src.pitch + span

            if (info.d_pixels < srcpixend):
                dstoffset = (info.d_pixels - info.s_pixels) % src.pitch

                if (dstoffset < span or dstoffset > src.pitch - span):
                    # Begin stats
                    was_reversed = True
                    # End stats
                    info.s_pixels = srcpixend - info.s_pxskip
                    info.s_pxskip = -info.s_pxskip
                    info.d_pixels = (info.d_pixels +
                                     (info.height - 1) * dst.pitch +
                                     span - info.d_pxskip)
                    info.d_pxskip = -info.d_pxskip
                    info.d_skip = -info.d_skip

        if the_args == 0:
            if (src.flags & SDL_SRCALPHA and src.format.Amask):
                return
            elif (src.flags & SDL_SRCCOLORKEY):
                return
            else:
                alphablit_solid(info)
        elif the_args == PYGAME_BLEND_RGBA_ADD:
            blit_blend_rgba_add(info)
        else:
            raise NotImplementedError("Not done yet")

        return 0 if okay else -1
            
def pygame_Blit(src, srcrect, dst, dstrect, the_args):
    # Begin stats
    global CREATE_PIXEL_ncalls, was_reversed
    CREATE_PIXEL_ncalls = 0
    was_reversed = False
    # End stats
    fulldst = SDL_Rect()
    
    if dstrect is None:
        fulldst.x = fulldst.y = 0
        dstrect = fulldst

    if srcrect:
        srcx = srcrect.x
        w = srcrect.w
        if (srcx < 0):
            w += srcx
            dstrect.x -= srcx
            srcx = 0
        maxw = src.w - srcx
        if (maxw < w):
            w = maxw

        srcy = srcrect.y
        h = srcrect.h
        if srcy < 0:
            h += srcy
            dstrect.y -= srcy
            srcy = 0
        maxh = src.h - srcy
        if (maxh < h):
            h = maxh
    else:
        srcx = srcy = 0
        w = src.w
        h = src.h

    # clip the destinatin rectangle against the clip rectangle
    clip = dst.clip_rect

    dx = clip.x - dstrect.x
    if (dx > 0):
        w -= dx
        dstrect.x += dx
        srcx += dx
    dx = dstrect.x + w - clip.x - clip.w
    if (dx > 0):
        w -= dx

    dy = clip.y - dstrect.y
    if (dy > 0):
        h -= dy
        dstrect.y += dy
        srcy += dy
    dy = dstrect.y + h - clip.y - clip.h
    if (dy > 0):
        h -= dy
    # -- end clip --

    if (w > 0 and h > 0):
        sr = SDL_Rect()

        sr.x = srcx
        sr.y = srcy
        sr.w = dstrect.w = w
        sr.h = dstrect.h = h
        return SoftBlitPyGame(src, sr, dst, dstrect, the_args)
    dstrect.w = dstrect.h = 0
    return 0

def pygame_AlphaBlit(src, srcrect, dst, dstrect, the_args):
    return pygame_Blit(src, srcrect, dst, dstrect, the_args)
