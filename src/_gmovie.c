/*
  pygame - Python Game Library

  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU Library General Public
  License as published by the Free Software Foundation; either
  version 2 of the License, or (at your option) any later version.

  This library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Library General Public License for more details.

  You should have received a copy of the GNU Library General Public
  License along with this library; if not, write to the Free
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
  
*/

/*
 * _movie - movie support for pygame with ffmpeg
 * Author: Tyler Laing
 *
 * This module allows for the loading of, playing, pausing, stopping, and so on
 *  of a video file. Any format supported by ffmpeg is supported by this
 *  video player. Any bugs, please email trinioler@gmail.com :)
 */
 

#ifndef _GMOVIE_H_
#include "_gmovie.h"
#endif

#ifdef __MINGW32__
#undef main /* We don't want SDL to override our main() */
#endif

/* packet queue handling */
void packet_queue_init(PacketQueue *q)
{
    if(!q)
    {
        q=(PacketQueue *)PyMem_Malloc(sizeof(PacketQueue));
    }
    if(!q->mutex)
        q->mutex = SDL_CreateMutex();
    q->abort_request=0;
}

void packet_queue_flush(PacketQueue *q)
{
    AVPacketList *pkt, *pkt1;

    if(q->mutex)
        SDL_LockMutex(q->mutex);

    for(pkt = q->first_pkt; pkt != NULL; pkt = pkt1)
    {
        pkt1 = pkt->next;
        av_free_packet(&pkt->pkt);
        PyMem_Free(pkt);
    }
    q->last_pkt = NULL;
    q->first_pkt = NULL;
    q->nb_packets = 0;
    q->size = 0;

    if(q->mutex)
        SDL_UnlockMutex(q->mutex);
}

void packet_queue_end(PacketQueue *q, int end)
{
    AVPacketList *pkt, *pkt1;

    for(pkt = q->first_pkt; pkt != NULL; pkt = pkt1)
    {
        pkt1 = pkt->next;
        av_free_packet(&pkt->pkt);
    }
    if(end==0)
    {
        //we only destroy the mutex if its the last loop. This way we just reuse and don't fragment memory.
        if(q->mutex)
        {
            SDL_DestroyMutex(q->mutex);
        }
    }
}

int packet_queue_put(PacketQueue *q, AVPacket *pkt)
{
    AVPacketList *pkt1;

    /* duplicate the packet */
    if (pkt!=&flush_pkt && av_dup_packet(pkt) < 0)
        return -1;

    pkt1 = PyMem_Malloc(sizeof(AVPacketList));

    if (!pkt1)
        return -1;
    pkt1->pkt = *pkt;
    pkt1->next = NULL;

    if(q->mutex)
        SDL_LockMutex(q->mutex);

    if (!q->last_pkt)
        q->first_pkt = pkt1;
    else
        q->last_pkt->next = pkt1;
    q->last_pkt = pkt1;
    q->nb_packets++;
    q->size += pkt1->pkt.size;
    /* XXX: should duplicate packet data in DV case */


    if(q->mutex)
        SDL_UnlockMutex(q->mutex);

    return 0;
}

void packet_queue_abort(PacketQueue *q)
{
    if(q->mutex)
        SDL_LockMutex(q->mutex);

    q->abort_request = 1;

    if(q->mutex)
        SDL_UnlockMutex(q->mutex);
}

/* return < 0 if aborted, 0 if no packet and > 0 if packet.  */
int packet_queue_get(PacketQueue *q, AVPacket *pkt, int block)
{
    AVPacketList *pkt1;
    int ret;

    if(q->mutex)
        SDL_LockMutex(q->mutex);

    for(;;)
    {
        if (q->abort_request)
        {
            ret = -1;
            break;
        }

        pkt1 = q->first_pkt;
        if (pkt1)
        {
            q->first_pkt = pkt1->next;
            if (!q->first_pkt)
                q->last_pkt = NULL;
            q->nb_packets--;
            q->size -= pkt1->pkt.size;
            *pkt = pkt1->pkt;
            PyMem_Free(pkt1);
            ret = 1;
            break;
        }
        else if (!block)
        {
            ret = 0;
            break;
        }
        else
        {
            ret=0;
            break;
        }
    }
    if(q->mutex)
        SDL_UnlockMutex(q->mutex);

    return ret;
}

/* subtitles don't work yet. Code remains until it is needed */
#if 0
void blend_subrect(AVPicture *dst, const AVSubtitleRect *rect, int imgw, int imgh)
{
    int wrap, wrap3, width2, skip2;
    int y, u, v, a, u1, v1, a1, w, h;
    uint8_t *lum, *cb, *cr;
    const uint8_t *p;
    const uint32_t *pal;
    int dstx, dsty, dstw, dsth;

    dstw = av_clip(rect->w, 0, imgw);
    dsth = av_clip(rect->h, 0, imgh);
    dstx = av_clip(rect->x, 0, imgw - dstw);
    dsty = av_clip(rect->y, 0, imgh - dsth);
    lum = dst->data[0] + dsty * dst->linesize[0];
    cb = dst->data[1] + (dsty >> 1) * dst->linesize[1];
    cr = dst->data[2] + (dsty >> 1) * dst->linesize[2];

    width2 = ((dstw + 1) >> 1) + (dstx & ~dstw & 1);
    skip2 = dstx >> 1;
    wrap = dst->linesize[0];
    wrap3 = rect->pict.linesize[0];
    p = rect->pict.data[0];
    pal = (const uint32_t *)rect->pict.data[1];  /* Now in YCrCb! */

    if (dsty & 1)
    {
        lum += dstx;
        cb += skip2;
        cr += skip2;

        if (dstx & 1)
        {
            YUVA_IN(y, u, v, a, p, pal);
            lum[0] = _ALPHA_BLEND(a, lum[0], y, 0);
            cb[0] = _ALPHA_BLEND(a >> 2, cb[0], u, 0);
            cr[0] = _ALPHA_BLEND(a >> 2, cr[0], v, 0);
            cb++;
            cr++;
            lum++;
            p += BPP;
        }
        for(w = dstw - (dstx & 1); w >= 2; w -= 2)
        {
            YUVA_IN(y, u, v, a, p, pal);
            u1 = u;
            v1 = v;
            a1 = a;
            lum[0] = _ALPHA_BLEND(a, lum[0], y, 0);

            YUVA_IN(y, u, v, a, p + BPP, pal);
            u1 += u;
            v1 += v;
            a1 += a;
            lum[1] = _ALPHA_BLEND(a, lum[1], y, 0);
            cb[0] = _ALPHA_BLEND(a1 >> 2, cb[0], u1, 1);
            cr[0] = _ALPHA_BLEND(a1 >> 2, cr[0], v1, 1);
            cb++;
            cr++;
            p += 2 * BPP;
            lum += 2;
        }
        if (w)
        {
            YUVA_IN(y, u, v, a, p, pal);
            lum[0] = _ALPHA_BLEND(a, lum[0], y, 0);
            cb[0] = _ALPHA_BLEND(a >> 2, cb[0], u, 0);
            cr[0] = _ALPHA_BLEND(a >> 2, cr[0], v, 0);
            p++;
            lum++;
        }
        p += wrap3 - dstw * BPP;
        lum += wrap - dstw - dstx;
        cb += dst->linesize[1] - width2 - skip2;
        cr += dst->linesize[2] - width2 - skip2;
    }
    for(h = dsth - (dsty & 1); h >= 2; h -= 2)
    {
        lum += dstx;
        cb += skip2;
        cr += skip2;

        if (dstx & 1)
        {
            YUVA_IN(y, u, v, a, p, pal);
            u1 = u;
            v1 = v;
            a1 = a;
            lum[0] = _ALPHA_BLEND(a, lum[0], y, 0);
            p += wrap3;
            lum += wrap;
            YUVA_IN(y, u, v, a, p, pal);
            u1 += u;
            v1 += v;
            a1 += a;
            lum[0] = _ALPHA_BLEND(a, lum[0], y, 0);
            cb[0] = _ALPHA_BLEND(a1 >> 2, cb[0], u1, 1);
            cr[0] = _ALPHA_BLEND(a1 >> 2, cr[0], v1, 1);
            cb++;
            cr++;
            p += -wrap3 + BPP;
            lum += -wrap + 1;
        }
        for(w = dstw - (dstx & 1); w >= 2; w -= 2)
        {
            YUVA_IN(y, u, v, a, p, pal);
            u1 = u;
            v1 = v;
            a1 = a;
            lum[0] = _ALPHA_BLEND(a, lum[0], y, 0);

            YUVA_IN(y, u, v, a, p + BPP, pal);
            u1 += u;
            v1 += v;
            a1 += a;
            lum[1] = _ALPHA_BLEND(a, lum[1], y, 0);
            p += wrap3;
            lum += wrap;

            YUVA_IN(y, u, v, a, p, pal);
            u1 += u;
            v1 += v;
            a1 += a;
            lum[0] = _ALPHA_BLEND(a, lum[0], y, 0);

            YUVA_IN(y, u, v, a, p + BPP, pal);
            u1 += u;
            v1 += v;
            a1 += a;
            lum[1] = _ALPHA_BLEND(a, lum[1], y, 0);

            cb[0] = _ALPHA_BLEND(a1 >> 2, cb[0], u1, 2);
            cr[0] = _ALPHA_BLEND(a1 >> 2, cr[0], v1, 2);

            cb++;
            cr++;
            p += -wrap3 + 2 * BPP;
            lum += -wrap + 2;
        }
        if (w)
        {
            YUVA_IN(y, u, v, a, p, pal);
            u1 = u;
            v1 = v;
            a1 = a;
            lum[0] = _ALPHA_BLEND(a, lum[0], y, 0);
            p += wrap3;
            lum += wrap;
            YUVA_IN(y, u, v, a, p, pal);
            u1 += u;
            v1 += v;
            a1 += a;
            lum[0] = _ALPHA_BLEND(a, lum[0], y, 0);
            cb[0] = _ALPHA_BLEND(a1 >> 2, cb[0], u1, 1);
            cr[0] = _ALPHA_BLEND(a1 >> 2, cr[0], v1, 1);
            cb++;
            cr++;
            p += -wrap3 + BPP;
            lum += -wrap + 1;
        }
        p += wrap3 + (wrap3 - dstw * BPP);
        lum += wrap + (wrap - dstw - dstx);
        cb += dst->linesize[1] - width2 - skip2;
        cr += dst->linesize[2] - width2 - skip2;
    }
    /* handle odd height */
    if (h)
    {
        lum += dstx;
        cb += skip2;
        cr += skip2;

        if (dstx & 1)
        {
            YUVA_IN(y, u, v, a, p, pal);
            lum[0] = _ALPHA_BLEND(a, lum[0], y, 0);
            cb[0] = _ALPHA_BLEND(a >> 2, cb[0], u, 0);
            cr[0] = _ALPHA_BLEND(a >> 2, cr[0], v, 0);
            cb++;
            cr++;
            lum++;
            p += BPP;
        }
        for(w = dstw - (dstx & 1); w >= 2; w -= 2)
        {
            YUVA_IN(y, u, v, a, p, pal);
            u1 = u;
            v1 = v;
            a1 = a;
            lum[0] = _ALPHA_BLEND(a, lum[0], y, 0);

            YUVA_IN(y, u, v, a, p + BPP, pal);
            u1 += u;
            v1 += v;
            a1 += a;
            lum[1] = _ALPHA_BLEND(a, lum[1], y, 0);
            cb[0] = _ALPHA_BLEND(a1 >> 2, cb[0], u, 1);
            cr[0] = _ALPHA_BLEND(a1 >> 2, cr[0], v, 1);
            cb++;
            cr++;
            p += 2 * BPP;
            lum += 2;
        }
        if (w)
        {
            YUVA_IN(y, u, v, a, p, pal);
            lum[0] = _ALPHA_BLEND(a, lum[0], y, 0);
            cb[0] = _ALPHA_BLEND(a >> 2, cb[0], u, 0);
            cr[0] = _ALPHA_BLEND(a >> 2, cr[0], v, 0);
        }
    }
}

void free_subpicture(SubPicture *sp)
{
    int i;

    for (i = 0; i < sp->sub.num_rects; i++)
    {
        av_freep(&sp->sub.rects[i]->pict.data[0]);
        av_freep(&sp->sub.rects[i]->pict.data[1]);
        av_freep(&sp->sub.rects[i]);
    }

    av_free(sp->sub.rects);

    memset(&sp->sub, 0, sizeof(AVSubtitle));
}
#endif

/* Sets the value of the variable width. Acts like a macro */
void inline get_width(PyMovie *movie, int *width)
{
    if(movie->resize_w)
    {
        *width=movie->width;
    }
    else
    {
        if(movie->video_st)
            *width=movie->video_st->codec->width;
    }
}
/* Sets the value of the variable height. Acts like a macro */
void inline get_height(PyMovie *movie, int *height)
{
    if(movie->resize_h)
    {
        *height=movie->height;
    }
    else
    {
        if(movie->video_st)
            *height=movie->video_st->codec->height;
    }
}

void get_height_width(PyMovie *movie, int *height, int*width)
{
    get_height(movie, height);
    get_width(movie, width);
}

inline void jamPixels(int ix, AVPicture *picture, uint32_t *rgb, SDL_Surface *surface)
{
    //uint32_t *rgb           = surface->pixels;
    uint8_t red   = picture->data[0][ix];
    uint8_t green = picture->data[0][ix+1];
    uint8_t blue  = picture->data[0][ix+2];
    //skip the alpha... we don't care
    /* shift components to the correct place in pixel */

    *rgb = ( red   << (long) surface->format->Rshift) | /* red */
           ( blue  << (long) surface->format->Bshift ) | /* green */
           ( green << (long) surface->format->Gshift ) | /* blue */
           ( 0   << (long) surface->format->Ashift);
}

//transfers data from the AVPicture written to by swscale to a surface
void WritePicture2Surface(AVPicture *picture, SDL_Surface *surface, int w, int h)
{
    /* AVPicture initialized with PIX_FMT_RGBA only fills pict->data[0]
     *  This however is only in {R,G,B, A} format. So we just copy the data over. 
     */
    /* Loop unrolling:
     * 	We define a blocksize, and so we increment the index counter by blocksize*rgbstep
     * 	All common resolutions are nicely divisible by 8(because 8 is a power of 2...)
     *  An uncommon resolution  could have between 1 and 7 bytes left to convert... 
     *   which I guess we'll leave alone. Its just 1-2 pixels in the lower right corner.
     *  So we repeat the same actions blocksize times.  
     */
    int64_t   blocksize     = 8;
    uint32_t *rgb           = surface->pixels;
    int       BytesPerPixel = RGBSTEP;
    int64_t   size          = w*h*BytesPerPixel;
    int64_t   ix            = 0;
    int64_t   blocklimit    = (size/blocksize)*blocksize;
    while(ix<blocklimit)
    {
        //this will be unrolled by the compiler, meaning that we do less comparisons by a factor of blocksize
        int i =0;
        for(;i<blocksize;i++)
        {
            jamPixels(ix, picture, rgb, surface);
            rgb++;
            ix+=RGBSTEP;
        }
    }
}


int video_display(PyMovie *movie)
{
    double ret=1;

    VidPicture *vp = &movie->pictq[movie->pictq_rindex];
    if (movie->video_stream>=0 && vp->ready)
    {
        video_image_display(movie);
    }
    else if(!vp->ready)
    {
        ret=0;
    }

    /* If we didn't actually display the image, we need to not clear our timer out in decoder */
    return ret;
}

int video_image_display(PyMovie *movie)
{
    /* Wrapped by video_display, which has a lock on the movie object */
    DECLAREGIL

    VidPicture *vp;
    //SubPicture *sp;
    float aspect_ratio;
    int width, height, x, y;
    vp = &movie->pictq[movie->pictq_rindex];
    vp->ready =0;
    //GRABGIL
    //PySys_WriteStdout("video_current_pts: %f\tvp->pts: %f\ttime: %f\n", movie->video_current_pts, vp->pts, (av_gettime()/1000.0)-(movie->timing+movie->last_showtime));
    //RELEASEGIL
    //set up the aspect ratio values..
#if LIBAVFORMAT_VERSION_INT>= 3415808
    
        if (movie->video_st->sample_aspect_ratio.num)
            aspect_ratio = av_q2d(movie->video_st->sample_aspect_ratio);
        else if (movie->video_st->codec->sample_aspect_ratio.num)
            aspect_ratio = av_q2d(movie->video_st->codec->sample_aspect_ratio);
        else
            aspect_ratio = 0;
        if (aspect_ratio <= 0.0)
            aspect_ratio = 1.0;
    
#else
    
        aspect_ratio = 1.0;
    
#endif
    //then we load in width and height values based on the aspect ration and w/h.
    int w=0;
    int h=0;
    get_height_width(movie, &h, &w);
    aspect_ratio *= (float)w / h;
    /* XXX: we suppose the screen has a 1.0 pixel ratio */
    height = vp->height;
    width = ((int)rint(height * aspect_ratio)) & ~1;
    if (width > vp->width)
    {
        width = vp->width;
        height = ((int)rint(width / aspect_ratio)) & ~1;
    }
    x = (vp->width - width) / 2;
    y = (vp->height - height) / 2;

    //we set the rect to have the values we need for blitting/overlay display
    	
	    vp->dest_rect.x = vp->xleft + x;
    	vp->dest_rect.y = vp->ytop  + y;

    	vp->dest_rect.w=width;
 	   	vp->dest_rect.h=height;
	
    if (vp->dest_overlay && vp->overlay>0 && !movie->skip_frame)
    {
        //SDL_Delay(10);
#if 0
        if (movie->sub_st)
        {
            if (movie->subpq_size > 0)
            {
                sp = &movie->subpq[movie->subpq_rindex];
                AVPicture pict;
                int i;

                if (vp->pts >= sp->pts + ((float) sp->sub.start_display_time / 1000))
                {
                    SDL_LockYUVOverlay (vp->dest_overlay);

                    pict.data[0] = vp->dest_overlay->pixels[0];
                    pict.data[1] = vp->dest_overlay->pixels[2];
                    pict.data[2] = vp->dest_overlay->pixels[1];

                    pict.linesize[0] = vp->dest_overlay->pitches[0];
                    pict.linesize[1] = vp->dest_overlay->pitches[2];
                    pict.linesize[2] = vp->dest_overlay->pitches[1];

                    for (i = 0; i < sp->sub.num_rects; i++)
                        blend_subrect(&pict, sp->sub.rects[i],
                                      vp->dest_overlay->w, vp->dest_overlay->h);

                    SDL_UnlockYUVOverlay (vp->dest_overlay);
                }
            }
        }
#endif
        if(vp->overlay>0)
        {
            SDL_LockYUVOverlay(vp->dest_overlay);
            SDL_DisplayYUVOverlay(vp->dest_overlay, &vp->dest_rect);
            SDL_UnlockYUVOverlay(vp->dest_overlay);
        }
    }
    else if(vp->dest_surface && vp->overlay<=0)
    {
        SDL_BlitSurface(vp->dest_surface, &vp->dest_rect, movie->canon_surf, &vp->dest_rect);
    }

    movie->pictq_rindex= (movie->pictq_rindex+1)%VIDEO_PICTURE_QUEUE_SIZE;
    movie->pictq_size--;
    if(movie->skip_frame)
        movie->skip_frame=0;
    video_refresh_timer(movie);
    return 1;
}

int video_open(PyMovie *movie, int index)
{
    int w=0;
    int h=0;

    DECLAREGIL
    get_height_width(movie, &h, &w);
    int tw=w;
    int th=h;
    VidPicture *vp;
    for(index=0;index<VIDEO_PICTURE_QUEUE_SIZE; index++)
    {
        vp = &movie->pictq[index];
        if(
            //If we have no overlay, and we are supposed to, we jump right in
            (!vp->dest_overlay && movie->overlay>0) ||
            (
                /* otherwise, we need to enter this block if
                 * we need to resize AND there is an overlay AND
                 *  it is not the right size
                 */
                (movie->resize_w||movie->resize_h) &&
                vp->dest_overlay &&
                (vp->height!=h || vp->width!=w)
            )
        )
        {
            if(movie->resize_w || movie->resize_h)
            {
                //we free this overlay, because we KNOW its not the right size.
                SDL_FreeYUVOverlay(vp->dest_overlay);
            }
            if(movie->overlay>0)
            {
                //now we have to open an overlay up
                SDL_Surface *screen;
                if (!SDL_WasInit (SDL_INIT_VIDEO))
                {
                    GRABGIL
                    RAISE(PyExc_SDLError,"cannot create overlay without pygame.display initialized");
                    RELEASEGIL
                    return -1;
                }
                screen = SDL_GetVideoSurface ();
                if (!screen || (screen && (screen->w!=w || screen->h !=h)))
                {
                    //resize the main screen
                    screen = SDL_SetVideoMode(w, h, 0, SDL_SWSURFACE);
                    if(!screen)
                    {
                        GRABGIL
                        RAISE(PyExc_SDLError, "Could not initialize a new video surface.");
                        RELEASEGIL
                        return -1;
                    }
                }
                //create a new overlay
                vp->dest_overlay = SDL_CreateYUVOverlay (w, h, SDL_YV12_OVERLAY, screen);
                if (!vp->dest_overlay)
                {
                    GRABGIL
                    RAISE (PyExc_SDLError, "Cannot create overlay");
                    RELEASEGIL
                    return -1;
                }
                vp->overlay = movie->overlay;
            }
        }
        if (!vp->dest_surface && movie->overlay<=0)
        {
            //now we have to open an overlay up
            if(movie->overlay<=0)
            {
                SDL_Surface *screen = movie->canon_surf;
                if (!SDL_WasInit (SDL_INIT_VIDEO))
                {
                    GRABGIL
                    RAISE(PyExc_SDLError,"cannot create surfaces without pygame.display initialized");
                    RELEASEGIL
                    return -1;
                }
                if (!screen)
                {
                    GRABGIL
                    RAISE(PyExc_SDLError, "No video surface given."); //ideally this should have
                    RELEASEGIL										  // happen if there's some cleaning up.
                    return -1;
                }
                if(screen->h!=h)
                {
                    th=screen->h;
                }
                if(screen->w!=w)
                {
                    tw=screen->w;
                }

                vp->dest_surface = SDL_CreateRGBSurface(screen->flags,
                                                        tw,
                                                        th,
                                                        screen->format->BitsPerPixel,
                                                        screen->format->Rmask,
                                                        screen->format->Gmask,
                                                        screen->format->Bmask,
                                                        screen->format->Amask);

                if (!vp->dest_surface)
                {
                    GRABGIL
                    RAISE (PyExc_SDLError, "Cannot create new surface.");
                    RELEASEGIL
                    return -1;
                }
                vp->overlay = movie->overlay;
            }
        }
        vp->width = tw;
        vp->height = th;
        vp->ytop=movie->ytop;
        vp->xleft=movie->xleft;

    }
    return 0;
}

/* called to determine a time to show each frame */
void video_refresh_timer(PyMovie* movie)
{
    DECLAREGIL
    double actual_delay, delay, sync_threshold, ref_clock, diff;
    VidPicture *vp;

    double cur_time=av_gettime();

    if (movie->video_st)
    { /*shouldn't ever even get this far if no video_st*/
        movie->diff_co ++;

        /* dequeue the picture */
        vp = &movie->pictq[movie->pictq_rindex];

        /* update current video pts */
        movie->video_current_pts = vp->pts;
        movie->video_current_pts_time = cur_time;

        /* compute nominal delay */
        delay = movie->video_current_pts - movie->frame_last_pts;
        if (delay <= 0 || delay >= 10.0)
        {
            /* if incorrect delay, use previous one */
            delay = movie->frame_last_delay;
        }
        else
        {
            movie->frame_last_delay = delay;
        }
        movie->frame_last_pts = movie->video_current_pts;

        /* update delay to follow master synchronisation source */
        if (((movie->av_sync_type == AV_SYNC_AUDIO_MASTER && movie->audio_st) ||
                movie->av_sync_type == AV_SYNC_EXTERNAL_CLOCK))
        {
            /* if video is slave, we try to correct big delays by
               duplicating or deleting a frame */
            ref_clock = getAudioClock();
            diff = movie->video_current_pts - ref_clock;
            /* skip or repeat frame. We take into account the
               delay to compute the threshold. I still don't know
               if it is the best guess */

            sync_threshold = FFMAX(AV_SYNC_THRESHOLD, delay);
            if (fabs(diff) < AV_NOSYNC_THRESHOLD)
            {
                if (diff <= -sync_threshold)
                {
                    movie->skip_frame=1;
                    delay = 0;
                }
                else if (diff >= sync_threshold)
                    delay = 2 * delay;
            }
        }

        movie->frame_timer += delay;
        /* compute the REAL delay (we need to do that to avoid
           long term errors */
        actual_delay = movie->frame_timer - (cur_time / 1000000.0);
        if (actual_delay < 0.010)
        {
            /* XXX: should skip picture */
            actual_delay = 0.010;
        }

        GRABGIL
        //PySys_WriteStdout("Actual Delay: %f\tdelay: %f\tdiff: %f\tpts: %f\tFrame-timer: %f\tCurrent_time: %f\tsync_thres: %f\n", (actual_delay*1000.0)+10, delay, diff, movie->video_current_pts, movie->frame_timer, (cur_time / 1000000.0), sync_threshold);
        //double audio = getAudioClock();
        //PySys_WriteStdout("Audio_Clock: %f\tVideo_Clock: %f\tDiff: %f\n", ref_clock, movie->video_current_pts, ref_clock-movie->video_current_pts);
        movie->timing = (actual_delay*1000.0)+10;
        RELEASEGIL
    }

}

int queue_picture(PyMovie *movie, AVFrame *src_frame)
{
    DECLAREGIL
    int dst_pix_fmt;
    AVPicture pict;
    VidPicture *vp;
    struct SwsContext *img_convert_ctx=movie->img_convert_ctx;

    vp = &movie->pictq[movie->pictq_windex];

    int w=0;
    int h=0;
    get_height_width(movie, &h, &w);
    int pw, ph;

    if(vp->dest_surface)
    {
        pw=vp->dest_surface->w;
        ph=vp->dest_surface->h;
    }

    dst_pix_fmt = PIX_FMT_YUV420P;
#ifdef PROFILE

    int64_t before = av_gettime();
#endif

    if (vp->dest_overlay && vp->overlay>0)
    {
        /* get a pointer on the bitmap */
        SDL_LockYUVOverlay(vp->dest_overlay);

        pict.data[0] = vp->dest_overlay->pixels[0];
        pict.data[1] = vp->dest_overlay->pixels[2];
        pict.data[2] = vp->dest_overlay->pixels[1];
        pict.linesize[0] = vp->dest_overlay->pitches[0];
        pict.linesize[1] = vp->dest_overlay->pitches[2];
        pict.linesize[2] = vp->dest_overlay->pitches[1];
    }
    else if(vp->dest_surface)
    {
        /* get a pointer on the bitmap */
        if(RGB24)
        {
            dst_pix_fmt = PIX_FMT_RGB24;
        }
        else if(RGBA)
        {
            dst_pix_fmt = PIX_FMT_RGBA;
        }
        avpicture_alloc(&pict, dst_pix_fmt, pw, ph);
        SDL_LockSurface(vp->dest_surface);
    }
    int sws_flags = SWS_BICUBIC;
    img_convert_ctx = sws_getCachedContext(img_convert_ctx,
                                           movie->video_st->codec->width,
                                           movie->video_st->codec->height,
                                           movie->video_st->codec->pix_fmt,
                                           w,
                                           h,
                                           dst_pix_fmt,
                                           sws_flags,
                                           NULL, NULL, NULL);
    if (img_convert_ctx == NULL)
    {
        fprintf(stderr, "Cannot initialize the conversion context\n");
        exit(1);
    }
    movie->img_convert_ctx = img_convert_ctx;

    if((movie->resize_w||movie->resize_h) || vp->dest_overlay)
    {
        sws_scale(img_convert_ctx,
                  src_frame->data,
                  src_frame->linesize,
                  0,
                  h,
                  pict.data,
                  pict.linesize);
    }
    else if(vp->dest_surface)
    {
        sws_scale(img_convert_ctx,
                  src_frame->data,
                  src_frame->linesize,
                  0,
                  ph,
                  pict.data,
                  pict.linesize);
    }

    if (vp->dest_overlay && vp->overlay>0)
    {
        SDL_UnlockYUVOverlay(vp->dest_overlay);
    }
    else if(vp->dest_surface)
    {
        WritePicture2Surface(&pict, vp->dest_surface, pw, ph);
        SDL_UnlockSurface(vp->dest_surface);
        //avpicture_free(&pict);
    }
#ifdef PROFILE
    TimeSampleNode *sample = (TimeSampleNode *)PyMem_Malloc(sizeof(TimeSampleNode));
    sample->next=NULL;
    sample->sample = av_gettime()-before;
    if (!movie->istats->last || movie->istats->first->sample==0)
        movie->istats->first = sample;
    else
        movie->istats->last->next = sample;
    movie->istats->last = sample;
    movie->istats->n_samples++;
#endif

    vp->pts = movie->pts;
    movie->pictq_windex = (movie->pictq_windex+1)%VIDEO_PICTURE_QUEUE_SIZE;
    movie->pictq_size++;
    vp->ready=1;
    return 0;
}


void update_video_clock(PyMovie *movie, AVFrame* frame, double pts1)
{
    double frame_delay, pts;

    pts = pts1;

    if (pts != 0)
    {
        /* update video clock with pts, if present */
        movie->video_current_pts = pts;
    }
    else
    {
        pts = movie->video_current_pts;
    }
    /* update video clock for next frame */
    frame_delay = av_q2d(movie->video_st->codec->time_base);
    /* for MPEG2, the frame can be repeated, so we update the
       clock accordingly */
    frame_delay += frame->repeat_pict * (frame_delay * 0.5);
    movie->video_current_pts += frame_delay;

    movie->pts = pts;
}

/* get the current audio clock value */
double get_audio_clock(PyMovie *movie)
{
    return getAudioClock();
}

/* get the current video clock value */
double get_video_clock(PyMovie *movie)
{
    double delta;

    if (movie->paused)
    {
        delta = 0;
    }
    else
    {
        delta = (av_gettime() - movie->video_current_pts_time) / 1000000.0;
    }
    double temp = movie->video_current_pts+delta;
    return temp;
}

/* get the current external clock value */
double get_external_clock(PyMovie *movie)
{
    int64_t ti;
    ti = av_gettime();
    double res = movie->external_clock + ((ti - movie->external_clock_time) * 1e-6);
    return res;
}

/* get the current master clock value */
double get_master_clock(PyMovie *movie)
{
    double val;

    if (movie->av_sync_type == AV_SYNC_VIDEO_MASTER)
    {
        if (movie->video_st)
            val = get_video_clock(movie);
        else
            val = get_audio_clock(movie);
    }
    else if (movie->av_sync_type == AV_SYNC_AUDIO_MASTER)
    {
        if (movie->audio_st)
            val = get_audio_clock(movie);
        else
            val = get_video_clock(movie);
    }
    else
    {
        val = get_external_clock(movie);
    }
    return val;
}

void registerCommands(PyMovie *self)
{
    self->seekCommandType=registerCommand(self->commands);
    self->pauseCommandType=registerCommand(self->commands);
    self->stopCommandType=registerCommand(self->commands);
    self->resizeCommandType=registerCommand(self->commands);
    self->shiftCommandType = registerCommand(self->commands);
    self->surfaceCommandType = registerCommand(self->commands);
}

/* seek in the stream */
void stream_seek(PyMovie *movie, int64_t pos, int rel)
{
    ALLOC_COMMAND(seekCommand, seek)
    seek->pos = pos;
    seek->rel = rel;
    addCommand(movie->commands, (Command *)seek);
}

/* pause or resume the video */
void stream_pause(PyMovie *movie)
{
    ALLOC_COMMAND(pauseCommand, pause);
    addCommand(movie->commands, (Command *)pause);
}

int audio_thread(void *arg)
{
    PyMovie *movie = arg;
    DECLAREGIL
    GRABGIL
    Py_INCREF(movie);
    RELEASEGIL
    AVPacket *pkt = &movie->audio_pkt;
    AVCodecContext *dec= movie->audio_st->codec;
    int len1, data_size;
    int filled =0;
    len1=0;
    for(;;)
    {
        if(movie->stop || movie->audioq.abort_request)
        {
            pauseBuffer(movie->channel);
            stopBuffer(movie->channel);
            goto closing;
        }
        if(movie->paused!=movie->audio_paused)
        {
            pauseBuffer(movie->channel);
            movie->audio_paused=movie->paused;
            if(movie->audio_paused)
            {
                movie->working=0;
                continue;
            }
        }

        if(movie->paused)
        {
            SDL_Delay(10);
            continue;
        }
        //check if the movie has ended

        if(getBufferQueueSize()>20)
        {
            SDL_Delay(100);
            continue;
        }

        //fill up the buffer
        while(movie->audio_pkt_size > 0)
        {
            data_size = sizeof(movie->audio_buf1);
            len1 += avcodec_decode_audio2(dec, (int16_t *)movie->audio_buf1, &data_size, movie->audio_pkt_data, movie->audio_pkt_size);
            if (len1 < 0)
            {
                /* if error, we skip the frame */
                movie->audio_pkt_size = 0;
                break;
            }
            movie->audio_pkt_data += len1;
            movie->audio_pkt_size -= len1;
            if (data_size <= 0)
                continue;
            //reformat_ctx here, but deleted
            filled=1;

        }
        if(filled)
        {
            /* Buffer is filled up with a new frame, we spin lock/wait for a signal, where we then call playBuffer */
            int chan = playBuffer(movie->audio_buf1, data_size, movie->channel, movie->audio_pts);
            if(chan==-1)
            {
                GRABGIL
                char *s = Mix_GetError();
                PySys_WriteStdout("%s\n", s);
                RELEASEGIL
            }
            movie->channel = chan;
            filled=0;
            len1=0;
        }

        //either buffer filled or no packets yet
        /* free the current packet */
        if (pkt->data)
            av_free_packet(pkt);

        /* read next packet */
        if (packet_queue_get(&movie->audioq, pkt, 1) <= 0)
        {
            SDL_Delay(10);
            continue;
        }

        if(pkt->data == flush_pkt.data)
        {
            avcodec_flush_buffers(dec);
            SDL_Delay(10);
            continue;
        }

        movie->audio_pts      = pkt->pts;
        movie->audio_pkt_data = pkt->data;
        movie->audio_pkt_size = pkt->size;

    }
closing:
    GRABGIL
    Py_DECREF(movie);
    RELEASEGIL
    return 0;
}

/* open a given stream. Return 0 if OK */
int stream_component_open(PyMovie *movie, int stream_index, int threaded)
{
    DECLAREGIL
    if(threaded)
    {
        GRABGIL
    }
    Py_INCREF( movie);
    if(threaded)
    {
        RELEASEGIL
    }

    AVFormatContext *ic = movie->ic;
    AVCodecContext *enc;
    if (stream_index < 0 || stream_index >= ic->nb_streams)
    {
        if(threaded)
            {GRABGIL}
        Py_DECREF(movie);
        if(threaded)
            {RELEASEGIL}
        return -1;
    }

    initialize_codec(movie, stream_index, threaded);

    enc = ic->streams[stream_index]->codec;
    switch(enc->codec_type)
    {
    case CODEC_TYPE_AUDIO:
        movie->audio_stream = stream_index;
        movie->audio_st = ic->streams[stream_index];
        break;
    case CODEC_TYPE_VIDEO:
        movie->video_stream = stream_index;
        movie->video_st = ic->streams[stream_index];
        break;
#if 0

    case CODEC_TYPE_SUBTITLE:
        movie->sub_stream = stream_index;
        movie->sub_st     = ic->streams[stream_index];
#endif

    default:
        break;
    }
    if(threaded)
    {
        GRABGIL
    }
    Py_DECREF( movie);
    if(threaded)
    {
        RELEASEGIL
    }
    return 0;
}
/* open a given stream. Return 0 if OK */
int stream_component_start(PyMovie *movie, int stream_index, int threaded)
{
    DECLAREGIL
    if(threaded)
    {
        GRABGIL
    }
    Py_INCREF( movie);
    if(threaded)
    {
        RELEASEGIL
    }
    AVFormatContext *ic = movie->ic;
    AVCodecContext *enc;

    if (stream_index < 0 || stream_index >= ic->nb_streams)
    {
        if(threaded)
            GRABGIL
            Py_DECREF(movie);
        if(threaded)
            RELEASEGIL
            return -1;
    }
    initialize_codec(movie, stream_index, threaded);
    enc = ic->streams[stream_index]->codec;
    switch(enc->codec_type)
    {
    case CODEC_TYPE_AUDIO:
        if(movie->replay)
        {
            movie->audio_st = ic->streams[stream_index];
            movie->audio_stream = stream_index;
        }
        memset(&movie->audio_pkt, 0, sizeof(movie->audio_pkt));
        packet_queue_init(&movie->audioq);
        movie->audio_mutex = SDL_CreateMutex();
        soundStart();
        movie->audio_tid = SDL_CreateThread(audio_thread, movie);
        break;
    case CODEC_TYPE_VIDEO:
        if(movie->replay)
        {
            movie->video_stream = stream_index;
            movie->video_st = ic->streams[stream_index];
        }
        movie->frame_last_delay = 40e-3;
        movie->frame_timer = (double)av_gettime() / 1000000.0;
        movie->video_current_pts_time = av_gettime();
        packet_queue_init(&movie->videoq);
        break;
#if 0

    case CODEC_TYPE_SUBTITLE:
        if(movie->replay)
        {
            movie->sub_stream = stream_index;
            movie->sub_st     = ic->streams[stream_index];
        }
        packet_queue_init(&movie->subq);
#endif

    default:
        break;
    }
    if(threaded)
    {
        GRABGIL
    }
    Py_DECREF( movie);
    if(threaded)
    {
        RELEASEGIL
    }
    return 0;
}

void stream_component_end(PyMovie *movie, int stream_index, int threaded)
{
    DECLAREGIL
    if(threaded)
        GRABGIL
        if(movie->ob_refcnt!=0)
            Py_INCREF( movie);
    if(threaded)
        RELEASEGIL
        AVFormatContext *ic = movie->ic;
    AVCodecContext *enc;

    if (stream_index < 0 || stream_index >= ic->nb_streams)
    {
        if(threaded)
            GRABGIL
            if(movie->ob_refcnt!=0)
            {
                Py_DECREF(movie);
            }
        if(threaded)
            RELEASEGIL
            return;
    }
    movie->replay=1;
    enc = ic->streams[stream_index]->codec;
    int i;
    VidPicture *vp;
    //SubPicture *sp;
    switch(enc->codec_type)
    {
    case CODEC_TYPE_AUDIO:
        packet_queue_abort(&movie->audioq);
        SDL_WaitThread(movie->audio_tid, NULL);
        SDL_DestroyMutex(movie->audio_mutex);
        soundEnd();
        memset(&movie->audio_buf1, 0, sizeof(movie->audio_buf1));
        packet_queue_flush(&movie->audioq);
        break;
    case CODEC_TYPE_VIDEO:
        for(i=0;i<VIDEO_PICTURE_QUEUE_SIZE;i++)
        {
            vp = &movie->pictq[i];
            vp->ready=0;
        }
        movie->video_current_pts=0;
        packet_queue_abort(&movie->videoq);
        packet_queue_flush(&movie->videoq);
        break;
#if 0

    case CODEC_TYPE_SUBTITLE:
        packet_queue_abort(&movie->subq);
        packet_queue_flush(&movie->subq);
        break;
#endif

    default:
        break;
    }
    ic->streams[stream_index]->discard = AVDISCARD_ALL;

    if(threaded)
        GRABGIL
        if(movie->ob_refcnt!=0)
        {
            Py_DECREF( movie);
        }
    if(threaded)
        RELEASEGIL
    }
void stream_component_close(PyMovie *movie, int stream_index, int threaded)
{
    DECLAREGIL
    if(threaded)
        GRABGIL
        if(movie->ob_refcnt!=0)
            Py_INCREF( movie);
    if(threaded)
        RELEASEGIL
        AVFormatContext *ic = movie->ic;
    AVCodecContext *enc;

    if (stream_index < 0 || stream_index >= ic->nb_streams)
    {
        if(threaded)
            GRABGIL
            if(movie->ob_refcnt!=0)
            {
                Py_DECREF(movie);
            }
        if(threaded)
            RELEASEGIL
            return;
    }
    enc = ic->streams[stream_index]->codec;
    int end = movie->loops;
    switch(enc->codec_type)
    {
    case CODEC_TYPE_AUDIO:
        soundQuit();
        packet_queue_end(&movie->audioq, end);
        break;
    case CODEC_TYPE_VIDEO:
        packet_queue_end(&movie->videoq, end);
        break;
    default:
        break;
    }

    ic->streams[stream_index]->discard = AVDISCARD_ALL;
    avcodec_close(enc);
    switch(enc->codec_type)
    {
    case CODEC_TYPE_AUDIO:
        movie->audio_st = NULL;
        movie->audio_stream = -1;
        break;
    case CODEC_TYPE_VIDEO:
        movie->video_st = NULL;
        movie->video_stream = -1;
        break;
#if 0

    case CODEC_TYPE_SUBTITLE:
        movie->sub_st=NULL;
        movie->sub_stream = -1;
#endif

    default:
        break;
    }

    if(threaded)
        GRABGIL
        if(movie->ob_refcnt!=0)
        {
            Py_DECREF( movie);
        }
    if(threaded)
        RELEASEGIL
    }

void stream_open(PyMovie *movie, const char *filename, AVInputFormat *iformat, int threaded)
{
    if (!movie)
        return;
    DECLAREGIL
    if(threaded)
    {
        GRABGIL
    }
    Py_INCREF(movie);
    if(threaded)
    {
        RELEASEGIL
    }
    int  i, ret, video_index, audio_index, subtitle_index;

    strncpy(movie->filename, filename, strlen(filename)+1);
    movie->iformat = iformat;

    /* start video dmovieplay */
    movie->dest_mutex = SDL_CreateMutex();

    //in case we've called stream open once before...
    movie->abort_request = 0;
    movie->av_sync_type = AV_SYNC_AUDIO_MASTER;

    video_index = -1;
    audio_index = -1;
    subtitle_index = -1;
    movie->video_stream = -1;
    movie->audio_stream = -1;
    //movie->sub_stream = -1;

    initialize_context(movie, threaded); //moved a bunch of convenience stuff out of here for access at other times

    int wanted_video_stream=1;
    int wanted_audio_stream=1;
    int wanted_subti_stream=-1;
    /* if seeking requested, we execute it */
    if (movie->start_time != AV_NOPTS_VALUE)
    {
        int64_t timestamp;

        timestamp = movie->start_time;
        /* add the stream start time */
        if (movie->ic->start_time != AV_NOPTS_VALUE)
            timestamp += movie->ic->start_time;
        ret = av_seek_frame(movie->ic, -1, timestamp, AVSEEK_FLAG_BACKWARD);
        if (ret < 0)
        {
            if(threaded)
                GRABGIL
                PyErr_Format(PyExc_IOError, "%s: could not seek to position %0.3f", movie->filename, (double)timestamp/AV_TIME_BASE);
            if(threaded)
                RELEASEGIL
            }
    }
    for(i = 0; i < movie->ic->nb_streams; i++)
    {
        AVCodecContext *enc = movie->ic->streams[i]->codec;
        movie->ic->streams[i]->discard = AVDISCARD_ALL;
        switch(enc->codec_type)
        {
        case CODEC_TYPE_AUDIO:
            if (wanted_audio_stream-- >= 0 && !movie->audio_disable)
                audio_index = i;
            break;
        case CODEC_TYPE_VIDEO:
            if (wanted_video_stream-- >= 0 && !movie->video_disable)
                video_index = i;
            break;
        case CODEC_TYPE_SUBTITLE:
            //if(wanted_subti_stream -- >= 0 && !movie->subtitle_disable)
            //	subtitle_index=i;
        default:
            break;
        }
    }

    /* open the streams */
    if (audio_index >= 0)
    {
        stream_component_open(movie, audio_index, threaded);
    }

    if (video_index >= 0)
    {
        stream_component_open(movie, video_index, threaded);
    }
    if(subtitle_index >= 0)
    {
        stream_component_open(movie, subtitle_index, threaded);
    }

    if (movie->video_stream < 0 && movie->audio_stream < 0)
    {
        if(threaded)
        {
            GRABGIL
        }
        PyErr_Format(PyExc_IOError, "%s: could not open codecs", movie->filename);
        if(threaded)
        {
            RELEASEGIL
        }
        ret = -1;
        goto fail;
    }
    movie->frame_delay = av_q2d(movie->video_st->codec->time_base);

    ret = 0;
fail:
    if(ret!=0)
    {
        if(threaded)
        {
            GRABGIL
        }
        PyObject *er;
        er=PyErr_Occurred();
        if(er)
        {
            PyErr_Print();
        }
        Py_DECREF(movie);
        if(threaded)
        {
            RELEASEGIL
        }
        return;
    }
    if(threaded)
    {
        GRABGIL
    }
    Py_DECREF(movie);
    if(threaded)
    {
        RELEASEGIL
    }
    return;
}


int initialize_context(PyMovie *movie, int threaded)
{
    DECLAREGIL
    AVFormatContext *ic;
    AVFormatParameters params, *ap = &params;
    int ret, err;

    memset(ap, 0, sizeof(*ap));
    ap->width = 0;
    ap->height= 0;
    ap->time_base= (AVRational)
                   {
                       1, 25
                   };
    ap->pix_fmt = PIX_FMT_NONE;

    if (movie->ic)
    {
        av_close_input_file(movie->ic);
        movie->ic = NULL; /* safety */
    }
    movie->iformat = NULL;
    err = av_open_input_file(&ic, movie->filename, movie->iformat, 0, ap);
    if (err < 0)
    {
        if(threaded)
            GRABGIL
            PyErr_Format(PyExc_IOError, "There was a problem opening up %s, due to %i", movie->filename, err);
        if(threaded)
            RELEASEGIL
            ret = -1;
        goto fail;
    }
    err = av_find_stream_info(ic);
    if (err < 0)
    {
        if(threaded)
            GRABGIL
            PyErr_Format(PyExc_IOError, "%s: could not find codec parameters", movie->filename);
        if(threaded)
            RELEASEGIL
            ret = -1;
        goto fail;
    }
    if(ic->pb)
        ic->pb->eof_reached= 0; //FIXME hack, ffplay maybe should not use url_feof() to test for the end


    movie->ic = ic;
    ret=0;
fail:
    return ret;

}

int initialize_codec(PyMovie *movie, int stream_index, int threaded)
{
    DECLAREGIL
    if(threaded)
    {
        GRABGIL
    }
    Py_INCREF(movie);
    if(threaded)
    {
        RELEASEGIL
    }
    AVFormatContext *ic = movie->ic;
    AVCodecContext *enc;
    AVCodec *codec;
    int freq, channels;
    enc = ic->streams[stream_index]->codec;
    /* prepare audio output */
    if (enc->codec_type == CODEC_TYPE_AUDIO)
    {
        if (enc->channels > 0)
        {
            enc->request_channels = FFMIN(2, enc->channels);
        }
        else
        {
            enc->request_channels = 2;
        }
    }
    codec = avcodec_find_decoder(enc->codec_id);
    enc->debug_mv = 0;
    enc->debug = 0;
    enc->workaround_bugs = 1;
    enc->lowres = 0;
    enc->idct_algo= FF_IDCT_AUTO;
    if(0)
        enc->flags2 |= CODEC_FLAG2_FAST;
    enc->skip_frame= AVDISCARD_DEFAULT;
    enc->skip_idct= AVDISCARD_DEFAULT;
    enc->skip_loop_filter= AVDISCARD_DEFAULT;
#if LIBAVCODEC_VERSION_INT>=3412992 //(52<<16)+(20<<8)+0 ie 52.20.0
    
        enc->error_recognition= FF_ER_CAREFUL;
#endif
    enc->error_concealment= 3;


    //TODO:proper error reporting here please
    int ret =avcodec_open(enc, codec);
    if(ret < 0)
    {
        if(threaded)
        {
            GRABGIL
        }
        Py_DECREF(movie);
        if(threaded)
        {
            RELEASEGIL
        }
        return -1;
    }
    /* prepare audio output */
    if (enc->codec_type == CODEC_TYPE_AUDIO)
    {

        freq = enc->sample_rate;
        channels = enc->channels;
        if(!movie->replay)
        {
            if (soundInit  (freq, -16, channels, 1024, av_q2d(ic->streams[stream_index]->time_base)) < 0)
            {
                RAISE(PyExc_SDLError, SDL_GetError ());
            }
        }
        movie->audio_src_fmt= AUDIO_S16SYS;
    }

    enc->thread_count= 1;
    ic->streams[stream_index]->discard = AVDISCARD_DEFAULT;
    if(threaded)
    {
        GRABGIL
    }
    Py_DECREF(movie);
    if(threaded)
    {
        RELEASEGIL
    }
    return 0;
}
void stream_close(PyMovie *movie, int threaded)
{
    DECLAREGIL
    if(threaded)
        GRABGIL
        if(movie->ob_refcnt!=0)
            Py_INCREF(movie);
    if(threaded)
        RELEASEGIL
        movie->abort_request = 1;
    SDL_WaitThread(movie->parse_tid, NULL);
    VidPicture *vp;

    if(movie)
    {
        int i;
        for( i =0; i<VIDEO_PICTURE_QUEUE_SIZE; i++)
        {
            vp = &movie->pictq[i];
            if (vp->dest_overlay)
            {
                SDL_FreeYUVOverlay(vp->dest_overlay);
                vp->dest_overlay = NULL;
            }
            if(vp->dest_surface)
            {
                SDL_FreeSurface(vp->dest_surface);
                vp->dest_surface=NULL;
            }

        }
        SDL_DestroyMutex(movie->dest_mutex);
        if(movie->img_convert_ctx)
        {
            sws_freeContext(movie->img_convert_ctx);
            movie->img_convert_ctx=NULL;
        }
    }
    /* close each stream */
    if (movie->audio_stream >= 0)
    {
        stream_component_close(movie, movie->audio_stream, threaded);
    }
    if (movie->video_stream >= 0)
    {
        stream_component_close(movie, movie->video_stream, threaded);
    }
#if 0
    if (movie->sub_stream >= 0)
    {
        stream_component_close(movie, movie->sub_stream, threaded);
    }
#endif
    if (movie->ic)
    {
        av_close_input_file(movie->ic);
        movie->ic = NULL; /* safety */
    }

    if(movie->ob_refcnt!=0)
    {
        if(threaded)
            GRABGIL
            Py_DECREF(movie);
        if(threaded)
            RELEASEGIL
        }
}

void stream_cycle_channel(PyMovie *movie, int codec_type)
{
    AVFormatContext *ic = movie->ic;
    int start_index=0;
    int stream_index;
    AVStream *st;
    DECLAREGIL
    GRABGIL
    Py_INCREF(movie);
    RELEASEGIL

    if (codec_type == CODEC_TYPE_VIDEO)
        start_index = movie->video_stream;
    else if (codec_type == CODEC_TYPE_AUDIO)
        start_index = movie->audio_stream;
    stream_index = start_index;
    for(;;)
    {
        if (++stream_index >= movie->ic->nb_streams)
        {
            stream_index = 0;
        }
        if (stream_index == start_index)
            return;
        st = ic->streams[stream_index];
        if (st->codec->codec_type == codec_type)
        {
            /* check that parameters are OK */
            switch(codec_type)
            {
            case CODEC_TYPE_AUDIO:
                if (st->codec->sample_rate != 0 &&
                        st->codec->channels != 0)
                    goto the_end;
                break;
            case CODEC_TYPE_VIDEO:
            default:
                break;
            }
        }
    }
the_end:
    stream_component_close(movie, start_index, 1);
    stream_component_open(movie, stream_index, 1);
    GRABGIL
    Py_DECREF(movie);
    RELEASEGIL
}

int decoder_wrapper(void *arg)
{
    PyMovie *movie = arg;
    DECLAREGIL
    GRABGIL
    Py_INCREF(movie);
    RELEASEGIL
    int state=0;
    int eternity =0;
    if(movie->loops==-1)
    {
        eternity=1;
    }
    while((movie->loops>-1||eternity) && !movie->stop )
    {
        movie->loops--;
        movie->paused=0;
        //we have to (re)initialize the context when we start over. Most efficient way really.
        if(movie->replay)
            initialize_context(movie, 1);
        /*starting a stream is different from opening it.
         * *nix weenies see that they are the same, when really, they're not.
         * In this case, starting a stream means we set all the values and stuff we need 
         * to play it as if it had just started, like initial time values, etc.
         */

        if(movie->video_st)
            stream_component_start(movie, movie->video_stream, 1);
        if(movie->audio_st)
            stream_component_start(movie, movie->audio_stream, 1);
#if 0

        if(movie->sub_st)
            stream_component_start(movie, movie->sub_stream, 1);
#endif
        /* Now we call the function that does the ACTUAL work. Just like us to loaf
         * around while we make decoder work so hard...
         */
        state =decoder(movie);
        /* We've returned... for any number of reasons, like stopping, we're finished
         * or as an omen of the impending apocalypse. I recommend you use the
         * necronomicon to diagnose this.
         * 
         * And now, we need to end a stream. Again, this is different from closing a stream. 
         * This just sets various variables and structs to their ended state, but they're still
         * ready to be started again. We only close streams when we dealloc the movie. We want 
         * to be able to reuse the memory.  Every memory page we reuse is another
         * electronic tree saved!
         */
        if(movie->video_st)
            stream_component_end(movie, movie->video_st->index, 1);
        if(movie->audio_st)
            stream_component_end(movie, movie->audio_st->index, 1);
#if 0

        if(movie->sub_st)
            stream_component_end(movie, movie->sub_st->index, 1);
#endif

        if(movie->stop)
        {
            VidPicture *vp;

            int i;
            for( i =0; i<VIDEO_PICTURE_QUEUE_SIZE; i++)
            {
                vp = &movie->pictq[i];
                if (vp->dest_overlay)
                {
                    SDL_FreeYUVOverlay(vp->dest_overlay);
                    vp->dest_overlay = NULL;
                }
                if(vp->dest_surface)
                {
                    SDL_FreeSurface(vp->dest_surface);
                    vp->dest_surface=NULL;
                }

            }
            /*if (SDL_WasInit (SDL_INIT_VIDEO))
            	SDL_QuitSubSystem (SDL_INIT_VIDEO);*/
        }
        if(state==-1)
        {
        	if(PyErr_Occurred())
        	{
        		PyErr_Print();
        	}
        	break;
        }
    }
    GRABGIL
    Py_DECREF(movie);
    RELEASEGIL
    movie->playing=0;
    movie->paused=0;
    return state;
}

int decoder(void *arg)
{
    /* This is the most-hardworking function in the entire module. So respect it!
     * He's the blue collar worker amongst the white-collar data pushing functions. There's
     * a few other blue collar functions, but decoder is boss o' them all.
     * 
     * Here's decoder's work schedule:
     * 	loop:
     * 		checks status
     * 		deals with seeking
     * 		handles eofs and stuff
     * 		read frame
     * 		load frame into A/V queue
     * 		video_render()
     * 		//audio_thread()
     * 		first two loops:
     * 			video_refresh_timer <--- we do this or else we'd never start display frames
     * 		if timing AND timing >=now:
     * 			video_display
     * 
     * And thats it! decoder does this till it ends. Then it cleans some stuff up, and exits gracefully for most any situation.
     */
    PyMovie *movie = arg;
    DECLAREGIL
    GRABGIL
#ifdef PROFILE
    if(movie->istats==NULL)
        movie->istats = (ImageScaleStats *)PyMem_Malloc(sizeof(ImageScaleStats));
#endif

    Py_INCREF( movie);
    RELEASEGIL
    AVFormatContext *ic;
    int ret;
    AVPacket pkt1, *pkt = &pkt1;
    movie->stop=0;
    movie->finished =0;
    movie->playing=1;
    ic=movie->ic;
    int co=0;
    int video_packet=0;
	//we do video open as a batch, instead of on-demand. Much better performance that way.
    video_open(movie, 0);
    movie->last_showtime = av_gettime()/1000.0;
    int seeking =0;
    for(;;)
    {
        if(hasCommand(movie->commands) && !movie->working)
        {
            Command *comm = getCommand(movie->commands);

            if(comm->type==movie->seekCommandType)
            {
                seekCommand *seek = (seekCommand *)comm;
                movie->seek_req=1;
                movie->seek_pos = seek->pos;
                movie->seek_flags |= seek->rel;
                /* clear stuff away now */
                comm = NULL;
                PyMem_Free(seek);
                movie->working=1;
            }
            else if(comm->type==movie->pauseCommandType)
            {
                pauseCommand *pause = (pauseCommand *)comm;
                movie->paused = !movie->paused;
                movie->working = movie->paused;
                if (!movie->paused)
                {
                    movie->video_current_pts = get_video_clock(movie);
                    movie->frame_timer += (av_gettime() - movie->video_current_pts_time) / 1000000.0;
                }
                comm=NULL;
                PyMem_Free(pause);
            }
            else if (comm->type ==movie->stopCommandType)
            {
                stopCommand *stop = (stopCommand *)comm;
                movie->stop=1;
                comm=NULL;
                PyMem_Free(stop);
                movie->working=1;
            }
            else if (comm->type == movie->resizeCommandType)
            {
                resizeCommand *resize=(resizeCommand *)comm;
                if (resize->h!=0)
                {
                    movie->resize_h=1;
                    movie->height=resize->h;
                }
                if(resize->w!=0)
                {
                    movie->resize_w=1;
                    movie->width= resize->w;
                }
                comm=NULL;
                PyMem_Free(resize);
                video_open(movie, 0);
            }
            else if (comm->type == movie->shiftCommandType)
            {
                shiftCommand *shift = (shiftCommand *)comm;

                movie->xleft=shift->xleft;
                movie->ytop=shift->ytop;
                comm=NULL;
                PyMem_Free(shift);
                video_open(movie, 0);
            }
            else if(comm->type == movie->surfaceCommandType)
            {
                surfaceCommand *surf = (surfaceCommand *)comm;
                if(movie->canon_surf)
                {
                    SDL_FreeSurface(movie->canon_surf);
                    movie->canon_surf=NULL;
                }
                if(surf->surface)
                {
                    movie->canon_surf=surf->surface;
                    movie->overlay=0;
                }
                else
                {
                    movie->overlay=1;
                }

            }

        }

        if (movie->abort_request)
        {
            break;
        }
        if(movie->stop)
        {
            break;
        }
        if (movie->paused != movie->last_paused)
        {
            movie->last_paused = movie->paused;
            if (movie->paused)
            {
                av_read_pause(ic);
            }
            else
            {
                movie->last_showtime=av_gettime()/1000.0;
                av_read_play(ic);
            }
        }
        if(movie->paused)
        {
            SDL_Delay(10);
            continue;
        }
        if (movie->seek_req)
        {
            movie->working=0;
            int64_t seek_target= movie->seek_pos;
            seek_target-=1*AV_TIME_BASE;
            int aud_stream_index=-1;
            int vid_stream_index=-1;
            int64_t vid_seek_target=seek_target;
            if (movie->video_stream >= 0)
            {
                vid_stream_index= movie->video_stream;
            }
            else if (movie->audio_stream >=0)
            {
                aud_stream_index = movie->audio_stream;
            }
            if(vid_stream_index>=0)
                vid_seek_target= av_rescale_q(seek_target, AV_TIME_BASE_Q, ic->streams[vid_stream_index]->time_base);

            if(vid_stream_index>=0)
            {
                if(vid_seek_target > movie->video_st->duration)
                {
                    vid_seek_target = vid_seek_target%movie->video_st->duration;
                }
                ret = av_seek_frame(movie->ic, vid_stream_index, vid_seek_target, movie->seek_flags);

                if (ret < 0)
                {
                    PyErr_Format(PyExc_IOError, "%s: error while seeking", movie->ic->filename);
                }
            }
            if (movie->video_stream >= 0)
            {
                packet_queue_flush(&movie->videoq);
                packet_queue_put(&movie->videoq, &flush_pkt);
            }
            if (movie->audio_stream >= 0)
            {
                packet_queue_flush(&movie->audioq);
                packet_queue_put(&movie->audioq, &flush_pkt);
                seeking=1;

            }

            movie->seek_req = 0;
            //now we need to seek to a keyframe...
            if(ic->streams[vid_stream_index]->duration>0)
            {
                while(av_read_frame(ic, pkt)>=0)
                {
                    if(pkt->stream_index == movie->video_stream)
                    {
                        int bytesDecoded, frameFinished;
                        AVFrame *frame;
                        frame=avcodec_alloc_frame();
                        bytesDecoded = avcodec_decode_video(movie->video_st->codec, frame, &frameFinished, pkt->data, pkt->size);
                        if(frameFinished)
                        {
                            if((pkt->pts >= vid_seek_target) || (pkt->dts >= vid_seek_target))
                            {
                                av_free(frame);
                                break;
                            }
                        }
                        av_free(frame);
                    }
                    av_free_packet(pkt);
                }
                av_free_packet(pkt);
            }

        }
        /* if the queue are full, no need to read more */
        if ( //yay for short circuit logic testing
            (movie->videoq.size > MAX_VIDEOQ_SIZE ))
        {
            /* wait 10 ms */
            if(!movie->paused)
            {
                //we only forcefully pull a packet of the queues when the queue is too large and the video file is not paused.
                // The simple reason is that video_render and audio_thread do nothing, so its a waste of cycles to call them otherwise
                if(movie->videoq.size > MAX_VIDEOQ_SIZE && movie->video_st)
                {
                    video_render(movie);
                }
            }
            continue;
        }

        if(url_feof(ic->pb))
        {
            av_init_packet(pkt);
            pkt->data=NULL;
            pkt->size=0;
            pkt->stream_index= movie->video_stream;
            packet_queue_put(&movie->videoq, pkt);
            continue;
        }
        if(movie->pictq_size<VIDEO_PICTURE_QUEUE_SIZE)
        {
            ret = av_read_frame(ic, pkt);
            if (ret < 0)
            {
#if LIBAVCODEC_VERSION_INT>=3412992
				if (ret != AVERROR_EOF && url_ferror(ic->pb) == 0)
                {
                    goto fail;
                }
                else
                {
                    break;
                }
#else
				goto fail;
#endif
				}
            if (pkt->stream_index == movie->audio_stream)
            {
                if(seeking)
                {
                    seekBuffer(pkt->pts);
                    seeking=0;
                }
                packet_queue_put(&movie->audioq, pkt);

            }
            else if (pkt->stream_index == movie->video_stream)
            {
                packet_queue_put(&movie->videoq, pkt);
                video_packet=1;
            }
#if 0
            else if (pkt->stream_index == movie->sub_stream)
            {
                packet_queue_put(&movie->subq, pkt);
            }
#endif
            else if(pkt)
            {
                av_free_packet(pkt);
            }
        }
#if 0
        SubPicture *sp;
        SubPicture *sp2;
        if(movie->sub_stream>=0)
        {
            if (movie->sub_stream_changed)
            {
                SDL_LockMutex(movie->subpq_mutex);

                while (movie->subpq_size)
                {
                    free_subpicture(&movie->subpq[movie->subpq_rindex]);

                    // update queue size and signal for next picture
                    if (++movie->subpq_rindex == SUBPICTURE_QUEUE_SIZE)
                        movie->subpq_rindex = 0;

                    movie->subpq_size--;
                }
                movie->sub_stream_changed = 0;

                SDL_UnlockMutex(movie->subpq_mutex);
            }
            else
            {
                if (movie->subpq_size > 0)
                {
                    sp = &movie->subpq[movie->subpq_rindex];

                    if (movie->subpq_size > 1)
                        sp2 = &movie->subpq[(movie->subpq_rindex + 1) % SUBPICTURE_QUEUE_SIZE];
                    else
                        sp2 = NULL;

                    if ((movie->video_current_pts > (sp->pts + ((float) sp->sub.end_display_time / 1000)))
                            || (sp2 && movie->video_current_pts > (sp2->pts + ((float) sp2->sub.start_display_time / 1000))))
                    {
                        free_subpicture(sp);

                        // update queue size and signal for next picture
                        if (++movie->subpq_rindex == SUBPICTURE_QUEUE_SIZE)
                            movie->subpq_rindex = 0;

                        SDL_LockMutex(movie->subpq_mutex);
                        movie->subpq_size--;
                        SDL_UnlockMutex(movie->subpq_mutex);
                    }
                }
            }
        }
#endif

        if(movie->video_st && video_packet)
        {
            video_render(movie);
            video_packet=0;
        }
		/*This is very important: without this if check, 
		 * The video frames will not be displayed, ever. */
        if(co<1)
            movie->timing=40;
        co++;

        if(movie->timing>0)
        {
			/* Here, we check if the current time in milliseconds exceeds the scheduled time 
			 * and if so, we display the frame. */ 
            double showtime = movie->timing+movie->last_showtime;
            double now = av_gettime()/1000.0;
            if(now >= showtime)
            {
                double temp = movie->timing;
                double temp_showtime = movie->last_showtime;
                movie->timing =0;

                if(!video_display(movie))
                {
                    //we do this because we haven't shown a frame yet, so we need to preserve the timings, etc.
                    // other we end up with slowdowns and speedups. Not good!
                    movie->timing=temp;
                    movie->last_showtime=temp_showtime;
                }
                else
                {
                    movie->last_showtime = av_gettime()/1000.0;
                }
            }
        }

    }

    ret = 0;
fail:

    if(ret!=0)
    {
        GRABGIL
        RAISE(PyExc_Exception, "Something went wrong, so I'm throwing this error.");
        RELEASEGIL
        movie->abort_request=1;
    }
    movie->pictq_size=movie->pictq_rindex=movie->pictq_windex=0;
    packet_queue_flush(&movie->videoq);
    movie->finished=1;
    movie->working=0;
    GRABGIL
#ifdef PROFILE
    //mean
    int64_t sum =0;
    int64_t max = 0;
    int64_t min = 2<<24;
    TimeSampleNode *cur = movie->istats->first;
    while(cur!=NULL)
    {
        sum+= cur->sample;
        if(cur->sample>max)
            max=cur->sample;
        if(cur->sample<min)
            min=cur->sample;
        if(cur->next==NULL)
            break;
        cur=cur->next;
    }
    movie->istats->mean = (double)sum/movie->istats->n_samples;
    movie->istats->max = max;
    movie->istats->min = min;
    double total = 0;
    double mean = movie->istats->mean;
    cur = movie->istats->first;
    while(cur->next!=NULL)
    {
        total+= pow(((double)cur->sample-mean), (double)2);
        cur=cur->next;
    }
    movie->istats->stdev = sqrt(total/mean);
    PySys_WriteStdout("# Samples: %i\nMean: %f\nMin: %i\nMax: %i\nStDev: %f\n", (int)movie->istats->n_samples, mean, (int)min, (int)max, movie->istats->stdev);
    cur = movie->istats->first;
    TimeSampleNode *prev;
    while(cur->next!=NULL)
    {
        prev=cur;
        cur=cur->next;
        PyMem_Free(prev);
    }
    movie->istats->first=NULL;
    movie->istats->last=NULL;
    movie->istats->n_samples=0;
#endif

    Py_DECREF( movie);
    RELEASEGIL
    movie->playing=0;
    if(movie->abort_request)
    {
        return -1;
    }
    return 0;
}

int video_render(PyMovie *movie)
{
	/* Video render function. Only executed when there is a packet to decode. 
	 *  Here, we get the packet, check it, decode the packet, and queue it up.
	 */
    AVPacket pkt1, *pkt = &pkt1;
    int len1, got_picture;
    AVFrame *frame= avcodec_alloc_frame();
    double pts;

    do
    {

        if (movie->paused && !movie->videoq.abort_request)
        {
            break;
        }
        if (packet_queue_get(&movie->videoq, pkt, 0) <=0)
            break;

        if(pkt->data == flush_pkt.data)
        {
            avcodec_flush_buffers(movie->video_st->codec);
            continue;
        }

        /* NOTE: ipts is the PTS of the _first_ picture beginning in
           this packet, if any */
        int64_t opaque;
#if LIBAVCODEC_VERSION_INT<3412992 //(52<<16)+(20<<8)+0 ie 52.20.0
            opaque=pkt->pts;
#endif
        movie->video_st->codec->reordered_opaque= pkt->pts;
        len1 = avcodec_decode_video(movie->video_st->codec,
                                    frame, &got_picture,
                                    pkt->data, pkt->size);
#if LIBAVCODEC_VERSION_INT<3412992
		if((pkt->dts == AV_NOPTS_VALUE)) //(52<<16)+(20<<8)+0 ie 52.20.0
        {
            //due to short circuiting this checks first, then if that fails it does the invalid old checks. :)
            pts=opaque;
        }
#else
        if(( pkt->dts == AV_NOPTS_VALUE) && (frame->reordered_opaque != AV_NOPTS_VALUE))
        {
            pts= frame->reordered_opaque;
        }
#endif
        else if(pkt->dts != AV_NOPTS_VALUE)
        {
            pts= pkt->dts;
        }
        else
        {
            pts= 0;
        }
        pts *= av_q2d(movie->video_st->time_base);

        if (got_picture)
        {
            update_video_clock(movie, frame, pts);
            if (queue_picture(movie, frame) < 0)
            {
                goto the_end;
            }
        }
        av_free_packet(pkt);
        av_free(frame);
    }
    while(0);

the_end:
    return 0;
}
#if 0
int subtitle_render(void *arg)
{
    PyMovie *movie = arg;
    DECLAREGIL
    GRABGIL
    Py_INCREF(movie);
    RELEASEGIL
    SubPicture *sp;
    AVPacket pkt1, *pkt = &pkt1;
    int len1, got_subtitle;
    double pts;
    int i, j;
    int r, g, b, y, u, v, a;
    int co;

    for(co=0;co<2;co++)
    {
        if (movie->paused && !movie->subq.abort_request)
        {
            SDL_Delay(10);
            goto the_end;
        }
        if (packet_queue_get(&movie->subq, pkt, 1) < 0)
            break;

        if(pkt->data == flush_pkt.data)
        {
            avcodec_flush_buffers(movie->sub_st->codec);
            goto the_end;
        }
        SDL_LockMutex(movie->subpq_mutex);
        if (movie->subpq_size >= SUBPICTURE_QUEUE_SIZE &&
                !movie->subq.abort_request)
        {
            SDL_UnlockMutex(movie->subpq_mutex);
            goto the_end;
        }
        SDL_UnlockMutex(movie->subpq_mutex);

        if (movie->subq.abort_request)
            goto the_end;

        sp = &movie->subpq[movie->subpq_windex];

        /* NOTE: ipts is the PTS of the _first_ picture beginning in
            this packet, if any */
        pts = 0;
        if (pkt->pts != AV_NOPTS_VALUE)
            pts = av_q2d(movie->sub_st->time_base)*pkt->pts;

        len1 = avcodec_decode_subtitle(movie->sub_st->codec,
                                       &sp->sub, &got_subtitle,
                                       pkt->data, pkt->size);
        if (got_subtitle && sp->sub.format == 0)
        {
            sp->pts = pts;

            for (i = 0; i < sp->sub.num_rects; i++)
            {
                for (j = 0; j < sp->sub.rects[i]->nb_colors; j++)
                {
                    RGBA_IN(r, g, b, a, (uint32_t*)sp->sub.rects[i]->pict.data[1] + j);
                    y = RGB_TO_Y_CCIR(r, g, b);
                    u = RGB_TO_U_CCIR(r, g, b, 0);
                    v = RGB_TO_V_CCIR(r, g, b, 0);
                    YUVA_OUT((uint32_t*)sp->sub.rects[i]->pict.data[1] + j, y, u, v, a);
                }
            }

            /* now we can update the picture count */
            if (++movie->subpq_windex == SUBPICTURE_QUEUE_SIZE)
                movie->subpq_windex = 0;
            SDL_LockMutex(movie->subpq_mutex);
            movie->subpq_size++;
            SDL_UnlockMutex(movie->subpq_mutex);
        }
        av_free_packet(pkt);
    }
the_end:
    GRABGIL
    Py_DECREF(movie);
    RELEASEGIL
    return 0;
}
#endif
