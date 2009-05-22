/* Tyler Laing
 * May 16, 2009
 * ff_movie.c
 * Wrapper around ffmpeg libraries for use with pygame/python
 * Imported by movie.py(hopefully)
 */

#include "ff_movie.h"

#ifdef __MINGW32__
#undef main /* We don't want SDL to override our main() */
#endif

#undef exit

//#define DEBUG_SYNC
/* options specified by the user */
static AVInputFormat *file_iformat;
static const char *input_filename;
static int frame_width = 0;
static int frame_height = 0;
static enum PixelFormat frame_pix_fmt = PIX_FMT_NONE;
static int audio_disable;
static int video_disable;
static int wanted_audio_stream= 1;
static int wanted_video_stream= 1;
static int wanted_subtitle_stream= 0;
static int seek_by_bytes;
static int display_disable;
static int show_status;
static int av_sync_type = AV_SYNC_AUDIO_MASTER;
static int64_t start_time = AV_NOPTS_VALUE;
static int debug = 0;
static int debug_mv = 0;
static int step = 0;
static int thread_count = 1;
static int workaround_bugs = 1;
static int fast = 0;
static int genpts = 0;
static int lowres = 0;
static int idct = FF_IDCT_AUTO;
static enum AVDiscard skip_frame= AVDISCARD_DEFAULT;
static enum AVDiscard skip_idct= AVDISCARD_DEFAULT;
static enum AVDiscard skip_loop_filter= AVDISCARD_DEFAULT;
static int error_recognition = FF_ER_CAREFUL;
static int error_concealment = 3;
static int decoder_reorder_pts= 0;

/* current context */
static int64_t audio_callback_time;

static AVPacket flush_pkt;

#define FF_ALLOC_EVENT   (SDL_USEREVENT)
#define FF_REFRESH_EVENT (SDL_USEREVENT + 1)
#define FF_QUIT_EVENT    (SDL_USEREVENT + 2)

static PyAudioStream* _new_audio_stream();
static PyVideoStream* _new_video_stream();
static PySubtitleStream* _new_sub_stream();

//static int sws_flags = SWS_BICUBIC;


/*internal functions for video playing. Not accessible to Python */


/* packet queue handling */
static void packet_queue_init(PacketQueue *q)
{
    memset(q, 0, sizeof(PacketQueue));
    q->mutex = SDL_CreateMutex();
    q->cond = SDL_CreateCond();
}

static void packet_queue_flush(PacketQueue *q)
{
    AVPacketList *pkt, *pkt1;

    SDL_LockMutex(q->mutex);
    for(pkt = q->first_pkt; pkt != NULL; pkt = pkt1) {
        pkt1 = pkt->next;
        av_free_packet(&pkt->pkt);
        av_freep(&pkt);
    }
    q->last_pkt = NULL;
    q->first_pkt = NULL;
    q->nb_packets = 0;
    q->size = 0;
    SDL_UnlockMutex(q->mutex);
}

static void packet_queue_end(PacketQueue *q)
{
    packet_queue_flush(q);
    SDL_DestroyMutex(q->mutex);
    SDL_DestroyCond(q->cond);
}

static int packet_queue_put(PacketQueue *q, AVPacket *pkt)
{
    AVPacketList *pkt1;

    /* duplicate the packet */
    if (pkt!=&flush_pkt && av_dup_packet(pkt) < 0)
        return -1;

    pkt1 = av_malloc(sizeof(AVPacketList));
    if (!pkt1)
        return -1;
    pkt1->pkt = *pkt;
    pkt1->next = NULL;


    SDL_LockMutex(q->mutex);

    if (!q->last_pkt)

        q->first_pkt = pkt1;
    else
        q->last_pkt->next = pkt1;
    q->last_pkt = pkt1;
    q->nb_packets++;
    q->size += pkt1->pkt.size + sizeof(*pkt1);
    /* XXX: should duplicate packet data in DV case */
    SDL_CondSignal(q->cond);

    SDL_UnlockMutex(q->mutex);
    return 0;
}

static void packet_queue_abort(PacketQueue *q)
{
    SDL_LockMutex(q->mutex);

    q->abort_request = 1;

    SDL_CondSignal(q->cond);

    SDL_UnlockMutex(q->mutex);
}

/* return < 0 if aborted, 0 if no packet and > 0 if packet.  */
static int packet_queue_get(PacketQueue *q, AVPacket *pkt, int block)
{
    AVPacketList *pkt1;
    int ret;

    SDL_LockMutex(q->mutex);

    for(;;) {
        if (q->abort_request) {
            ret = -1;
            break;
        }

        pkt1 = q->first_pkt;
        if (pkt1) {
            q->first_pkt = pkt1->next;
            if (!q->first_pkt)
                q->last_pkt = NULL;
            q->nb_packets--;
            q->size -= pkt1->pkt.size + sizeof(*pkt1);
            *pkt = pkt1->pkt;
            av_free(pkt1);
            ret = 1;
            break;
        } else if (!block) {
            ret = 0;
            break;
        } else {
            SDL_CondWait(q->cond, q->mutex);
        }
    }
    SDL_UnlockMutex(q->mutex);
    return ret;
}

#define BPP 1

static void blend_subrect(AVPicture *dst, const AVSubtitleRect *rect, int imgw, int imgh)
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

    if (dsty & 1) {
        lum += dstx;
        cb += skip2;
        cr += skip2;

        if (dstx & 1) {
            YUVA_IN(y, u, v, a, p, pal);
            lum[0] = ALPHA_BLEND(a, lum[0], y, 0);
            cb[0] = ALPHA_BLEND(a >> 2, cb[0], u, 0);
            cr[0] = ALPHA_BLEND(a >> 2, cr[0], v, 0);
            cb++;
            cr++;
            lum++;
            p += BPP;
        }
        for(w = dstw - (dstx & 1); w >= 2; w -= 2) {
            YUVA_IN(y, u, v, a, p, pal);
            u1 = u;
            v1 = v;
            a1 = a;
            lum[0] = ALPHA_BLEND(a, lum[0], y, 0);

            YUVA_IN(y, u, v, a, p + BPP, pal);
            u1 += u;
            v1 += v;
            a1 += a;
            lum[1] = ALPHA_BLEND(a, lum[1], y, 0);
            cb[0] = ALPHA_BLEND(a1 >> 2, cb[0], u1, 1);
            cr[0] = ALPHA_BLEND(a1 >> 2, cr[0], v1, 1);
            cb++;
            cr++;
            p += 2 * BPP;
            lum += 2;
        }
        if (w) {
            YUVA_IN(y, u, v, a, p, pal);
            lum[0] = ALPHA_BLEND(a, lum[0], y, 0);
            cb[0] = ALPHA_BLEND(a >> 2, cb[0], u, 0);
            cr[0] = ALPHA_BLEND(a >> 2, cr[0], v, 0);
            p++;
            lum++;
        }
        p += wrap3 - dstw * BPP;
        lum += wrap - dstw - dstx;
        cb += dst->linesize[1] - width2 - skip2;
        cr += dst->linesize[2] - width2 - skip2;
    }
    for(h = dsth - (dsty & 1); h >= 2; h -= 2) {
        lum += dstx;
        cb += skip2;
        cr += skip2;

        if (dstx & 1) {
            YUVA_IN(y, u, v, a, p, pal);
            u1 = u;
            v1 = v;
            a1 = a;
            lum[0] = ALPHA_BLEND(a, lum[0], y, 0);
            p += wrap3;
            lum += wrap;
            YUVA_IN(y, u, v, a, p, pal);
            u1 += u;
            v1 += v;
            a1 += a;
            lum[0] = ALPHA_BLEND(a, lum[0], y, 0);
            cb[0] = ALPHA_BLEND(a1 >> 2, cb[0], u1, 1);
            cr[0] = ALPHA_BLEND(a1 >> 2, cr[0], v1, 1);
            cb++;
            cr++;
            p += -wrap3 + BPP;
            lum += -wrap + 1;
        }
        for(w = dstw - (dstx & 1); w >= 2; w -= 2) {
            YUVA_IN(y, u, v, a, p, pal);
            u1 = u;
            v1 = v;
            a1 = a;
            lum[0] = ALPHA_BLEND(a, lum[0], y, 0);

            YUVA_IN(y, u, v, a, p + BPP, pal);
            u1 += u;
            v1 += v;
            a1 += a;
            lum[1] = ALPHA_BLEND(a, lum[1], y, 0);
            p += wrap3;
            lum += wrap;

            YUVA_IN(y, u, v, a, p, pal);
            u1 += u;
            v1 += v;
            a1 += a;
            lum[0] = ALPHA_BLEND(a, lum[0], y, 0);

            YUVA_IN(y, u, v, a, p + BPP, pal);
            u1 += u;
            v1 += v;
            a1 += a;
            lum[1] = ALPHA_BLEND(a, lum[1], y, 0);

            cb[0] = ALPHA_BLEND(a1 >> 2, cb[0], u1, 2);
            cr[0] = ALPHA_BLEND(a1 >> 2, cr[0], v1, 2);

            cb++;
            cr++;
            p += -wrap3 + 2 * BPP;
            lum += -wrap + 2;
        }
        if (w) {
            YUVA_IN(y, u, v, a, p, pal);
            u1 = u;
            v1 = v;
            a1 = a;
            lum[0] = ALPHA_BLEND(a, lum[0], y, 0);
            p += wrap3;
            lum += wrap;
            YUVA_IN(y, u, v, a, p, pal);
            u1 += u;
            v1 += v;
            a1 += a;
            lum[0] = ALPHA_BLEND(a, lum[0], y, 0);
            cb[0] = ALPHA_BLEND(a1 >> 2, cb[0], u1, 1);
            cr[0] = ALPHA_BLEND(a1 >> 2, cr[0], v1, 1);
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
    if (h) {
        lum += dstx;
        cb += skip2;
        cr += skip2;

        if (dstx & 1) {
            YUVA_IN(y, u, v, a, p, pal);
            lum[0] = ALPHA_BLEND(a, lum[0], y, 0);
            cb[0] = ALPHA_BLEND(a >> 2, cb[0], u, 0);
            cr[0] = ALPHA_BLEND(a >> 2, cr[0], v, 0);
            cb++;
            cr++;
            lum++;
            p += BPP;
        }
        for(w = dstw - (dstx & 1); w >= 2; w -= 2) {
            YUVA_IN(y, u, v, a, p, pal);
            u1 = u;
            v1 = v;
            a1 = a;
            lum[0] = ALPHA_BLEND(a, lum[0], y, 0);

            YUVA_IN(y, u, v, a, p + BPP, pal);
            u1 += u;
            v1 += v;
            a1 += a;
            lum[1] = ALPHA_BLEND(a, lum[1], y, 0);
            cb[0] = ALPHA_BLEND(a1 >> 2, cb[0], u, 1);
            cr[0] = ALPHA_BLEND(a1 >> 2, cr[0], v, 1);
            cb++;
            cr++;
            p += 2 * BPP;
            lum += 2;
        }
        if (w) {
            YUVA_IN(y, u, v, a, p, pal);
            lum[0] = ALPHA_BLEND(a, lum[0], y, 0);
            cb[0] = ALPHA_BLEND(a >> 2, cb[0], u, 0);
            cr[0] = ALPHA_BLEND(a >> 2, cr[0], v, 0);
        }
    }
}
static void free_subpicture(SubPicture *sp)
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

static void video_image_display(PyMovie *is)
{
    Py_INCREF((PyObject *) is);
    VideoPicture *vp;
    SubPicture *sp;
    AVPicture pict;
    float aspect_ratio;
    int width, height, x, y;
    SDL_Rect rect;
    int i;

    //vp = &is->pictq[is->pictq_rindex];
    PyVideoStream *pvs = (PyVideoStream *)PySequence_GetItem(is->streams, (Py_ssize_t) is->video_stream);
    Py_INCREF((PyObject *)pvs);
    
    vp = &pvs->pictq[pvs->pictq_rindex];
    if (pvs->out_surf || pvs->bmp) {
        /* XXX: use variable in the frame */
        if (pvs->video_st->sample_aspect_ratio.num)
            aspect_ratio = av_q2d(pvs->video_st->sample_aspect_ratio);
        else if (pvs->video_st->codec->sample_aspect_ratio.num)
            aspect_ratio = av_q2d(pvs->video_st->codec->sample_aspect_ratio);
        else
            aspect_ratio = 0;
        if (aspect_ratio <= 0.0)
            aspect_ratio = 1.0;
        aspect_ratio *= (float)pvs->video_st->codec->width / pvs->video_st->codec->height;
        /* if an active format is indicated, then it overrides the
           mpeg format */
#if 0
        if (is->video_st->codec->dtg_active_format != is->dtg_active_format) {
            is->dtg_active_format = is->video_st->codec->dtg_active_format;
            printf("dtg_active_format=%d\n", is->dtg_active_format);
        }
#endif
#if 0
        switch(is->video_st->codec->dtg_active_format) {
        case FF_DTG_AFD_SAME:
        default:
            /* nothing to do */
            break;
        case FF_DTG_AFD_4_3:
            aspect_ratio = 4.0 / 3.0;
            break;
        case FF_DTG_AFD_16_9:
            aspect_ratio = 16.0 / 9.0;
            break;
        case FF_DTG_AFD_14_9:
            aspect_ratio = 14.0 / 9.0;
            break;
        case FF_DTG_AFD_4_3_SP_14_9:
            aspect_ratio = 14.0 / 9.0;
            break;
        case FF_DTG_AFD_16_9_SP_14_9:
            aspect_ratio = 14.0 / 9.0;
            break;
        case FF_DTG_AFD_SP_4_3:
            aspect_ratio = 4.0 / 3.0;
            break;
        }
#endif
        
        if (is->subtitle_stream>-1)
        {
            PySubtitleStream *pss;
            pss = (PySubtitleStream *)PySequence_GetItem(is->streams, (Py_ssize_t) is->subtitle_stream);
            Py_INCREF((PyObject *)pss);
            
            if (pss->subpq_size > 0)
            {
                sp = &pss->subpq[pss->subpq_rindex];

                if (pvs->pts >= pss->pts + ((float) sp->sub.start_display_time / 1000))
                {

                    if(is->overlay>0)
                    {
                        SDL_LockYUVOverlay (pvs->bmp);

                        pict.data[0] = pvs->bmp->pixels[0];
                        pict.data[1] = pvs->bmp->pixels[2];
                        pict.data[2] = pvs->bmp->pixels[1];

                        pict.linesize[0] = pvs->bmp->pitches[0];
                        pict.linesize[1] = pvs->bmp->pitches[2];
                        pict.linesize[2] = pvs->bmp->pitches[1];

                        for (i = 0; i < sp->sub.num_rects; i++)
                            blend_subrect(&pict, sp->sub.rects[i],
                                          pvs->bmp->w, pvs->bmp->h);

                        SDL_UnlockYUVOverlay (pvs->bmp);
                    }
                    else
                    {
                        /*if (pvs->out_surf->flags & SDL_OPENGL && !(pvs->out_surf->flags & (SDL_OPENGLBLIT & ~SDL_OPENGL)))
                                return RAISE (PyExc_SDLError,
                                              "Cannot blit to OPENGL Surfaces (OPENGLBLIT is ok)");*/
                        //TODO:fix blitting to surface, and blend_subrect
                        #if 0
                        SDL_LockSurface(pvs->out_surf);
                        pict.data[0] = (Uint8 *)pvs->out_surf->pixels[0];
                        pict.data[1] = pvs->out_surf->pixels[1];
                        pict.data[2] = pvs->out_surf->pixels[2];

                        pict.linesize[0] = pvs->out_surf->pitch;
                        pict.linesize[1] = pvs->out_surf->pitch;
                        pict.linesize[2] = pvs->out_surf->pitch;

                        for (i = 0; i < sp->sub.num_rects; i++)
                            //TODO:check if blend_subrect works with RGB
                            blend_subrect(&pict, sp->sub.rects[i],
                                          pvs->out_surf->w, pvs->out_surf->h);
                                              
                        SDL_UnlockSurface(pvs->out_surf);
                        #endif
                    }
                }
            }
        Py_DECREF((PyObject *)pss);
        }


        /* XXX: we suppose the screen has a 1.0 pixel ratio */
        height = pvs->height;
        width = ((int)rint(height * aspect_ratio)) & ~1;
        if (width > pvs->width) {
            width = pvs->width;
            height = ((int)rint(width / aspect_ratio)) & ~1;
        }
        x = (pvs->width - width) / 2;
        y = (pvs->height - height) / 2;
       
        rect.x = pvs->xleft + x;
        rect.y = pvs->ytop  + y;
        rect.w = width;
        rect.h = height;
        if(is->overlay>0) 
        {       
            SDL_DisplayYUVOverlay(pvs->bmp, &rect);
        }
        
    } else {
#if 0
        fill_rectangle(screen,
                       is->xleft, is->ytop, is->width, is->height,
                       QERGB(0x00, 0x00, 0x00));
#endif
    }
    Py_DECREF((PyObject *)pvs);
    Py_DECREF((PyObject *) is);
}


static inline int compute_mod(int a, int b)
{
    a = a % b;
    if (a >= 0)
        return a;
    else
        return a + b;
}

static int video_open(PyMovie *is){
    int w,h;
    Py_INCREF((PyObject *) is);
    PyVideoStream *pvs;
    pvs = (PyVideoStream *)PySequence_GetItem(is->streams, (Py_ssize_t) is->video_stream);
    Py_INCREF((PyObject *) pvs);
    
    w = pvs->video_st->codec->width;
    h = pvs->video_st->codec->height;

    if(!pvs->bmp && is->overlay>0)
    {
        //now we have to open an overlay up
        SDL_Surface *screen;
        if (!SDL_WasInit (SDL_INIT_VIDEO))
        return RAISE
            (PyExc_SDLError,
             "cannot create overlay without pygame.display initialized");
        screen = SDL_GetVideoSurface ();
        if (!screen)
            return RAISE (PyExc_SDLError, "Display mode not set");
        pvs->bmp = SDL_CreateYUVOverlay (w, h, SDL_YV12_OVERLAY, screen);
        if (!pvs->bmp)
            return RAISE (PyExc_SDLError, "Cannot create overlay");

    } 
    else if (!pvs->out_surf && is->overlay<=0)
    {
        int flags = SDL_HWSURFACE|SDL_ASYNCBLIT|SDL_HWACCEL;
        //we create a pygame surface
        SDL_Surface *screen;
        #ifndef __APPLE__
        screen = SDL_SetVideoMode(w, h, 0, flags);
        #else
        /* setting bits_per_pixel = 0 or 32 causes blank video on OS X */
        screen = SDL_SetVideoMode(w, h, 24, flags);
        #endif
        pvs->out_surf=(SDL_Surface *)PyMem_Malloc(sizeof(SDL_Surface));
        if (!pvs->out_surf)
            return RAISE (PyExc_SDLError, "Could not create Surface object");
    }


    pvs->width = w;
    pvs->height = h;
    Py_DECREF((PyObject *) pvs);
    Py_DECREF((PyObject *) is);
    return 0;
}

/* display the current picture, if any */
static void video_display(PyMovie *is)
{
    Py_INCREF((PyObject *) is);
    PyVideoStream *pvs = (PyVideoStream *)PySequence_GetItem(is->streams, (Py_ssize_t) is->video_stream);
    Py_INCREF((PyObject *) pvs);
    
    if (!pvs->out_surf||!pvs->bmp)
        video_open(is);

    else if (is->vid_stream_ix>0)
        video_image_display(is);
    Py_DECREF((PyObject *) pvs);
    Py_DECREF((PyObject *) is);
}

static Uint32 sdl_refresh_timer_cb(Uint32 interval, void *opaque)
{
    Py_INCREF((PyObject *) opaque);
    SDL_Event event;
    event.type = FF_REFRESH_EVENT;
    event.user.data1 = opaque;
    SDL_PushEvent(&event);
    Py_DECREF((PyObject *) opaque);
    return 0; /* 0 means stop timer */
}

/* schedule a video refresh in 'delay' ms */
static void schedule_refresh(PyMovie *is, int delay)
{
    Py_INCREF((PyObject *) is);
    if(!delay) delay=1; //SDL seems to be buggy when the delay is 0
    SDL_AddTimer(delay, sdl_refresh_timer_cb, is);
    Py_DECREF((PyObject *) is);
}


static int audio_write_get_buf_size(PyMovie *is)
{
    Py_INCREF((PyObject *) is);
    PyAudioStream *pas = (PyAudioStream *)PySequence_GetItem(is->streams, (Py_ssize_t) is->audio_stream);
    Py_INCREF((PyObject *) pas);
    int temp = pas->audio_buf_size - pas->audio_buf_index;
    Py_DECREF((PyObject *) pas);
    Py_DECREF((PyObject *) is);
    return temp;
}

/* get the current audio clock value */
static double get_audio_clock(PyMovie *is)
{
    Py_INCREF((PyObject *) is);
    double pts;
    int hw_buf_size, bytes_per_sec;

    PyAudioStream *pas = (PyAudioStream *)PySequence_GetItem(is->streams, (Py_ssize_t) is->audio_stream);
    Py_INCREF((PyObject *) pas);
    
    pts = pas->audio_clock;
    hw_buf_size = audio_write_get_buf_size(is);
    bytes_per_sec = 0;
    if (pas->audio_st) {
        bytes_per_sec = pas->audio_st->codec->sample_rate *
            2 * pas->audio_st->codec->channels;
    }
    if (bytes_per_sec)
        pts -= (double)hw_buf_size / bytes_per_sec;
    Py_DECREF((PyObject *) pas);
    Py_DECREF((PyObject *) is);
    return pts;
}

/* get the current video clock value */
static double get_video_clock(PyMovie *is)
{
    Py_INCREF((PyObject *) is);
    double delta;
    PyVideoStream *pvs = (PyVideoStream *)PySequence_GetItem(is->streams, (Py_ssize_t) is->video_stream);
    Py_INCREF((PyObject *)pvs);
    
    if (pvs->paused) {
        delta = 0;
    } else {
        delta = (av_gettime() - pvs->video_current_pts_time) / 1000000.0;
    }
    double temp = pvs->video_current_pts+delta;
    Py_DECREF((PyObject *) pvs);
    Py_DECREF((PyObject *) is);
    return temp;
}

/* get the current external clock value */
static double get_external_clock(PyMovie *is)
{
    Py_INCREF((PyObject *) is);
    int64_t ti;
    ti = av_gettime();
    double res = is->external_clock + ((ti - is->external_clock_time) * 1e-6);
    Py_DECREF((PyObject *) is);
    return res;
}

/* get the current master clock value */
static double get_master_clock(PyMovie *is)
{
    Py_INCREF((PyObject *) is);
    double val;
    PyVideoStream *pvs;
    PyAudioStream *pas;
    
    pvs = (PyVideoStream *)PySequence_GetItem(is->streams, (Py_ssize_t) is->video_stream);
    Py_INCREF((PyObject *)pvs);
    
    pas = (PyAudioStream *)PySequence_GetItem(is->streams, (Py_ssize_t) is->audio_stream);
    Py_INCREF((PyObject *)pas);
    
    if (is->av_sync_type == AV_SYNC_VIDEO_MASTER) {
        if (pvs->video_st)
            val = get_video_clock(is);
        else
            val = get_audio_clock(is);
    } else if (is->av_sync_type == AV_SYNC_AUDIO_MASTER) {
        if (pas->audio_st)
            val = get_audio_clock(is);
        else
            val = get_video_clock(is);
    } else {
        val = get_external_clock(is);
    }
    Py_DECREF((PyObject *)pvs);
    Py_DECREF((PyObject *)pas);
    Py_DECREF((PyObject *) is);
    return val;
}

/* seek in the stream */
static void stream_seek(PyMovie *is, int64_t pos, int rel)
{
    Py_INCREF((PyObject *) is);
    if (!is->seek_req) {
        is->seek_pos = pos;
        is->seek_flags = rel < 0 ? AVSEEK_FLAG_BACKWARD : 0;

        is->seek_req = 1;
    }
    Py_DECREF((PyObject *) is);
}

/* pause or resume the video */
static void stream_pause(PyMovie *is)
{
    Py_INCREF((PyObject *) is);
    is->paused = !is->paused;
    if (!is->paused) {
        PyVideoStream *pvs;
        PyAudioStream *pas;
        pvs = (PyVideoStream *)PySequence_GetItem(is->streams, (Py_ssize_t) is->video_stream);
        Py_INCREF((PyObject *)pvs);
        
        pas = (PyAudioStream *)PySequence_GetItem(is->streams, (Py_ssize_t) is->audio_stream);
        Py_INCREF((PyObject *)pas);

        pvs->video_current_pts = get_video_clock(is);
        
        is->frame_timer += (av_gettime() - pvs->video_current_pts_time) / 1000000.0;
        Py_DECREF((PyObject *) pvs);
        Py_DECREF((PyObject *) pas);
    }
    Py_DECREF((PyObject *) is);
}

static double compute_frame_delay(double frame_current_pts, PyMovie *is)
{
    Py_INCREF((PyObject *) is);

    double actual_delay, delay, sync_threshold, ref_clock, diff;

    /* compute nominal delay */
    delay = frame_current_pts - is->frame_last_pts;
    if (delay <= 0 || delay >= 10.0) {
        /* if incorrect delay, use previous one */
        delay = is->frame_last_delay;
    } else {
        is->frame_last_delay = delay;
    }
    is->frame_last_pts = frame_current_pts;

    PyVideoStream *pvs;
    pvs = (PyVideoStream *)PySequence_GetItem(is->streams, (Py_ssize_t) is->video_stream);
    Py_INCREF((PyObject *)pvs);
       
    PyAudioStream *pas; 
    pas = (PyAudioStream *)PySequence_GetItem(is->streams, (Py_ssize_t) is->audio_stream);
    Py_INCREF((PyObject *)pas);
   
    /* update delay to follow master synchronisation source */
    if (((is->av_sync_type == AV_SYNC_AUDIO_MASTER && pas->audio_st) ||
         is->av_sync_type == AV_SYNC_EXTERNAL_CLOCK)) {
        /* if video is slave, we try to correct big delays by
           duplicating or deleting a frame */
        ref_clock = get_master_clock(is);
        diff = frame_current_pts - ref_clock;

        /* skip or repeat frame. We take into account the
           delay to compute the threshold. I still don't know
           if it is the best guess */
        sync_threshold = FFMAX(AV_SYNC_THRESHOLD, delay);
        if (fabs(diff) < AV_NOSYNC_THRESHOLD) {
            if (diff <= -sync_threshold)
                delay = 0;
            else if (diff >= sync_threshold)
                delay = 2 * delay;
        }
    }

    is->frame_timer += delay;
    /* compute the REAL delay (we need to do that to avoid
       long term errors */
    actual_delay = is->frame_timer - (av_gettime() / 1000000.0);
    if (actual_delay < 0.010) {
        /* XXX: should skip picture */
        actual_delay = 0.010;
    }

#if defined(DEBUG_SYNC)
    printf("video: delay=%0.3f actual_delay=%0.3f pts=%0.3f A-V=%f\n",
            delay, actual_delay, frame_current_pts, -diff);
#endif
    Py_DECREF((PyObject *) pvs);
    Py_DECREF((PyObject *) pas);
    Py_DECREF((PyObject *) is);
    return actual_delay;
}



/* called to display each frame */
static void video_refresh_timer(void *opaque)
{
    PyMovie *is = opaque;
    Py_INCREF((PyObject *) is);
    
    VideoPicture *vp;

    SubPicture *sp, *sp2;

    PyVideoStream *pvs;
    PyAudioStream *pas;
    PySubtitleStream   *pss;

    pvs = (PyVideoStream *)PySequence_GetItem(is->streams, (Py_ssize_t) is->video_stream);
    Py_INCREF(pvs); 
        
    pas = (PyAudioStream *)PySequence_GetItem(is->streams, (Py_ssize_t) is->audio_stream);
    Py_INCREF(pas);

    if (pvs->video_st) {
        if (pvs->pictq_size == 0) {
            /* if no picture, need to wait */
            schedule_refresh(is, 1);
        } else {
            /* dequeue the picture */
            vp = &pvs->pictq[pvs->pictq_rindex];

            /* update current video pts */
            pvs->video_current_pts = vp->pts;
            pvs->video_current_pts_time = av_gettime();

            /* launch timer for next picture */
            schedule_refresh(is, (int)(compute_frame_delay(vp->pts, is) * 1000 + 0.5));

            if(is->subtitle_stream>-1) {
                pss = (PySubtitleStream *)PySequence_GetItem(is->streams, (Py_ssize_t) is->subtitle_stream);
                Py_INCREF(pss);
                
                if (pss->subtitle_stream_changed) {
                    SDL_LockMutex(pss->subpq_mutex);

                    while (pss->subpq_size) {
                        free_subpicture(&pss->subpq[pss->subpq_rindex]);

                        /* update queue size and signal for next picture */
                        if (++pss->subpq_rindex == SUBPICTURE_QUEUE_SIZE)
                            pss->subpq_rindex = 0;

                        pss->subpq_size--;
                    }
                    pss->subtitle_stream_changed = 0;

                    SDL_CondSignal(pss->subpq_cond);
                    SDL_UnlockMutex(pss->subpq_mutex);
                } else {
                    if (pss->subpq_size > 0) {
                        sp = &pss->subpq[pss->subpq_rindex];

                        if (pss->subpq_size > 1)
                            sp2 = &pss->subpq[(pss->subpq_rindex + 1) % SUBPICTURE_QUEUE_SIZE];
                        else
                            sp2 = NULL;

                        if ((pvs->video_current_pts > (sp->pts + ((float) sp->sub.end_display_time / 1000)))
                                || (sp2 && pvs->video_current_pts > (sp2->pts + ((float) sp2->sub.start_display_time / 1000))))
                        {
                            free_subpicture(sp);

                            /* update queue size and signal for next picture */
                            if (++pss->subpq_rindex == SUBPICTURE_QUEUE_SIZE)
                                pss->subpq_rindex = 0;

                            SDL_LockMutex(pss->subpq_mutex);
                            pss->subpq_size--;
                            SDL_CondSignal(pss->subpq_cond);
                            SDL_UnlockMutex(pss->subpq_mutex);
                        }
                    }
                }
                Py_DECREF((PyObject *)pss);
            }

            /* display picture */
            video_display(is);

            /* update queue size and signal for next picture */
            pvs->pictq_rindex++;
            if (pvs->pictq_rindex == VIDEO_PICTURE_QUEUE_SIZE)
                pvs->pictq_rindex = 0;

            SDL_LockMutex(pvs->pictq_mutex);
            pvs->pictq_size--;
            SDL_CondSignal(pvs->pictq_cond);
            SDL_UnlockMutex(pvs->pictq_mutex);
        }
    } else if (pas->audio_st) {
        /* draw the next audio frame */

        schedule_refresh(is, 40);

        /* if only audio stream, then display the audio bars (better
           than nothing, just to test the implementation */

        /* display picture */
        video_display(is);
    } else {
        schedule_refresh(is, 100);
    }
    Py_DECREF((PyObject *) pvs);
    Py_DECREF((PyObject *) pas);
    Py_DECREF((PyObject *) is);
}

/* allocate a picture (needs to do that in main thread to avoid
   potential locking problems */
static void alloc_picture(void *opaque)
{
    PyMovie *is = opaque;
    Py_INCREF((PyObject *) is);
    VideoPicture *vp;
    
    PyVideoStream *pvs;

    pvs = (PyVideoStream *)PySequence_GetItem(is->streams, (Py_ssize_t) is->video_stream);
    Py_INCREF(pvs);
    
    vp = &pvs->pictq[pvs->pictq_windex];

    if (pvs->bmp)
        SDL_FreeYUVOverlay(pvs->bmp);
    if (pvs->out_surf)
        SDL_FreeSurface(pvs->out_surf);
#if 0
    /* XXX: use generic function */
    /* XXX: disable overlay if no hardware acceleration or if RGB format */
    switch(is->video_st->codec->pix_fmt) {
    case PIX_FMT_YUV420P:
    case PIX_FMT_YUV422P:
    case PIX_FMT_YUV444P:
    case PIX_FMT_YUYV422:
    case PIX_FMT_YUV410P:
    case PIX_FMT_YUV411P:
        is_yuv = 1;
        break;
    default:
        is_yuv = 0;
        break;
    }
#endif
    if(is->overlay>0)
    {
        SDL_Surface *screen = SDL_GetVideoSurface ();
        pvs->bmp = SDL_CreateYUVOverlay(pvs->video_st->codec->width,
                                   pvs->video_st->codec->height,
                                   SDL_YV12_OVERLAY,
                                   screen);
    }
    else
    {
        pvs->out_surf= SDL_GetVideoSurface ();
    }
    vp->width = pvs->video_st->codec->width;
    vp->height = pvs->video_st->codec->height;

    SDL_LockMutex(pvs->pictq_mutex);
    vp->allocated = 1;
    SDL_CondSignal(pvs->pictq_cond);
    SDL_UnlockMutex(pvs->pictq_mutex);
    Py_DECREF((PyObject *) pvs);
    Py_DECREF((PyObject *) is);
}

/**
 *
 * @param pts the dts of the pkt / pts of the frame and guessed if not known
 */
static int queue_picture(PyMovie *is, AVFrame *src_frame, double pts)
{
    Py_INCREF((PyObject *) is);
    VideoPicture *vp;

    PyVideoStream *pvs;

    pvs = (PyVideoStream *)PySequence_GetItem(is->streams, (Py_ssize_t) is->video_stream);
    Py_INCREF(pvs);

    int dst_pix_fmt;
    AVPicture pict;
    static struct SwsContext *img_convert_ctx;

    /* wait until we have space to put a new picture */
    SDL_LockMutex(pvs->pictq_mutex);
    while (pvs->pictq_size >= VIDEO_PICTURE_QUEUE_SIZE &&
           !pvs->videoq.abort_request) {
        SDL_CondWait(pvs->pictq_cond, pvs->pictq_mutex);
    }
    SDL_UnlockMutex(pvs->pictq_mutex);

    if (pvs->videoq.abort_request)
        return -1;

    vp = &pvs->pictq[pvs->pictq_windex];

    /* alloc or resize hardware picture buffer */
    if (!pvs->bmp || !pvs->out_surf ||
        vp->width != pvs->video_st->codec->width ||
        vp->height != pvs->video_st->codec->height) {
        SDL_Event event;

        vp->allocated = 0;

        /* the allocation must be done in the main thread to avoid
           locking problems */
        event.type = FF_ALLOC_EVENT;
        event.user.data1 = is;
        SDL_PushEvent(&event);

        /* wait until the picture is allocated */
        SDL_LockMutex(pvs->pictq_mutex);
        while (!vp->allocated && !pvs->videoq.abort_request) {
            SDL_CondWait(pvs->pictq_cond, pvs->pictq_mutex);
        }
        SDL_UnlockMutex(pvs->pictq_mutex);

        if (pvs->videoq.abort_request)
            return -1;
    }

    /* if the frame is not skipped, then display it */
    if (pvs->bmp||pvs->out_surf) {
        /* get a pointer on the bitmap */
        if(is->overlay>0)
        {
            dst_pix_fmt = PIX_FMT_YUV422;
              
            SDL_LockYUVOverlay (pvs->bmp);

            pict.data[0] = pvs->bmp->pixels[0];
            pict.data[1] = pvs->bmp->pixels[2];
            pict.data[2] = pvs->bmp->pixels[1];

            pict.linesize[0] = pvs->bmp->pitches[0];
            pict.linesize[1] = pvs->bmp->pitches[2];
            pict.linesize[2] = pvs->bmp->pitches[1];
            //sws_flags = av_get_int(sws_opts, "sws_flags", NULL);
            img_convert_ctx = sws_getCachedContext(img_convert_ctx,
                pvs->video_st->codec->width, pvs->video_st->codec->height,
                pvs->video_st->codec->pix_fmt,
                pvs->video_st->codec->width, pvs->video_st->codec->height,
                dst_pix_fmt, sws_flags, NULL, NULL, NULL);
            if (img_convert_ctx == NULL) {
                fprintf(stderr, "Cannot initialize the conversion context\n");
                exit(1);
            }
            sws_scale(img_convert_ctx, src_frame->data, src_frame->linesize,
                      0, pvs->video_st->codec->height, pict.data, pict.linesize);
            /* update the bitmap content */
            SDL_UnlockYUVOverlay(pvs->bmp);
        }
        else
        {
            //TODO:fix this as well
            #if 0
            dst_pix_fmt = PIX_FMT_RGB24;
              
            SDL_LockSurface (pvs->out_surf);

            pict.data[0] = pvs->out_surf->pixels[0];
            pict.data[1] = pvs->out_surf->pixels[1];
            pict.data[2] = pvs->out_surf->pixels[2];

            pict.linesize[0] = pvs->out_surf->pitch;
            pict.linesize[1] = pvs->out_surf->pitch;
            pict.linesize[2] = pvs->out_surf->pitch;
            //sws_flags = av_get_int(sws_opts, "sws_flags", NULL);
            img_convert_ctx = sws_getCachedContext(img_convert_ctx,
                pvs->video_st->codec->width, pvs->video_st->codec->height,
                pvs->video_st->codec->pix_fmt,
                pvs->video_st->codec->width, pvs->video_st->codec->height,
                dst_pix_fmt, sws_flags, NULL, NULL, NULL);
            if (img_convert_ctx == NULL) {
                
                PyErr_SetString(PyExc_MemoryError ,"Cannot initialize the conversion context.");
                //fprintf(stderr, "Cannot initialize the conversion context\n");
                //exit(1);
            }
            sws_scale(img_convert_ctx, src_frame->data, src_frame->linesize,
                      0, pvs->video_st->codec->height, pict.data, pict.linesize);
            /* update the bitmap content */
            SDL_UnlockSurface(pvs->out_surf);
            #endif
        }
        
        vp->pts = pts;

        /* now we can update the picture count */
        if (++pvs->pictq_windex == VIDEO_PICTURE_QUEUE_SIZE)
            pvs->pictq_windex = 0;
        SDL_LockMutex(pvs->pictq_mutex);
        pvs->pictq_size++;
        SDL_UnlockMutex(pvs->pictq_mutex);
    }
    Py_DECREF((PyObject *) pvs);
    Py_DECREF((PyObject *) is);
    return 0;
}


/**
 * compute the exact PTS for the picture if it is omitted in the stream
 * @param pts1 the dts of the pkt / pts of the frame
 */
static int output_picture2(PyMovie *is, AVFrame *src_frame, double pts1)
{
    Py_INCREF((PyObject *) is);
    
    double frame_delay, pts;

    pts = pts1;

    PyVideoStream *pvs;

    pvs = (PyVideoStream *)PySequence_GetItem(is->streams, (Py_ssize_t) is->video_stream);
    Py_INCREF((PyObject *) pvs);
    
    if (pts != 0) {
        /* update video clock with pts, if present */
        pvs->video_clock = pts;
    } else {
        pts = pvs->video_clock;
    }
    /* update video clock for next frame */
    frame_delay = av_q2d(pvs->video_st->codec->time_base);
    /* for MPEG2, the frame can be repeated, so we update the
       clock accordingly */
    frame_delay += src_frame->repeat_pict * (frame_delay * 0.5);
    pvs->video_clock += frame_delay;

#if defined(DEBUG_SYNC) && 0
    {
        int ftype;
        if (src_frame->pict_type == FF_B_TYPE)
            ftype = 'B';
        else if (src_frame->pict_type == FF_I_TYPE)
            ftype = 'I';
        else
            ftype = 'P';
        printf("frame_type=%c clock=%0.3f pts=%0.3f\n",
               ftype, pts, pts1);
    }
#endif
    Py_DECREF((PyObject *)pvs);
    Py_DECREF((PyObject *) is);
    return queue_picture(is, src_frame, pts);
}

static int video_thread(void *arg)
{
    PyMovie *is = arg;
    Py_DECREF((PyObject *) is);
    AVPacket pkt1, *pkt = &pkt1;
    int len1, got_picture;
    AVFrame *frame= avcodec_alloc_frame();
    double pts;

    PyVideoStream *pvs;

    pvs = (PyVideoStream *)PySequence_GetItem(is->streams, (Py_ssize_t) is->video_stream);
    Py_INCREF((PyObject *)pvs);
    
    for(;;) {
        while (pvs->paused && !pvs->videoq.abort_request) {
            SDL_Delay(10);
        }
        if (packet_queue_get(&pvs->videoq, pkt, 1) < 0)
            break;

        if(pkt->data == flush_pkt.data){
            avcodec_flush_buffers(pvs->video_st->codec);
            continue;
        }

        /* NOTE: ipts is the PTS of the _first_ picture beginning in
           this packet, if any */
        pvs->video_st->codec->reordered_opaque= pkt->pts;
        len1 = avcodec_decode_video(pvs->video_st->codec,
                                    frame, &got_picture,
                                    pkt->data, pkt->size);

        if(   (decoder_reorder_pts || pkt->dts == AV_NOPTS_VALUE)
           && frame->reordered_opaque != AV_NOPTS_VALUE)
            pts= frame->reordered_opaque;
        else if(pkt->dts != AV_NOPTS_VALUE)
            pts= pkt->dts;
        else
            pts= 0;
        pts *= av_q2d(pvs->video_st->time_base);

//            if (len1 < 0)
//                break;
        if (got_picture) {
            if (output_picture2(is, frame, pts) < 0)
                goto the_end;
        }
        av_free_packet(pkt);
        if (step)
            if (is)
                stream_pause(is);
    }
 the_end:
    Py_DECREF((PyObject *) pvs);
    Py_DECREF((PyObject *) is);
    av_free(frame);
    return 0;
}

static int subtitle_thread(void *arg)
{
    PyMovie *is = arg;
    Py_INCREF((PyObject *) is);
    SubPicture *sp;
    AVPacket pkt1, *pkt = &pkt1;
    int len1, got_subtitle;
    double pts;
    int i, j;
    int r, g, b, y, u, v, a;

    PySubtitleStream   *pss;

    pss = (PySubtitleStream *)PySequence_GetItem(is->streams, (Py_ssize_t) is->subtitle_stream);
    Py_INCREF((PyObject *)pss);
    
    for(;;) {
        while (pss->paused && !pss->subtitleq.abort_request) {
            SDL_Delay(10);
        }
        if (packet_queue_get(&pss->subtitleq, pkt, 1) < 0)
            break;

        if(pkt->data == flush_pkt.data){
            avcodec_flush_buffers(pss->subtitle_st->codec);
            continue;
        }
        SDL_LockMutex(pss->subpq_mutex);
        while (pss->subpq_size >= SUBPICTURE_QUEUE_SIZE &&
               !pss->subtitleq.abort_request) {
            SDL_CondWait(pss->subpq_cond, pss->subpq_mutex);
        }
        SDL_UnlockMutex(pss->subpq_mutex);

        if (pss->subtitleq.abort_request)
            goto the_end;

        sp = &pss->subpq[pss->subpq_windex];

       /* NOTE: ipts is the PTS of the _first_ picture beginning in
           this packet, if any */
        pts = 0;
        if (pkt->pts != AV_NOPTS_VALUE)
            pts = av_q2d(pss->subtitle_st->time_base)*pkt->pts;

        len1 = avcodec_decode_subtitle(pss->subtitle_st->codec,
                                    &sp->sub, &got_subtitle,
                                    pkt->data, pkt->size);
//            if (len1 < 0)
//                break;
        if (got_subtitle && sp->sub.format == 0) {
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
            if (++pss->subpq_windex == SUBPICTURE_QUEUE_SIZE)
                pss->subpq_windex = 0;
            SDL_LockMutex(pss->subpq_mutex);
            pss->subpq_size++;
            SDL_UnlockMutex(pss->subpq_mutex);
        }
        av_free_packet(pkt);
//        if (step)
//            if (cur_stream)
//                stream_pause(cur_stream);
    }
 the_end:
    Py_DECREF((PyObject *) pss);
    Py_DECREF((PyObject *) is);
    return 0;
}

/* return the new audio buffer size (samples can be added or deleted
   to get better sync if video or external master clock) */
static int synchronize_audio(PyMovie *is, short *samples,
                             int samples_size1, double pts)
{
    Py_INCREF((PyObject *) is);
    
    int n, samples_size;
    double ref_clock;

    PyAudioStream   *pas;
    PyVideoStream   *pvs;

    pas = (PyAudioStream *)PySequence_GetItem(is->streams, (Py_ssize_t) is->audio_stream);
    Py_INCREF((PyObject *)pas);

    pvs = (PyVideoStream *)PySequence_GetItem(is->streams, (Py_ssize_t) is->video_stream);
    Py_INCREF((PyObject *)pvs);

    n = 2 * pas->audio_st->codec->channels;
    samples_size = samples_size1;

    /* if not master, then we try to remove or add samples to correct the clock */
    if (((pas->av_sync_type == AV_SYNC_VIDEO_MASTER && pvs->video_st) ||
         pas->av_sync_type == AV_SYNC_EXTERNAL_CLOCK)) {
        double diff, avg_diff;
        int wanted_size, min_size, max_size, nb_samples;

        ref_clock = get_master_clock(is);
        diff = get_audio_clock(is) - ref_clock;

        if (diff < AV_NOSYNC_THRESHOLD) {
            pas->audio_diff_cum = diff + pas->audio_diff_avg_coef * pas->audio_diff_cum;
            if (pas->audio_diff_avg_count < AUDIO_DIFF_AVG_NB) {
                /* not enough measures to have a correct estimate */
                pas->audio_diff_avg_count++;
            } else {
                /* estimate the A-V difference */
                avg_diff = pas->audio_diff_cum * (1.0 - pas->audio_diff_avg_coef);

                if (fabs(avg_diff) >= pas->audio_diff_threshold) {
                    wanted_size = samples_size + ((int)(diff * pas->audio_st->codec->sample_rate) * n);
                    nb_samples = samples_size / n;

                    min_size = ((nb_samples * (100 - SAMPLE_CORRECTION_PERCENT_MAX)) / 100) * n;
                    max_size = ((nb_samples * (100 + SAMPLE_CORRECTION_PERCENT_MAX)) / 100) * n;
                    if (wanted_size < min_size)
                        wanted_size = min_size;
                    else if (wanted_size > max_size)
                        wanted_size = max_size;

                    /* add or remove samples to correction the synchro */
                    if (wanted_size < samples_size) {
                        /* remove samples */
                        samples_size = wanted_size;
                    } else if (wanted_size > samples_size) {
                        uint8_t *samples_end, *q;
                        int nb;

                        /* add samples */
                        nb = (samples_size - wanted_size);
                        samples_end = (uint8_t *)samples + samples_size - n;
                        q = samples_end + n;
                        while (nb > 0) {
                           
                            memcpy(q, samples_end, n);
                            q += n;
                            nb -= n;
                        }
                        samples_size = wanted_size;
                    }
                }
#if 0
                printf("diff=%f adiff=%f sample_diff=%d apts=%0.3f vpts=%0.3f %f\n",
                       diff, avg_diff, samples_size - samples_size1,
                       is->audio_clock, is->video_clock, is->audio_diff_threshold);
#endif
            }
        } else {
            /* too big difference : may be initial PTS errors, so
               reset A-V filter */
            pas->audio_diff_avg_count = 0;
            pas->audio_diff_cum = 0;
        }
    }
    Py_DECREF((PyObject *)pas);
    Py_DECREF((PyObject *)pvs);
    Py_DECREF((PyObject *) is);
    return samples_size;
}

/* decode one audio frame and returns its uncompressed size */
static int audio_decode_frame(PyMovie *is, double *pts_ptr)
{
    Py_INCREF((PyObject *) is);
    PyAudioStream   *pas;

    pas = (PyAudioStream *)PySequence_GetItem(is->streams, (Py_ssize_t) is->audio_stream);
    Py_INCREF(pas);
    
    AVPacket *pkt = &pas->audio_pkt;
    AVCodecContext *dec= pas->audio_st->codec;
    int n, len1, data_size;
    double pts;

    for(;;) {
        /* NOTE: the audio packet can contain several frames */
        while (pas->audio_pkt_size > 0) {
            data_size = sizeof(pas->audio_buf1);
            len1 = avcodec_decode_audio2(dec,
                                        (int16_t *)pas->audio_buf1, &data_size,
                                        pas->audio_pkt_data, pas->audio_pkt_size);
            if (len1 < 0) {
                /* if error, we skip the frame */
                pas->audio_pkt_size = 0;
                break;
            }

            pas->audio_pkt_data += len1;
            pas->audio_pkt_size -= len1;
            if (data_size <= 0)
                continue;

            if (dec->sample_fmt != pas->audio_src_fmt) {
                if (pas->reformat_ctx)
                    av_audio_convert_free(pas->reformat_ctx);
                pas->reformat_ctx= av_audio_convert_alloc(SAMPLE_FMT_S16, 1,
                                                         dec->sample_fmt, 1, NULL, 0);
                if (!pas->reformat_ctx) {
                    fprintf(stderr, "Cannot convert %s sample format to %s sample format\n",
                        avcodec_get_sample_fmt_name(dec->sample_fmt),
                        avcodec_get_sample_fmt_name(SAMPLE_FMT_S16));
                        break;
                }
                pas->audio_src_fmt= dec->sample_fmt;
            }

            if (pas->reformat_ctx) {
                const void *ibuf[6]= {pas->audio_buf1};
                void *obuf[6]= {pas->audio_buf2};
                int istride[6]= {av_get_bits_per_sample_format(dec->sample_fmt)/8};
                int ostride[6]= {2};
                int len= data_size/istride[0];
                if (av_audio_convert(pas->reformat_ctx, obuf, ostride, ibuf, istride, len)<0) {
                    PyErr_WarnEx(NULL, "av_audio_convert() failed", 1);
                    //printf("av_audio_convert() failed\n");
                    break;
                }
                pas->audio_buf= pas->audio_buf2;
                /* FIXME: existing code assume that data_size equals framesize*channels*2
                          remove this legacy cruft */
                data_size= len*2;
            }else{
                pas->audio_buf= pas->audio_buf1;
            }

            /* if no pts, then compute it */
            pts = pas->audio_clock;
            *pts_ptr = pts;
            n = 2 * dec->channels;
            pas->audio_clock += (double)data_size /
                (double)(n * dec->sample_rate);
#if defined(DEBUG_SYNC)
            {
                static double last_clock;
                printf("audio: delay=%0.3f clock=%0.3f pts=%0.3f\n",
                       pas->audio_clock - last_clock,
                       pas->audio_clock, pts);
                last_clock = pas->audio_clock;
            }
#endif
            Py_DECREF((PyObject *)pas);
            return data_size;
        }

        /* free the current packet */
        if (pkt->data)
            av_free_packet(pkt);

        if (pas->paused || pas->audioq.abort_request) {
            return -1;
        }

        /* read next packet */
        if (packet_queue_get(&pas->audioq, pkt, 1) < 0)
            return -1;
        if(pkt->data == flush_pkt.data){
            avcodec_flush_buffers(dec);
            continue;
        }

        pas->audio_pkt_data = pkt->data;
        pas->audio_pkt_size = pkt->size;

        /* if update the audio clock with the pts */
        if (pkt->pts != AV_NOPTS_VALUE) {
            pas->audio_clock = av_q2d(pas->audio_st->time_base)*pkt->pts;
        }
    }
    Py_DECREF((PyObject *) pas);
    Py_DECREF((PyObject *) is);
}



/* prepare a new audio buffer */
static void sdl_audio_callback(void *opaque, Uint8 *stream, int len)
{
    PyMovie *is = opaque;
    Py_INCREF((PyObject *) is);
    int audio_size, len1;
    double pts;

    PyAudioStream   *pas;

    pas = (PyAudioStream *)PySequence_GetItem(is->streams, (Py_ssize_t) is->audio_stream);
    Py_INCREF(pas);

    audio_callback_time = av_gettime();

    while (len > 0) {
        if (pas->audio_buf_index >= pas->audio_buf_size) {
           audio_size = audio_decode_frame(is, &pts);
           if (audio_size < 0) {
                /* if error, just output silence */
               pas->audio_buf = pas->audio_buf1;
               pas->audio_buf_size = 1024;
               memset(pas->audio_buf, 0, pas->audio_buf_size);
           } else {
               
               audio_size = synchronize_audio(is, (int16_t *)pas->audio_buf, audio_size,
                                              pts);
               pas->audio_buf_size = audio_size;
           }
           pas->audio_buf_index = 0;
        }
        len1 = pas->audio_buf_size - pas->audio_buf_index;
        if (len1 > len)
            len1 = len;
        memcpy(stream, (uint8_t *)pas->audio_buf + pas->audio_buf_index, len1);
        len -= len1;
        stream += len1;
        pas->audio_buf_index += len1;
    }
    Py_DECREF((PyObject *) pas);
    Py_DECREF((PyObject *) is);
}

/* open a given stream. Return 0 if OK */
static int stream_component_open(PyMovie *is, int stream_index)
{
    Py_INCREF((PyObject *) is);
    AVFormatContext *ic = is->ic;
    AVCodecContext *enc;
    AVCodec *codec;
    SDL_AudioSpec wanted_spec, spec;

    PyAudioStream* pas; 
    PyVideoStream* pvs;
    PySubtitleStream*   pss;

    if (stream_index < 0 || stream_index >= ic->nb_streams)
        return -1;
    enc = ic->streams[stream_index]->codec;

    /* prepare audio output */
    if (enc->codec_type == CODEC_TYPE_AUDIO) {
        if (enc->channels > 0) {
            enc->request_channels = FFMIN(2, enc->channels);
        } else {
            enc->request_channels = 2;
        }
    }

    codec = avcodec_find_decoder(enc->codec_id);
    enc->debug_mv = debug_mv;
    enc->debug = debug;
    enc->workaround_bugs = workaround_bugs;
    enc->lowres = lowres;
    if(lowres) enc->flags |= CODEC_FLAG_EMU_EDGE;
    enc->idct_algo= idct;
    if(fast) enc->flags2 |= CODEC_FLAG2_FAST;
    enc->skip_frame= skip_frame;
    enc->skip_idct= skip_idct;
    enc->skip_loop_filter= skip_loop_filter;
    enc->error_recognition= error_recognition;
    enc->error_concealment= error_concealment;

    //set_context_opts(enc, avctx_opts[enc->codec_type], 0);

    if (!codec ||
        avcodec_open(enc, codec) < 0)
        return -1;

    /* prepare audio output */
    if (enc->codec_type == CODEC_TYPE_AUDIO) {
        
        
        pas = _new_audio_stream();
        Py_INCREF((PyObject *) pas);
        wanted_spec.freq = enc->sample_rate;
        wanted_spec.format = AUDIO_S16SYS;
        wanted_spec.channels = enc->channels;
        wanted_spec.silence = 0;
        wanted_spec.samples = SDL_AUDIO_BUFFER_SIZE;
        wanted_spec.callback = sdl_audio_callback;
        wanted_spec.userdata = is;
        if (SDL_OpenAudio(&wanted_spec, &spec) < 0) {
            RAISE(PyExc_SDLError, SDL_GetError ());
            //fprintf(stderr, "SDL_OpenAudio: %s\n", SDL_GetError());
            //return -1;
        }
        pas->audio_hw_buf_size = spec.size;
        pas->audio_src_fmt= SAMPLE_FMT_S16;
    }

    if(thread_count>1)
        avcodec_thread_init(enc, thread_count);
    enc->thread_count= thread_count;
    ic->streams[stream_index]->discard = AVDISCARD_DEFAULT;
    
    switch(enc->codec_type) {
    case CODEC_TYPE_AUDIO:
        is->aud_stream_ix = stream_index;
        pas->audio_st = ic->streams[stream_index];
        pas->audio_buf_size = 0;
        pas->audio_buf_index = 0;

        /* init averaging filter */
        pas->audio_diff_avg_coef = exp(log(0.01) / AUDIO_DIFF_AVG_NB);
        pas->audio_diff_avg_count = 0;
        /* since we do not have a precise anough audio fifo fullness,
           we correct audio sync only if larger than this threshold */
        pas->audio_diff_threshold = 2.0 * SDL_AUDIO_BUFFER_SIZE / enc->sample_rate;
        
        //TODO:replace these memory allocations with python heap allocations
        memset(&pas->audio_pkt, 0, sizeof(pas->audio_pkt));
        packet_queue_init(&pas->audioq);
        is->audio_stream=(int) PyList_Size(is->streams);
        PyList_Append(is->streams, (PyObject *)pas);
        Py_DECREF((PyObject *) pas);
        SDL_PauseAudio(0);
        break;
    case CODEC_TYPE_VIDEO:

        pvs = _new_video_stream();
        Py_INCREF((PyObject *)pvs);
        is->vid_stream_ix = stream_index;
        pvs->video_st = ic->streams[stream_index];

        pvs->frame_last_delay = 40e-3;
        pvs->frame_timer = (double)av_gettime() / 1000000.0;
        pvs->video_current_pts_time = av_gettime();

        packet_queue_init(&pvs->videoq);
        pvs->video_tid = SDL_CreateThread(video_thread, is);

        is->video_stream=(int) PyList_Size(is->streams);
        PyList_Append(is->streams, (PyObject *)pvs);        

        Py_DECREF((PyObject *)pvs);
        break;
    case CODEC_TYPE_SUBTITLE:
        is->sub_stream_ix = stream_index;
        
        pss = _new_sub_stream();
        Py_INCREF((PyObject *) pss);
        pss->subtitle_st = ic->streams[stream_index];
        packet_queue_init(&pss->subtitleq);

        is->subtitle_stream=(int) PyList_Size(is->streams);
        PyList_Append(is->streams, (PyObject *)pss); 

        pss->subtitle_tid = SDL_CreateThread(subtitle_thread, is);
        Py_DECREF((PyObject *) pss);
        break;
    default:
        break;
    }
    Py_DECREF((PyObject *) is);
    return 0;
}

static void stream_component_close(PyMovie *is, int stream_index)
{
    Py_INCREF((PyObject *) is);
    AVFormatContext *ic = is->ic;
    AVCodecContext *enc;

    PyAudioStream* pas; 
    PyVideoStream* pvs;
    PySubtitleStream*   pss;
    if (stream_index < 0 || stream_index >= ic->nb_streams)
        return;
    enc = ic->streams[stream_index]->codec;

    switch(enc->codec_type) {
    case CODEC_TYPE_AUDIO:
        pas = (PyAudioStream *)PySequence_GetItem(is->streams, (Py_ssize_t) is->audio_stream);
        Py_INCREF((PyObject *)pas);
        
        packet_queue_abort(&pas->audioq);

        SDL_CloseAudio();

        packet_queue_end(&pas->audioq);
        if (pas->reformat_ctx)
            av_audio_convert_free(pas->reformat_ctx);
        break;
    case CODEC_TYPE_VIDEO:
        pvs = (PyVideoStream *)PySequence_GetItem(is->streams, (Py_ssize_t) is->video_stream);
        Py_INCREF((PyObject *)pvs);
        
        packet_queue_abort(&pvs->videoq);

        /* note: we also signal this mutex to make sure we deblock the
           video thread in all cases */
        SDL_LockMutex(pvs->pictq_mutex);
        SDL_CondSignal(pvs->pictq_cond);
        SDL_UnlockMutex(pvs->pictq_mutex);

        SDL_WaitThread(pvs->video_tid, NULL);

        packet_queue_end(&pvs->videoq);
        break;
    case CODEC_TYPE_SUBTITLE:
        pss = (PySubtitleStream *)PySequence_GetItem(is->streams, (Py_ssize_t) is->subtitle_stream);
        Py_INCREF((PyObject *)pss);
        
        packet_queue_abort(&pss->subtitleq);

        /* note: we also signal this mutex to make sure we deblock the
           video thread in all cases */
        SDL_LockMutex(pss->subpq_mutex);
        pss->subtitle_stream_changed = 1;

        SDL_CondSignal(pss->subpq_cond);
        SDL_UnlockMutex(pss->subpq_mutex);

        SDL_WaitThread(pss->subtitle_tid, NULL);

        packet_queue_end(&pss->subtitleq);
        break;
    default:
        break;
    }

    ic->streams[stream_index]->discard = AVDISCARD_ALL;
    avcodec_close(enc);
    switch(enc->codec_type) {
    case CODEC_TYPE_AUDIO:
        pas->audio_st = NULL;
        PySequence_SetItem(is->streams, (Py_ssize_t)is->audio_stream, Py_None);
        is->audio_stream = -1;
        Py_DECREF((PyObject *) pas);
        //_dealloc_aud_stream(pas);
        break;
    case CODEC_TYPE_VIDEO:
        pvs->video_st = NULL;
        PySequence_SetItem(is->streams, (Py_ssize_t)is->video_stream, Py_None);
        is->video_stream = -1;
        Py_DECREF((PyObject *) pvs);
        //_dealloc_vid_stream(pvs);
        break;
    case CODEC_TYPE_SUBTITLE:
        pss->subtitle_st = NULL;
        PySequence_SetItem(is->streams, (Py_ssize_t)is->subtitle_stream, Py_None);
        Py_DECREF(pss);
        is->subtitle_stream = -1;
        //_dealloc_sub_stream(pss);
        break;
    default:
        break;
    }
    Py_DECREF((PyObject *) is);
}

static int decode_interrupt_cb(void)
{
    return 0;
}

/* this thread gets the stream from the disk or the network */
static int decode_thread(void *arg)
{
    PyMovie *is = arg;
    Py_INCREF((PyObject *) is);
    AVFormatContext *ic;
    int err, i, ret, video_index, audio_index, subtitle_index;
    AVPacket pkt1, *pkt = &pkt1;
    AVFormatParameters params, *ap = &params;

    video_index = -1;
    audio_index = -1;
    subtitle_index = -1;
    is->video_stream = -1;
    is->audio_stream = -1;
    is->subtitle_stream = -1;

    PyAudioStream* pas;
    PyVideoStream* pvs;
    PySubtitleStream*   pss;

    url_set_interrupt_cb(decode_interrupt_cb);

    memset(ap, 0, sizeof(*ap));

    ap->width = frame_width;
    ap->height= frame_height;
    ap->time_base= (AVRational){1, 25};
    ap->pix_fmt = frame_pix_fmt;

    err = av_open_input_file(&ic, is->filename, is->iformat, 0, ap);
    if (err < 0) {
        PyErr_Format(PyExc_IOError, "There was a problem opening up %s", is->filename);
        //print_error(is->filename, err);
        ret = -1;
        goto fail;
    }
    is->ic = ic;

    if(genpts)
        ic->flags |= AVFMT_FLAG_GENPTS;

    err = av_find_stream_info(ic);
    if (err < 0) {
        PyErr_Format(PyExc_IOError, "%s: could not find codec parameters", is->filename);
        //fprintf(stderr, "%s: could not find codec parameters\n", is->filename);
        ret = -1;
        goto fail;
    }
    if(ic->pb)
        ic->pb->eof_reached= 0; //FIXME hack, ffplay maybe should not use url_feof() to test for the end

    /* if seeking requested, we execute it */
    if (start_time != AV_NOPTS_VALUE) {
        int64_t timestamp;

        timestamp = start_time;
        /* add the stream start time */
        if (ic->start_time != AV_NOPTS_VALUE)
            timestamp += ic->start_time;
        ret = av_seek_frame(ic, -1, timestamp, AVSEEK_FLAG_BACKWARD);
        if (ret < 0) {
            PyErr_Format(PyExc_IOError, "%s: could not seek to position %0.3f", is->filename, (double)timestamp/AV_TIME_BASE);
            //fprintf(stderr, "%s: could not seek to position %0.3f\n",
            //        is->filename, (double)timestamp / AV_TIME_BASE);
        }
    }

    for(i = 0; i < ic->nb_streams; i++) {
        AVCodecContext *enc = ic->streams[i]->codec;
        ic->streams[i]->discard = AVDISCARD_ALL;
        switch(enc->codec_type) {
        case CODEC_TYPE_AUDIO:
            if (wanted_audio_stream-- >= 0 && !audio_disable)
                audio_index = i;
            break;
        case CODEC_TYPE_VIDEO:
            if (wanted_video_stream-- >= 0 && !video_disable)
                video_index = i;
            break;
        case CODEC_TYPE_SUBTITLE:
            if (wanted_subtitle_stream-- >= 0 && !video_disable)
                subtitle_index = i;
            break;
        default:
            break;
        }
    }


    /* open the streams */
    if (audio_index >= 0) {
        stream_component_open(is, audio_index);
    }

    if (video_index >= 0) {
        stream_component_open(is, video_index);
    } 

    if (subtitle_index >= 0) {
        stream_component_open(is, subtitle_index);
    }

    if (is->video_stream < 0 && is->audio_stream < 0) {
        PyErr_Format(PyExc_IOError, "%s: could not open codecs", is->filename);
        //fprintf(stderr, "%s: could not open codecs\n", is->filename);
        ret = -1;
        goto fail;
    }

    for(;;) {
        SDL_LockMutex(is->general_mutex);        
        if (is->abort_request)
            SDL_UnlockMutex(is->general_mutex);
            break;
        if (is->paused != is->last_paused) {
            is->last_paused = is->paused;
            if (is->paused)
                av_read_pause(ic);
            else
                av_read_play(ic);
        }
#if CONFIG_RTSP_DEMUXER
        if (is->paused && !strcmp(ic->iformat->name, "rtsp")) {
            /* wait 10 ms to avoid trying to get another packet */
            /* XXX: horrible */
            SDL_Delay(10);
            SDL_UnlockMutex(is->general_mutex);
            continue;
        }
#endif
        if (is->seek_req) {
            int stream_index= -1;
            int64_t seek_target= is->seek_pos;

            if     (is->   vid_stream_ix >= 0) stream_index= is->   vid_stream_ix;
            else if(is->   aud_stream_ix >= 0) stream_index= is->   aud_stream_ix;
            else if(is->   sub_stream_ix >= 0) stream_index= is->   sub_stream_ix;

            if(stream_index>=0){
                seek_target= av_rescale_q(seek_target, AV_TIME_BASE_Q, ic->streams[stream_index]->time_base);
            }

    

            ret = av_seek_frame(is->ic, stream_index, seek_target, is->seek_flags);
            if (ret < 0) {
                PyErr_Format(PyExc_IOError, "%s: error while seeking", is->ic->filename);
                //fprintf(stderr, "%s: error while seeking\n", is->ic->filename);
            }else{
                
                
                if (is->audio_stream >= 0) {
                    pas = (PyAudioStream *)PySequence_GetItem(is->streams, (Py_ssize_t) is->audio_stream);
                    Py_INCREF((PyObject *)pas);
                    
        
                    packet_queue_flush(&pas->audioq);
                    packet_queue_put(&pas->audioq, &flush_pkt);
                    Py_DECREF((PyObject *)pas);
                    pas = NULL;
                }
                if (is->subtitle_stream >= 0) {
                    pss = (PySubtitleStream *)PySequence_GetItem(is->streams, (Py_ssize_t) is->subtitle_stream);
                    Py_INCREF((PyObject *)pss);

                    packet_queue_flush(&pss->subtitleq);
                    packet_queue_put(&pss->subtitleq, &flush_pkt);
                    Py_DECREF((PyObject *)pss);
                    pss = NULL;
                }
                if (is->video_stream >= 0) {
                    pvs = (PyVideoStream *)PySequence_GetItem(is->streams, (Py_ssize_t) is->video_stream);
                    Py_INCREF((PyObject *)pvs);

                    packet_queue_flush(&pvs->videoq);
                    packet_queue_put(&pvs->videoq, &flush_pkt);
                    Py_DECREF((PyObject *)pvs);
                    pvs = NULL;
                }
            }
            is->seek_req = 0;
        }
        if(!pas)
        {
            pas = (PyAudioStream *)PySequence_GetItem(is->streams, (Py_ssize_t) is->audio_stream);
            Py_XINCREF((PyObject *)pas);
        }
        if(!pss)
        {
            pss = (PySubtitleStream *)PySequence_GetItem(is->streams, (Py_ssize_t) is->subtitle_stream);
            Py_XINCREF((PyObject *) pss);
        }
        if(!pvs)
        {
            pvs = (PyVideoStream *)PySequence_GetItem(is->streams, (Py_ssize_t) is->video_stream);
            Py_XINCREF((PyObject *)pvs);
        }
        /* if the queue are full, no need to read more */
        if ((pas !=NULL && pas->audioq.size > MAX_AUDIOQ_SIZE) || //yay for short circuit logic testing
            (pvs !=NULL && pvs->videoq.size > MAX_VIDEOQ_SIZE )||
            (pss !=NULL && pss->subtitleq.size > MAX_SUBTITLEQ_SIZE)) {
            /* wait 10 ms */
            SDL_Delay(10);
            Py_XDECREF((PyObject *) pas);
            Py_XDECREF((PyObject *) pvs);
            Py_XDECREF((PyObject *) pss);
            SDL_UnlockMutex(is->general_mutex);
            continue;
        }
        if(url_feof(ic->pb)) {
            av_init_packet(pkt);
            pkt->data=NULL;
            pkt->size=0;
            pkt->stream_index= is->vid_stream_ix;
            packet_queue_put(&pvs->videoq, pkt);
            Py_XDECREF((PyObject *) pas);
            Py_XDECREF((PyObject *) pvs);
            Py_XDECREF((PyObject *) pss);
            SDL_UnlockMutex(is->general_mutex);
            continue;
        }
        ret = av_read_frame(ic, pkt);
        if (ret < 0) {
            if (ret != AVERROR_EOF && url_ferror(ic->pb) == 0) {
                Py_XDECREF((PyObject *) pas);
                Py_XDECREF((PyObject *) pvs);
                Py_XDECREF((PyObject *) pss);         
                SDL_UnlockMutex(is->general_mutex);       
                SDL_Delay(100); /* wait for user event */
                continue;
            } else
                SDL_UnlockMutex(is->general_mutex);
                break;
        }
        if (pkt->stream_index == is->aud_stream_ix) {
            packet_queue_put(&pas->audioq, pkt);
        } else if (pkt->stream_index == is->vid_stream_ix) {
            packet_queue_put(&pvs->videoq, pkt);
        } else if (pkt->stream_index == is->sub_stream_ix) {
            packet_queue_put(&pss->subtitleq, pkt);
        } else {
            av_free_packet(pkt);
        }
        Py_XDECREF((PyObject *) pas);
        Py_XDECREF((PyObject *) pvs);
        Py_XDECREF((PyObject *) pss);
        SDL_UnlockMutex(is->general_mutex);
    }
    
    /* wait until the end */
    while (!is->abort_request) {
        SDL_Delay(100);
    }

    ret = 0;
 fail:
    /* disable interrupting */


    /* close each stream */
    if (is->aud_stream_ix >= 0)
        stream_component_close(is, is->aud_stream_ix);
    if (is->vid_stream_ix >= 0)
        stream_component_close(is, is->vid_stream_ix);
    if (is->sub_stream_ix >= 0)
        stream_component_close(is, is->sub_stream_ix);
    if (is->ic) {
        av_close_input_file(is->ic);
        is->ic = NULL; /* safety */
    }
    url_set_interrupt_cb(NULL);

    if (ret != 0) {
        SDL_Event event;

        event.type = FF_QUIT_EVENT;
        event.user.data1 = is;
        SDL_PushEvent(&event);
    }
    Py_XDECREF((PyObject *)pas);
    Py_XDECREF((PyObject *)pvs);
    Py_XDECREF((PyObject *)pss);
    if(is->loops<0)
    {
        schedule_refresh(is, 40);
        is->parse_tid = SDL_CreateThread(decode_thread, is);
    }
    else if (is->loops>0)
    {   
        schedule_refresh(is, 40);
        is->loops--;
        is->parse_tid = SDL_CreateThread(decode_thread, is);
    }
    Py_DECREF((PyObject *) is);
    return 0;
}

static PyMovie *stream_open(PyMovie *is, const char *filename, AVInputFormat *iformat)
{
    PySys_WriteStdout("Within stream_open\n");

    if (!is)
        return NULL;
    av_strlcpy(is->filename, filename, sizeof(is->filename));
    is->iformat = iformat;
    is->ytop = 0;
    is->xleft = 0;

    /* start video display */
    /*
    is->pictq_mutex = SDL_CreateMutex();
    is->pictq_cond = SDL_CreateCond();

    is->subpq_mutex = SDL_CreateMutex();
    is->subpq_cond = SDL_CreateCond();
    */

    /* add the refresh timer to draw the picture */
    PySys_WriteStdout("stream_open: Before schedule_refesh\n");
    schedule_refresh(is, 40);

    is->av_sync_type = av_sync_type;
    Py_INCREF((PyObject *) is);
    PySys_WriteStdout("stream_open: Before launch of decode_thread\n");
    is->parse_tid = SDL_CreateThread(decode_thread, is);
    PySys_WriteStdout("stream_open: After launch of decode_thread\n");
    if (!is->parse_tid) {
        PyErr_SetString(PyExc_MemoryError, "Could not spawn a new thread.");
        Py_DECREF((PyObject *) is);
        is->ob_type->tp_free((PyObject *) is);
        //PyMem_Free((void *)is);
        return NULL;
    }
    PySys_WriteStdout("stream_open: Returning from within stream_open\n");
    return is;
}


static void stream_close(PyMovie *is)
{

    
    VideoPicture *vp;
    int i;
    /* XXX: use a special url_shutdown call to abort parse cleanly */
    is->abort_request = 1;
    SDL_WaitThread(is->parse_tid, NULL);

    /* free all pictures */
    
    PyVideoStream* pvs;
    PySubtitleStream  * pss;
    PyAudioStream   *pas;
    
    pvs = (PyVideoStream *)PySequence_GetItem(is->streams, (Py_ssize_t)is->video_stream);
    Py_XINCREF((PyObject *)pvs);

    pss = (PySubtitleStream *)PySequence_GetItem(is->streams, (Py_ssize_t)is->subtitle_stream);
    Py_XINCREF((PyObject *)pss);

    pas = (PyAudioStream *)PySequence_GetItem(is->streams, (Py_ssize_t)is->audio_stream);
    Py_XINCREF((PyObject *)pas);
    
    for(i=0;i<VIDEO_PICTURE_QUEUE_SIZE; i++) {
        vp = &pvs->pictq[i];
        if (pvs->bmp) {
            SDL_FreeYUVOverlay(pvs->bmp);
            pvs->bmp = NULL;
        }
        if(pvs->out_surf)
        {
            SDL_FreeSurface(pvs->out_surf);
        }
    }
    if(pvs)
    {
        SDL_DestroyMutex(pvs->pictq_mutex);
        SDL_DestroyCond(pvs->pictq_cond);
    }
    if(pss)
    {    
        SDL_DestroyMutex(pss->subpq_mutex);
        SDL_DestroyCond(pss->subpq_cond);
    }

    Py_XDECREF((PyObject *)pvs);
    Py_XDECREF((PyObject *)pss);
    Py_XDECREF((PyObject *)pas);
    
    if(is->audio_stream)
    {
        PyList_SetItem(is->streams, (Py_ssize_t)is->audio_stream, Py_None);
    }
    if(is->video_stream)
    {
        PyList_SetItem(is->streams, (Py_ssize_t)is->video_stream, Py_None);
    }
    if(is->subtitle_stream)
    {
        PyList_SetItem(is->streams, (Py_ssize_t)is->subtitle_stream, Py_None);
    }
    
    _dealloc_vid_stream(pvs);
    _dealloc_sub_stream(pss);
    _dealloc_aud_stream(pas);
    
    Py_INCREF(is->streams);
    PyObject *pyo=is->streams;
    Py_DECREF(is->streams);
    is->streams =NULL;
    Py_DECREF(pyo);
    //PyMem_Free(pyo);
    
    Py_DECREF((PyObject *)is);

}

static void stream_cycle_channel(PyMovie *is, int codec_type)
{
    AVFormatContext *ic = is->ic;
    int start_index, stream_index;
    AVStream *st;

    if (codec_type == CODEC_TYPE_VIDEO)
        start_index = is->vid_stream_ix;
    else if (codec_type == CODEC_TYPE_AUDIO)
        start_index = is->aud_stream_ix;
    else
        start_index = is->sub_stream_ix;
    if (start_index < (codec_type == CODEC_TYPE_SUBTITLE ? -1 : 0))
        return;
    stream_index = start_index;
    for(;;) {
        if (++stream_index >= is->ic->nb_streams)
        {
            if (codec_type == CODEC_TYPE_SUBTITLE)
            {
                stream_index = -1;
                goto the_end;
            } else
                stream_index = 0;
        }
        if (stream_index == start_index)
            return;
        st = ic->streams[stream_index];
        if (st->codec->codec_type == codec_type) {
            /* check that parameters are OK */
            switch(codec_type) {
            case CODEC_TYPE_AUDIO:
                if (st->codec->sample_rate != 0 &&
                    st->codec->channels != 0)
                    goto the_end;
                break;
            case CODEC_TYPE_VIDEO:
            case CODEC_TYPE_SUBTITLE:
                goto the_end;
            default:
                break;
            }
        }
    }
 the_end:
    stream_component_close(is, start_index);
    stream_component_open(is, stream_index);
}


static PyAudioStream* _new_audio_stream(void)
{
    PyAudioStream *pas;
    pas=(PyAudioStream *)PyMem_Malloc(sizeof(PyAudioStream));
    pas->paused      =0;
    pas->last_paused =0;
    pas->seek_req    =0;
    pas->seek_flags  =0;
    pas->seek_pos    =0;
    
    pas->av_sync_type=AV_SYNC_EXTERNAL_CLOCK;      /* Normally external. */
    pas->offset          =0; 
    pas->frame_timer     =0;
    pas->frame_last_pts  =0;
    pas->frame_last_delay=0;
    pas->frame_offset    =0;
    return pas;
}

static PyVideoStream* _new_video_stream(void)
{
    PyVideoStream *pvs;
    pvs=(PyVideoStream *)PyMem_Malloc(sizeof(PyVideoStream));
    pvs->paused      =0;
    pvs->last_paused =0;
    pvs->seek_req    =0;
    pvs->seek_flags  =0;
    pvs->seek_pos    =0;
    
    pvs->av_sync_type=AV_SYNC_EXTERNAL_CLOCK;      /* Normally external. */
    pvs->offset          =0; 
    pvs->frame_timer     =0;
    pvs->frame_last_pts  =0;
    pvs->frame_last_delay=0;
    pvs->frame_offset    =0;
    return pvs;

}

static PySubtitleStream* _new_sub_stream(void)
{
    PySubtitleStream *pss;
    pss=(PySubtitleStream *)PyMem_Malloc(sizeof(PySubtitleStream));
    pss->paused      =0;
    pss->last_paused =0;
    pss->seek_req    =0;
    pss->seek_flags  =0;
    pss->seek_pos    =0;
    
    pss->av_sync_type=AV_SYNC_EXTERNAL_CLOCK;      /* Normally external. */
    pss->offset          =0; 
    pss->frame_timer     =0;
    pss->frame_last_pts  =0;
    pss->frame_last_delay=0;
    pss->frame_offset    =0;
    return pss;

}

static void _dealloc_aud_stream(PyAudioStream *pas)
{
    pas->ob_type->tp_free((PyObject *) pas);
    //PyMem_Free((void *)pas);
}

static void _dealloc_vid_stream(PyVideoStream *pvs)
{

    if(pvs->out_surf)
    { 
        SDL_FreeSurface(pvs->out_surf);
        pvs->out_surf=NULL;
    }
    if(pvs->bmp)
    {
        SDL_FreeYUVOverlay(pvs->bmp);
        pvs->bmp=NULL;
    }
    pvs->ob_type->tp_free((PyObject *) pvs);
    //PyMem_Free((void *) pvs);
}

static void _dealloc_sub_stream(PySubtitleStream *pss)
{
    pss->ob_type->tp_free((PyObject *) pss);
    //PyMem_Free((void *) pss);
}

/* Python C-API stuff */

static PyMethodDef _movie_methods[] = {
   { "play",    (PyCFunction) _movie_play, METH_VARARGS,
               "Play the movie file from current time-mark. If loop<0, then it will loop infinitely. If there is no loop value, then it will play once." },
   { "stop", (PyCFunction) _movie_stop, METH_NOARGS,
                "Stop the movie, and set time-mark to 0:0"},
   { "pause", (PyCFunction) _movie_pause, METH_NOARGS,
                "Pause movie."},
   { "rewind", (PyCFunction) _movie_rewind, METH_VARARGS,
                "Rewind movie to time_pos. If there is no time_pos, same as stop."},
   { NULL, NULL, 0, NULL }
};

static PyGetSetDef _movie_getsets[] =
{
    { "paused", (getter) _movie_get_paused, NULL, NULL, NULL },
    { "playing", (getter) _movie_get_playing, NULL, NULL, NULL },
    { NULL, NULL, NULL, NULL, NULL }
};

static PyTypeObject PyMovie_Type =
{
    PyObject_HEAD_INIT(NULL)
    0, 
    "pygame.gmovie.Movie",          /* tp_name */
    sizeof (PyMovie),           /* tp_basicsize */
    0,                          /* tp_itemsize */
    (destructor) _movie_dealloc,/* tp_dealloc */
    0,                          /* tp_print */
    0,                          /* tp_getattr */
    0,                          /* tp_setattr */
    0,                          /* tp_compare */
    (reprfunc) _movie_repr,     /* tp_repr */
    0,                          /* tp_as_number */
    0,                          /* tp_as_sequence */
    0,                          /* tp_as_mapping */
    0,                          /* tp_hash */
    0,                          /* tp_call */
    (reprfunc) _movie_str,      /* tp_str */
    0,                          /* tp_getattro */
    0,                          /* tp_setattro */
    0,                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    0,                          /* tp_doc */
    0,                          /* tp_traverse */
    0,                          /* tp_clear */
    0,                          /* tp_richcompare */
    0,                          /* tp_weaklistoffset */
    0,                          /* tp_iter */
    0,                          /* tp_iternext */
    _movie_methods,             /* tp_methods */
    0,                          /* tp_members */
    _movie_getsets,             /* tp_getset */
    0,                          /* tp_base */
    0,                          /* tp_dict */
    0,                          /* tp_descr_get */
    0,                          /* tp_descr_set */
    0,                          /* tp_dictoffset */
    (initproc) _movie_init,                          /* tp_init */
    0,                          /* tp_alloc */
    0,                 /* tp_new */
    0,                          /* tp_free */
    0,                          /* tp_is_gc */
    0,                          /* tp_bases */
    0,                          /* tp_mro */
    0,                          /* tp_cache */
    0,                          /* tp_subclasses */
    0,                          /* tp_weaklist */
    0                           /* tp_del */
};



static PyObject* _movie_init_internal(PyTypeObject *type, char *filename, PyObject* surface)
{
    /*Expects filename. If surface is null, then it sets overlay to >0. */
    PySys_WriteStdout("Within _movie_init_internal\n");    
    //PyMovie *movie  = (PyMovie *)type->tp_alloc (type, 0);
    PyMovie *movie = (PyMovie *)PyMem_Malloc(sizeof(PyMovie));    
    PySys_WriteStdout("_movie_init_internal: after tp->alloc\n");
    if (!movie)
    {
        PyErr_SetString(PyExc_TypeError, "Did not work.");
        Py_RETURN_NONE;
    }
    PySys_WriteStdout("_movie_init_internal: after check for null\n");
    Py_INCREF((PyObject *)movie);

    if(!surface)
    {
        PySys_WriteStdout("_movie_init_internal: Overlay=True\n");
        movie->overlay=1;
    }
    else
    {
        PySys_WriteStdout("_movie_init_internal: Overlay=False\n");
        SDL_Surface *surf;
        surf = PySurface_AsSurface(surface);
        movie->out_surf=surf;
        movie->overlay=0;
    }
    AVInputFormat *iformat;
    movie->general_mutex=SDL_CreateMutex();
    PySys_WriteStdout("_movie_init_internal: Before stream_open with argument: %s\n", filename);
    movie = stream_open(movie, filename, iformat);
    PySys_WriteStdout("_movie_init_internal: After stream_open with argument: %s\n", filename);
    if(!movie)
    {
        PyErr_SetString(PyExc_IOError, "stream_open failed");
        //printf(stdout, "stream_open failed.\n");
        Py_DECREF((PyObject *) movie);
        Py_RETURN_NONE;
    }
    //Py_DECREF((PyObject *) movie);
    PySys_WriteStdout("_movie_init_internal: Returning from _movie_init_internal\n");
    return (PyObject *)movie;
}
    
static int _movie_init (PyTypeObject *type, PyObject *args)
{
    PyObject *obj;
    PyObject *obj2;
    PySys_WriteStdout("Within _movie_init\n");
    if (!PyArg_ParseTuple (args, "s|sO", &obj, &obj2))
    {
        PyErr_SetString(PyExc_TypeError, "No valid arguments");
        Py_RETURN_NONE;
    }
    PySys_WriteStdout("_movie_init: after PyArg_ParseTuple\n");
    
    if(!obj)
    {
        PySys_WriteStdout("_movie_init: No obj found\n");
    }
    
    PyObject *mov;
    char *s;
    s=PyString_AsString(obj);
    PySys_WriteStdout("_movie_init: Before _movie_init_internal\n");
    mov = _movie_init_internal(type, s, obj2);
    PySys_WriteStdout("_movie_init: After _movie_init_internal\n");
    PyObject *er;
    er = PyErr_Occurred();
    if(er)
    {
        PyErr_Print();
    }
    if(!mov)
    {
        PyErr_SetString(PyExc_IOError, "No movie object created.");
        PyErr_Print();
    }
    PySys_WriteStdout("Returning from _movie_init\n");
    return 0;
}   

static void _movie_dealloc(PyMovie *movie)
{
    stream_close(movie);
    movie->ob_type->tp_free((PyObject *) movie);
}

static PyObject* _movie_repr (PyMovie *movie)
{
    /*Eventually add a time-code call */
    char buf[1035];
    PyOS_snprintf(buf, sizeof(buf), "(Movie: %s)", movie->filename);
    return PyString_FromString(buf);
}

static PyObject* _movie_str(PyMovie *movie)
{
    return _movie_repr(movie);
}

static PyObject* _movie_play(PyMovie *movie, PyObject* args)
{
    PyObject *obj;
    int loops;
    PyArg_ParseTuple(args, "i", &obj);
    if(!obj)
    {
        loops =1;
    }
    SDL_LockMutex(movie->general_mutex);
    movie->loops =loops;
    movie->paused = 0;
    movie->playing = 1;
    SDL_UnlockMutex(movie->general_mutex);
    Py_RETURN_NONE;
}

static PyObject* _movie_stop(PyMovie *movie)
{
    SDL_LockMutex(movie->general_mutex);
    stream_pause(movie);
    movie->seek_req = 1;
    movie->seek_pos = 0;
    movie->seek_flags =AVSEEK_FLAG_BACKWARD;
    SDL_UnlockMutex(movie->general_mutex);  
    Py_RETURN_NONE;
}  

static PyObject* _movie_pause(PyMovie *movie)
{
    stream_pause(movie); 
    Py_RETURN_NONE;
}

static PyObject* _movie_rewind(PyMovie *movie, PyObject* args)
{
    /* For now, just alias rewind to stop */
    return _movie_stop(movie);
}

static PyObject* _movie_get_paused (PyMovie *movie, void *closure)
{
    return PyInt_FromLong((long)movie->paused);
}
static PyObject* _movie_get_playing (PyMovie *movie, void *closure)
{
    PyObject *pyo;
    pyo= PyInt_FromLong((long)movie->playing);
    return pyo;
}

static PyObject* PyMovie_New (char *fname, SDL_Surface *surf)
{
    return _movie_new_internal(&PyMovie_Type, fname, PySurface_FromSurface(surf));
}

PyMODINIT_FUNC
initgmovie(void)
{
    PyObject* module;

    /* imported needed apis; Do this first so if there is an error
       the module is not loaded.
    */
    //import_pygame_base ();
    if (PyErr_Occurred ()) {
        MODINIT_ERROR;
    }

    // Fill in some slots in the type, and make it ready
   PyMovie_Type.tp_new = PyType_GenericNew;
   if (PyType_Ready(&PyMovie_Type) < 0) {
      MODINIT_ERROR;
   }

   // Create the module
   
   module = Py_InitModule3 ("gmovie", NULL, "pygame.gmovie plays movies and streams."); //movie doc needed

   if (module == NULL) {
      return;
   }

   
   //Register all the fun stuff for movies.
   avcodec_register_all();
   avdevice_register_all();
   av_register_all();

   av_init_packet(&flush_pkt);
   uint8_t *s = (uint8_t *)"FLUSH";
   flush_pkt.data= s;

   // Add the type to the module.
   Py_INCREF(&PyMovie_Type);
   PyModule_AddObject(module, "Movie", (PyObject*)&PyMovie_Type);
}
