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
		DECLAREGIL
		GRABGIL    	
    	q=(PacketQueue *)PyMem_Malloc(sizeof(PacketQueue));
  		RELEASEGIL
    }
    if(!q->mutex)
	    q->mutex = SDL_CreateMutex();
    if(!q->cond)
    	q->cond = SDL_CreateCond();
    q->abort_request=0;

}

 void packet_queue_flush(PacketQueue *q)
{
	DECLAREGIL
    AVPacketList *pkt, *pkt1;
#if THREADFREE!=1
	if(q->mutex)
		SDL_LockMutex(q->mutex);
#endif    
    for(pkt = q->first_pkt; pkt != NULL; pkt = pkt1) {
        pkt1 = pkt->next;
        av_free_packet(&pkt->pkt);
        GRABGIL
        PyMem_Free(pkt);
    	RELEASEGIL
    }
    q->last_pkt = NULL;
    q->first_pkt = NULL;
    q->nb_packets = 0;
    q->size = 0;
#if THREADFREE!=1
	if(q->mutex)
		SDL_UnlockMutex(q->mutex);	
#endif
}

 void packet_queue_end(PacketQueue *q, int end)
{
    AVPacketList *pkt, *pkt1;

    for(pkt = q->first_pkt; pkt != NULL; pkt = pkt1) {
        pkt1 = pkt->next;
        av_free_packet(&pkt->pkt);
    }
    if(end==0)
    {
	    if(q->mutex)
	    {
		    SDL_DestroyMutex(q->mutex);
	    }
   		if(q->cond)
   		{
	    	SDL_DestroyCond(q->cond);
   		}
    }
}

 int packet_queue_put(PacketQueue *q, AVPacket *pkt)
{
    AVPacketList *pkt1;

	DECLAREGIL
	GRABGIL
    pkt1 = PyMem_Malloc(sizeof(AVPacketList));
    
    if (!pkt1)
        return -1;
    pkt1->pkt = *pkt;
    pkt1->next = NULL;

#if THREADFREE!=1
	if(q->mutex)
		SDL_LockMutex(q->mutex);	
#endif
    RELEASEGIL //Grab GIL first, then mutex.
    if (!q->last_pkt)

        q->first_pkt = pkt1;
    else
        q->last_pkt->next = pkt1;
    q->last_pkt = pkt1;
    q->nb_packets++;
    q->size += pkt1->pkt.size;
    /* XXX: should duplicate packet data in DV case */

#if THREADFREE!=1
	if(q->mutex)
		SDL_UnlockMutex(q->mutex);	
#endif
    return 0;
}

void packet_queue_abort(PacketQueue *q)
{
#if THREADFREE!=1
	if(q->mutex)
		SDL_LockMutex(q->mutex);	
#endif
    q->abort_request = 1;
#if THREADFREE!=1
	if(q->mutex)
		SDL_UnlockMutex(q->mutex);	
#endif
}

/* return < 0 if aborted, 0 if no packet and > 0 if packet.  */
int packet_queue_get(PacketQueue *q, AVPacket *pkt, int block)
{
	DECLAREGIL
	GRABGIL
    AVPacketList *pkt1;
    int ret;
    
#if THREADFREE!=1
	if(q->mutex)
		SDL_LockMutex(q->mutex);
#endif
	RELEASEGIL
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
            q->size -= pkt1->pkt.size;
            *pkt = pkt1->pkt;
            GRABGIL
            PyMem_Free(pkt1);
            RELEASEGIL
            ret = 1;
            break;
        } else if (!block) {
            ret = 0;
            break;
        } else {
            ret=0;
            break;
        }
    }
#if THREADFREE!=1
	if(q->mutex)
		SDL_UnlockMutex(q->mutex);	
#endif
    return ret;
}


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

int video_display(PyMovie *movie)
{
/*DECODE THREAD - from video_refresh_timer*/
	DECLAREGIL
	GRABGIL
	Py_INCREF(movie);
	double ret=1;
#if THREADFREE!=1
	SDL_LockMutex(movie->dest_mutex);
#endif
	VidPicture *vp = &movie->pictq[movie->pictq_rindex];
    RELEASEGIL
    if(!vp->dest_overlay)
    {
    	video_open(movie, movie->pictq_rindex);
    	ret=0;
    }
    else if (movie->video_stream>=0 && vp->ready)
    {
        video_image_display(movie);
    }
    else if(!vp->ready)
    {
    	ret=0;
    }
#if THREADFREE !=1
	SDL_UnlockMutex(movie->dest_mutex);
#endif
	GRABGIL
	Py_DECREF(movie);
	RELEASEGIL
	return ret;
}

void video_image_display(PyMovie *is)
{
	/* Wrapped by video_display, which has a lock on the movie object */
    DECLAREGIL
    GRABGIL
    Py_INCREF( is);
    RELEASEGIL
    SubPicture *sp;
    VidPicture *vp;
    AVPicture pict;
    float aspect_ratio;
    int width, height, x, y;
    
    int i;
    vp = &is->pictq[is->pictq_rindex];
    vp->ready =0;
    if (vp->dest_overlay) {
        /* XXX: use variable in the frame */
        if (is->video_st->sample_aspect_ratio.num)
            aspect_ratio = av_q2d(is->video_st->sample_aspect_ratio);
        else if (is->video_st->codec->sample_aspect_ratio.num)
            aspect_ratio = av_q2d(is->video_st->codec->sample_aspect_ratio);
        else
            aspect_ratio = 0;
        if (aspect_ratio <= 0.0)
            aspect_ratio = 1.0;
        aspect_ratio *= (float)is->video_st->codec->width / is->video_st->codec->height;
        /* if an active format is indicated, then it overrides the
           mpeg format */
        
        if (is->subtitle_stream>-1)
        {
            
            if (is->subpq_size > 0)
            {
                sp = &is->subpq[is->subpq_rindex];

                if (is->pts >= is->pts + ((float) sp->sub.start_display_time / 1000))
                {

                    if(is->overlay>0)
                    {
                        SDL_LockYUVOverlay (is->dest_overlay);

                        pict.data[0] = is->dest_overlay->pixels[0];
                        pict.data[1] = is->dest_overlay->pixels[2];
                        pict.data[2] = is->dest_overlay->pixels[1];

                        pict.linesize[0] = is->dest_overlay->pitches[0];
                        pict.linesize[1] = is->dest_overlay->pitches[2];
                        pict.linesize[2] = is->dest_overlay->pitches[1];

                        for (i = 0; i < sp->sub.num_rects; i++)
                            blend_subrect(&pict, sp->sub.rects[i],
                                          vp->dest_overlay->w, vp->dest_overlay->h);

                        SDL_UnlockYUVOverlay (is->dest_overlay);
                    }
                }
            }
        }


        /* XXX: we suppose the screen has a 1.0 pixel ratio */
        height = vp->height;
        width = ((int)rint(height * aspect_ratio)) & ~1;
        if (width > vp->width) {
            width = vp->width;
            height = ((int)rint(width / aspect_ratio)) & ~1;
        }
        x = (vp->width - width) / 2;
        y = (vp->height - height) / 2;
       
        vp->dest_rect.x = vp->xleft + x;
        vp->dest_rect.y = vp->ytop  + y;
        vp->dest_rect.w = width;
        vp->dest_rect.h = height;
        if(vp->overlay>0) 
        {      
        	SDL_LockYUVOverlay(vp->dest_overlay); 
            SDL_DisplayYUVOverlay(vp->dest_overlay, &vp->dest_rect);
        	SDL_UnlockYUVOverlay(vp->dest_overlay);
        }
        
    } 
    is->pictq_rindex= (is->pictq_rindex+1)%VIDEO_PICTURE_QUEUE_SIZE;
    is->pictq_size--;
    video_refresh_timer(is);
    GRABGIL
    Py_DECREF( is);
	RELEASEGIL
}

int video_open(PyMovie *is, int index){
    int w,h;
    DECLAREGIL
    GRABGIL
    Py_INCREF( is);
    RELEASEGIL
    w = is->video_st->codec->width;
    h = is->video_st->codec->height;

	VidPicture *vp;
	vp = &is->pictq[index];
    
    if(!vp->dest_overlay && is->overlay>0)
    {
        //now we have to open an overlay up
        SDL_Surface *screen;
        if (!SDL_WasInit (SDL_INIT_VIDEO))
        {
        	GRABGIL
        	RAISE(PyExc_SDLError,"cannot create overlay without pygame.display initialized");
        	Py_DECREF(is);
        	RELEASEGIL
        	return -1;
        }
        screen = SDL_GetVideoSurface ();
        if (!screen)
		{
			screen = SDL_SetVideoMode(w, h, 0, SDL_SWSURFACE);
			if(!screen)
			{
				GRABGIL
	        	RAISE(PyExc_SDLError, "Could not initialize a new video surface.");
	        	Py_DECREF(is);
	        	RELEASEGIL
	        	return -1;	
			}
		}
        vp->dest_overlay = SDL_CreateYUVOverlay (w, h, SDL_YV12_OVERLAY, screen);
        if (!vp->dest_overlay)
        {
        	GRABGIL
            RAISE (PyExc_SDLError, "Cannot create overlay");
			Py_DECREF(is);
			RELEASEGIL
			return -1;
        }
        vp->overlay = is->overlay;
    } 

    is->width = w;
    vp->width = w;
    is->height = h;
    vp->height = h;
	GRABGIL
    Py_DECREF( is);
    RELEASEGIL
    return 0;
}

/* called to display each frame */
 void video_refresh_timer(PyMovie* movie)
{
/*moving to DECODE THREAD, from queue_frame*/
	DECLAREGIL
	GRABGIL
	Py_INCREF(movie);
    RELEASEGIL
    double actual_delay, delay, sync_threshold, ref_clock, diff;
	VidPicture *vp;
	
    if (movie->video_st) { /*shouldn't ever even get this far if no video_st*/

        /* dequeue the picture */
        vp = &movie->pictq[movie->pictq_rindex];

        /* update current video pts */
        movie->video_current_pts = vp->pts;
        movie->video_current_pts_time = av_gettime();
		
	    /* compute nominal delay */
	    delay = movie->video_current_pts - movie->frame_last_pts;
	    if (delay <= 0 || delay >= 10.0) {
	        /* if incorrect delay, use previous one */
	        delay = movie->frame_last_delay;
	    } else {
	        movie->frame_last_delay = delay;
	    }
	    movie->frame_last_pts = movie->video_current_pts;
	
	    /* update delay to follow master synchronisation source */
	    if (((movie->av_sync_type == AV_SYNC_AUDIO_MASTER && movie->audio_st) ||
	         movie->av_sync_type == AV_SYNC_EXTERNAL_CLOCK)) {
	        /* if video is slave, we try to correct big delays by
	           duplicating or deleting a frame */
	        ref_clock = get_master_clock(movie);
	        diff = movie->video_current_pts - ref_clock;
	
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
	
	    movie->frame_timer += delay;
	    /* compute the REAL delay (we need to do that to avoid
	       long term errors */
	    actual_delay = movie->frame_timer - (av_gettime() / 1000000.0);
	    if (actual_delay < 0.010) {
	        /* XXX: should skip picture */
	        actual_delay = 0.010;
	    }
	    GRABGIL
		movie->timing = (actual_delay*1000.0)+0.5;
    	RELEASEGIL
    }
    GRABGIL
    Py_DECREF(movie);
	RELEASEGIL
}

 int queue_picture(PyMovie *movie, AVFrame *src_frame)
{
/*Video Thread LOOP*/
	DECLAREGIL
	GRABGIL
	Py_INCREF(movie);
    RELEASEGIL
    int dst_pix_fmt;
    AVPicture pict;
    VidPicture *vp;
    struct SwsContext *img_convert_ctx=movie->img_convert_ctx;

	vp = &movie->pictq[movie->pictq_windex];
	
	if(!vp->dest_overlay)
	{
		video_open(movie, movie->pictq_windex);
	}	
    if (vp->dest_overlay) {
        /* get a pointer on the bitmap */
        
        dst_pix_fmt = PIX_FMT_YUV420P;
           
        SDL_LockYUVOverlay(vp->dest_overlay);

        pict.data[0] = vp->dest_overlay->pixels[0];
        pict.data[1] = vp->dest_overlay->pixels[2];
        pict.data[2] = vp->dest_overlay->pixels[1];       
        pict.linesize[0] = vp->dest_overlay->pitches[0];
        pict.linesize[1] = vp->dest_overlay->pitches[2];
        pict.linesize[2] = vp->dest_overlay->pitches[1];

		int sws_flags = SWS_BICUBIC;
        img_convert_ctx = sws_getCachedContext(img_convert_ctx,
            movie->video_st->codec->width, movie->video_st->codec->height,
            movie->video_st->codec->pix_fmt,
            movie->video_st->codec->width, movie->video_st->codec->height,
            dst_pix_fmt, sws_flags, NULL, NULL, NULL);
        if (img_convert_ctx == NULL) {
            fprintf(stderr, "Cannot initialize the conversion context\n");
            exit(1);
        }
        movie->img_convert_ctx = img_convert_ctx;
        sws_scale(img_convert_ctx, src_frame->data, src_frame->linesize,
                  0, movie->video_st->codec->height, pict.data, pict.linesize);

        SDL_UnlockYUVOverlay(vp->dest_overlay);

        vp->pts = movie->pts;  
    	movie->pictq_windex = (movie->pictq_windex+1)%VIDEO_PICTURE_QUEUE_SIZE;
		movie->pictq_size++;
		vp->ready=1;
    }
	GRABGIL
	Py_DECREF(movie);
    RELEASEGIL
    return 0;
}


 void update_video_clock(PyMovie *movie, AVFrame* frame, double pts1) {
	DECLAREGIL
	GRABGIL
	Py_INCREF(movie);
	RELEASEGIL
	double frame_delay, pts;

    pts = pts1;

    if (pts != 0) {
        /* update video clock with pts, if present */
        movie->video_clock = pts;
    } else {
        pts = movie->video_clock;
    }
    /* update video clock for next frame */
    frame_delay = av_q2d(movie->video_st->codec->time_base);
    /* for MPEG2, the frame can be repeated, so we update the
       clock accordingly */
    frame_delay += frame->repeat_pict * (frame_delay * 0.5);
    movie->video_clock += frame_delay;

	movie->pts = pts;
	GRABGIL
	Py_DECREF(movie);
	RELEASEGIL
}

 int audio_write_get_buf_size(PyMovie *movie)
{
	DECLAREGIL
	GRABGIL
    Py_INCREF(movie);
    RELEASEGIL
    
    int temp = movie->audio_buf_size - movie->audio_buf_index;
   	GRABGIL
   	Py_DECREF(movie);
    RELEASEGIL
    return temp;
}

/* get the current audio clock value */
 double get_audio_clock(PyMovie *is)
{
	DECLAREGIL
	GRABGIL
    Py_INCREF( is);
    RELEASEGIL
    double pts;
    int hw_buf_size, bytes_per_sec;

    
    pts = is->audio_clock;
    hw_buf_size = audio_write_get_buf_size(is);
    bytes_per_sec = 0;
    if (is->audio_st) {
        bytes_per_sec = is->audio_st->codec->sample_rate *
            2 * is->audio_st->codec->channels;
    }
    if (bytes_per_sec)
        pts -= (double)hw_buf_size / bytes_per_sec;
    GRABGIL
    Py_DECREF( is);
    RELEASEGIL
    return pts;
}

/* get the current video clock value */
 double get_video_clock(PyMovie *is)
{
    Py_INCREF( is);
    double delta;
    
    if (is->paused) {
        delta = 0;
    } else {
        delta = (av_gettime() - is->video_current_pts_time) / 1000000.0;
    }
    double temp = is->video_current_pts+delta;
    Py_DECREF( is);
    return temp;
}

/* get the current external clock value */
 double get_external_clock(PyMovie *is)
{
	DECLAREGIL
	GRABGIL
    Py_INCREF( is);
    RELEASEGIL
    int64_t ti;
    ti = av_gettime();
    double res = is->external_clock + ((ti - is->external_clock_time) * 1e-6);
    GRABGIL
    Py_DECREF( is);
	RELEASEGIL
    return res;
}

/* get the current master clock value */
 double get_master_clock(PyMovie *is)
{
	DECLAREGIL
	GRABGIL
    Py_INCREF( is);
    RELEASEGIL
    double val;
    
    if (is->av_sync_type == AV_SYNC_VIDEO_MASTER) {
        if (is->video_st)
            val = get_video_clock(is);
        else
            val = get_audio_clock(is);
    } else if (is->av_sync_type == AV_SYNC_AUDIO_MASTER) {
        if (is->audio_st)
            val = get_audio_clock(is);
        else
            val = get_video_clock(is);
    } else {
        val = get_external_clock(is);
    }
    GRABGIL
    Py_DECREF( is);
	RELEASEGIL
    return val;
}

/* seek in the stream */
 void stream_seek(PyMovie *is, int64_t pos, int rel)
{
	DECLAREGIL
	GRABGIL
    Py_INCREF( is);
    RELEASEGIL
    if (!is->seek_req) {
        is->seek_pos = pos;
        is->seek_flags = rel < 0 ? AVSEEK_FLAG_BACKWARD : 0;

        is->seek_req = 1;
    }
    GRABGIL
    Py_DECREF( is);
	RELEASEGIL
}

/* pause or resume the video */
void stream_pause(PyMovie *is)
{
	DECLAREGIL
	GRABGIL
    Py_INCREF( is);
    RELEASEGIL
    is->paused = !is->paused;
    if (!is->paused) {

       is->video_current_pts = get_video_clock(is);
        
        is->frame_timer += (av_gettime() - is->video_current_pts_time) / 1000000.0;
    }
    GRABGIL
    Py_DECREF( is);
	RELEASEGIL
}


int subtitle_thread(void *arg)
{
	DECLAREGIL
    PyMovie *movie = arg;
    GRABGIL
    Py_INCREF( movie);
	RELEASEGIL
	    
    SubPicture *sp;
    AVPacket pkt1, *pkt = &pkt1;
    int len1, got_subtitle;
    double pts;
    int i, j;
    int r, g, b, y, u, v, a;

    
    for(;;) {
        while (movie->paused && !movie->subtitleq.abort_request) {
            SDL_Delay(10);
	    }
        if (packet_queue_get(&movie->subtitleq, pkt, 1) < 0)
        {    
            break;
        }
        if(pkt->data == flush_pkt.data){
            avcodec_flush_buffers(movie->subtitle_st->codec);
            continue;
        }
        SDL_LockMutex(movie->subpq_mutex);
        while (movie->subpq_size >= SUBPICTURE_QUEUE_SIZE &&
               !movie->subtitleq.abort_request) {
            SDL_CondWait(movie->subpq_cond, movie->subpq_mutex);
        }
        SDL_UnlockMutex(movie->subpq_mutex);

        if (movie->subtitleq.abort_request)
        {
            goto the_end;
        }
        sp = &movie->subpq[movie->subpq_windex];

       /* NOTE: ipts is the PTS of the _first_ picture beginning in
           this packet, if any */
        pts = 0;
        if (pkt->pts != AV_NOPTS_VALUE)
            pts = av_q2d(movie->subtitle_st->time_base)*pkt->pts;

        len1 = avcodec_decode_subtitle(movie->subtitle_st->codec,
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
            if (++movie->subpq_windex == SUBPICTURE_QUEUE_SIZE)
                movie->subpq_windex = 0;
            SDL_LockMutex(movie->subpq_mutex);
            movie->subpq_size++;
            SDL_UnlockMutex(movie->subpq_mutex);
        }
        av_free_packet(pkt);
//        if (step)
//            if (cur_stream)
//                stream_pause(cur_stream);
	
    }
 
 the_end:
	GRABGIL
    Py_DECREF( movie);
	RELEASEGIL
    return 0;
}


/* return the new audio buffer size (samples can be added or deleted
   to get better sync if video or external master clock) */
 int synchronize_audio(PyMovie *is, short *samples,
                             int samples_size1, double pts)
{
	
    Py_INCREF( is);
    
    int n, samples_size;
    double ref_clock;


    n = 2 * is->audio_st->codec->channels;
    samples_size = samples_size1;

    /* if not master, then we try to remove or add samples to correct the clock */
    if (((is->av_sync_type == AV_SYNC_VIDEO_MASTER && is->video_st) ||
         is->av_sync_type == AV_SYNC_EXTERNAL_CLOCK)) {
        double diff, avg_diff;
        int wanted_size, min_size, max_size, nb_samples;

        ref_clock = get_master_clock(is);
        diff = get_audio_clock(is) - ref_clock;

        if (diff < AV_NOSYNC_THRESHOLD) {
            is->audio_diff_cum = diff + is->audio_diff_avg_coef * is->audio_diff_cum;
            if (is->audio_diff_avg_count < AUDIO_DIFF_AVG_NB) {
                /* not enough measures to have a correct estimate */
                is->audio_diff_avg_count++;
            } else {
                /* estimate the A-V difference */
                avg_diff = is->audio_diff_cum * (1.0 - is->audio_diff_avg_coef);

                if (fabs(avg_diff) >= is->audio_diff_threshold) {
                    wanted_size = samples_size + ((int)(diff * is->audio_st->codec->sample_rate) * n);
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
            }
        } else {
            /* too big difference : may be initial PTS errors, so
               reset A-V filter */
            is->audio_diff_avg_count = 0;
            is->audio_diff_cum = 0;
        }
    }
    Py_DECREF( is);
    return samples_size;
}




int audio_thread(void *arg)
{
	PyMovie *movie = arg;
	DECLAREGIL
	GRABGIL
	Py_INCREF(movie);
	//PySys_WriteStdout("Inside audio_thread\n");
	RELEASEGIL
    double pts;
	AVPacket *pkt = &movie->audio_pkt;
    AVCodecContext *dec= movie->audio_st->codec;
    int n, len1, data_size;	
	int filled =0;
	len1=0;
	int co = 0;
	for(;co<10;co++)
	{
		/*GRABGIL
		PySys_WriteStdout("audio_thread: infinite looping(%i)...\n", co);
		RELEASEGIL*/
		//fill up the buffer
		while(movie->audio_pkt_size > 0)
        {
        	GRABGIL
        	PySys_WriteStdout("audio_thread: filling up the buffer...\n");
			RELEASEGIL
			data_size = sizeof(movie->audio_buf1);
            len1 += avcodec_decode_audio2(dec, (int16_t *)movie->audio_buf1, &data_size, movie->audio_pkt_data, movie->audio_pkt_size);
            if (len1 < 0) {
                /* if error, we skip the frame */
                movie->audio_pkt_size = 0;
                break;
            }

            movie->audio_pkt_data += len1;
            movie->audio_pkt_size -= len1;
            if (data_size <= 0)
                continue;
            //reformat_ctx here, but deleted    
            /* if no pts, then compute it */
            pts = movie->audio_clock;
            n = 2 * dec->channels;
            movie->audio_clock += (double)data_size / (double)(n * dec->sample_rate);
            filled=1;
        	   
        }
        //either buffer filled or no packets yet
        /* free the current packet */
        if (pkt->data)
            av_free_packet(pkt);

        //if (movie->paused) {
        //    continue;
        //}
        //check if the movie has ended
		if(movie->stop)
			goto closing;
        /* read next packet */
        if (packet_queue_get(&movie->audioq, pkt, 1) < 0)
        {
			SDL_Delay(10);         
            continue;
        }
        if(pkt->data == flush_pkt.data){
            avcodec_flush_buffers(dec);
            continue;
        }

        movie->audio_pkt_data = pkt->data;
        movie->audio_pkt_size = pkt->size;
        
        /* if update the audio clock with the pts */
        if (pkt->pts != AV_NOPTS_VALUE) {
            movie->audio_clock = av_q2d(movie->audio_st->time_base)*pkt->pts;
        }
        if(filled)
        {
        	/* Buffer is filled up with a new frame, we spin lock/wait for a signal, where we then call playBuffer */
        	SDL_LockMutex(movie->audio_mutex);
        	//SDL_CondWait(movie->audio_sig, movie->audio_mutex);
        	playBuffer(movie->audio_buf1, len1);
        	filled=0;
        	len1=0;
        	SDL_UnlockMutex(movie->audio_mutex);
        	goto closing;
        }
        else
        {
        	filled=0;
        }

    }
closing:
    GRABGIL
	Py_DECREF(movie);
	RELEASEGIL
	return 0;
}

/* open a given stream. Return 0 if OK */
 int stream_component_open(PyMovie *movie, int stream_index)
{
    DECLAREGIL
    GRABGIL
    Py_INCREF( movie);
    RELEASEGIL
    AVFormatContext *ic = movie->ic;
    AVCodecContext *enc;
    AVCodec *codec;
	int freq, channels;
    if (stream_index < 0 || stream_index >= ic->nb_streams)
    {
    	GRABGIL
    	Py_DECREF(movie);
        RELEASEGIL
        return -1;
    }
    
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
    enc->debug_mv = 0;
    enc->debug = 0;
    enc->workaround_bugs = 1;
    enc->lowres = 0;
    enc->idct_algo= FF_IDCT_AUTO;
    if(0)enc->flags2 |= CODEC_FLAG2_FAST;
    enc->skip_frame= AVDISCARD_DEFAULT;
    enc->skip_idct= AVDISCARD_DEFAULT;
    enc->skip_loop_filter= AVDISCARD_DEFAULT;
    enc->error_recognition= FF_ER_CAREFUL;
    enc->error_concealment= 3;


	//TODO:proper error reporting here please
    if (avcodec_open(enc, codec) < 0)
    {
    	GRABGIL
    	Py_DECREF(movie);
    	RELEASEGIL
        return -1;
    }
    /* prepare audio output */
    if (enc->codec_type == CODEC_TYPE_AUDIO) {
        
        freq = enc->sample_rate;
        channels = enc->channels;
        if (soundInit  (freq, (int)NULL, channels, 1024, movie->audio_sig) < 0) {
            RAISE(PyExc_SDLError, SDL_GetError ());
        }
        movie->audio_hw_buf_size = 1024;
        movie->audio_src_fmt= AUDIO_S16SYS;
    }

    enc->thread_count= 1;
    ic->streams[stream_index]->discard = AVDISCARD_DEFAULT;
    
    switch(enc->codec_type) {
    case CODEC_TYPE_AUDIO:
        movie->audio_stream = stream_index;
        movie->audio_st = ic->streams[stream_index];
        movie->audio_buf_size = 0;
        movie->audio_buf_index = 0;

        /* init averaging filter */
        movie->audio_diff_avg_coef = exp(log(0.01) / AUDIO_DIFF_AVG_NB);
        movie->audio_diff_avg_count = 0;
        /* since we do not have a precmoviee anough audio fifo fullness,
           we correct audio sync only if larger than thmovie threshold */
        movie->audio_diff_threshold = 2.0 * SDL_AUDIO_BUFFER_SIZE / enc->sample_rate;
        
        memset(&movie->audio_pkt, 0, sizeof(movie->audio_pkt));
        packet_queue_init(&movie->audioq);
		movie->audio_sig = SDL_CreateCond();
		movie->audio_mutex = SDL_CreateMutex();
		//movie->audio_tid = SDL_CreateThread(audio_thread, movie);
        
        break;
    case CODEC_TYPE_VIDEO:
        movie->video_stream = stream_index;
        movie->video_st = ic->streams[stream_index];

        movie->frame_last_delay = 40e-3;
        movie->frame_timer = (double)av_gettime() / 1000000.0;
        movie->video_current_pts_time = av_gettime();
        movie->video_last_pts_time=av_gettime();

        packet_queue_init(&movie->videoq);
      	break;
    case CODEC_TYPE_SUBTITLE:
        movie->subtitle_stream = stream_index;
        
        movie->subtitle_st = ic->streams[stream_index];
        packet_queue_init(&movie->subtitleq);

        movie->subtitle_tid = SDL_CreateThread(subtitle_thread, movie);
        break;
    default:
        break;
    }
    GRABGIL
    Py_DECREF( movie);
    RELEASEGIL
    return 0;
}

void stream_component_close(PyMovie *is, int stream_index)
{
	DECLAREGIL
	GRABGIL
    if(is->ob_refcnt!=0)Py_INCREF( is);
	RELEASEGIL
    AVFormatContext *ic = is->ic;
    AVCodecContext *enc;

    if (stream_index < 0 || stream_index >= ic->nb_streams)
    {
    	GRABGIL
    	if(is->ob_refcnt!=0)
    	{
    		Py_DECREF(is);
    	}
    	RELEASEGIL
        return;
    }
    enc = ic->streams[stream_index]->codec;
	int end = is->loops;
    switch(enc->codec_type) {
    case CODEC_TYPE_AUDIO:
        packet_queue_abort(&is->audioq);
        soundQuit();
        SDL_WaitThread(is->audio_tid, NULL);
        packet_queue_end(&is->audioq, end);
        if (is->reformat_ctx)
            av_audio_convert_free(is->reformat_ctx);
        break;
    case CODEC_TYPE_VIDEO:
        packet_queue_abort(&is->videoq);
        SDL_WaitThread(is->video_tid, NULL);
        packet_queue_end(&is->videoq, end);
        break;
    case CODEC_TYPE_SUBTITLE:
        packet_queue_abort(&is->subtitleq);
        /* note: we also signal this mutex to make sure we deblock the
           video thread in all cases */
        SDL_LockMutex(is->subpq_mutex);
        is->subtitle_stream_changed = 1;
        SDL_CondSignal(is->subpq_cond);
        SDL_UnlockMutex(is->subpq_mutex);
        SDL_WaitThread(is->subtitle_tid, NULL);
        packet_queue_end(&is->subtitleq, end);
        break;
    default:
        break;
    }

    ic->streams[stream_index]->discard = AVDISCARD_ALL;
    avcodec_close(enc);
    switch(enc->codec_type) {
    case CODEC_TYPE_AUDIO:
        is->audio_st = NULL;
        is->audio_stream = -1;
        break;
    case CODEC_TYPE_VIDEO:
        is->video_st = NULL;
        is->video_stream = -1;
        break;
    case CODEC_TYPE_SUBTITLE:
        is->subtitle_st = NULL;
        is->subtitle_stream = -1;
        break;
    default:
        break;
    }

	GRABGIL
    if(is->ob_refcnt!=0)
    {
    	Py_DECREF( is);
    }
	RELEASEGIL
}

PyMovie *stream_open(PyMovie *is, const char *filename, AVInputFormat *iformat)
{
    if (!is)
        return NULL;
    DECLAREGIL
    GRABGIL
    Py_INCREF(is);
	RELEASEGIL
	AVFormatContext *ic;
    int err, i, ret, video_index, audio_index, subtitle_index;
    AVFormatParameters params, *ap = &params;
    
	is->overlay=1;
    av_strlcpy(is->filename, filename, strlen(filename)+1);
    is->iformat = iformat;

    /* start video display */
    is->dest_mutex = SDL_CreateMutex();
	
    is->subpq_mutex = SDL_CreateMutex();
    //is->subpq_cond = SDL_CreateCond();
    
    is->paused = 1;
    is->av_sync_type = AV_SYNC_VIDEO_MASTER;
	
    video_index = -1;
    audio_index = -1;
    subtitle_index = -1;
    is->video_stream = -1;
    is->audio_stream = -1;
    is->subtitle_stream = -1;

    int wanted_video_stream=1;
    int wanted_audio_stream=1;
    memset(ap, 0, sizeof(*ap));
    ap->width = 0;
    ap->height= 0;
    ap->time_base= (AVRational){1, 25};
    ap->pix_fmt = PIX_FMT_NONE;
	
    err = av_open_input_file(&ic, is->filename, is->iformat, 0, ap);
    if (err < 0) {
    	GRABGIL
        PyErr_Format(PyExc_IOError, "There was a problem opening up %s", is->filename);
        RELEASEGIL
        ret = -1;
        goto fail;
    }
    err = av_find_stream_info(ic);
    if (err < 0) {
    	GRABGIL
        PyErr_Format(PyExc_IOError, "%s: could not find codec parameters", is->filename);
        RELEASEGIL
        ret = -1;
        goto fail;
    }
    if(ic->pb)
        ic->pb->eof_reached= 0; //FIXME hack, ffplay maybe should not use url_feof() to test for the end
	

	is->ic = ic;
    /* if seeking requested, we execute it */
    if (is->start_time != AV_NOPTS_VALUE) {
        int64_t timestamp;

        timestamp = is->start_time;
        /* add the stream start time */
        if (ic->start_time != AV_NOPTS_VALUE)
            timestamp += ic->start_time;
        ret = av_seek_frame(ic, -1, timestamp, AVSEEK_FLAG_BACKWARD);
        if (ret < 0) {
        	GRABGIL
            PyErr_Format(PyExc_IOError, "%s: could not seek to position %0.3f", is->filename, (double)timestamp/AV_TIME_BASE);
        	RELEASEGIL
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
            //if (wanted_subtitle_stream-- >= 0 && !video_disable)
            //    subtitle_index = i;
            break;
        default:
            break;
        }
    }

    /* open the streams */
    /*if (audio_index >= 0) {
		stream_component_open(is, audio_index);
   	}*/
	
    if (video_index >= 0) {
    	stream_component_open(is, video_index);
    } 

/*    if (subtitle_index >= 0) {
        stream_component_open(is, subtitle_index);
    }*/
    if (is->video_stream < 0 && is->audio_stream < 0) {
    	GRABGIL
        PyErr_Format(PyExc_IOError, "%s: could not open codecs", is->filename);
        RELEASEGIL
        ret = -1;		
		goto fail;
    }
    
	is->frame_delay = av_q2d(is->video_st->codec->time_base);
    
   ret = 0;
 fail:
    /* disable interrupting */

	//if(ap)
	//	av_freep(params);
	if(ret!=0)
	{
		GRABGIL
		//throw python error
		PyObject *er;
		er=PyErr_Occurred();
		if(er)
		{
			PyErr_Print();
		}
		Py_DECREF(is);
		RELEASEGIL
		return is;
	}
	
	GRABGIL
	Py_DECREF(is);
    RELEASEGIL
    return is;
}

 void stream_close(PyMovie *is)
{
	DECLAREGIL
	GRABGIL
	if(is->ob_refcnt!=0) Py_INCREF(is);
    RELEASEGIL
    is->abort_request = 1;
    SDL_WaitThread(is->parse_tid, NULL);
	VidPicture *vp;
    
    if(is)
    {
		int i;    	
    		for( i =0; i<VIDEO_PICTURE_QUEUE_SIZE; i++)
    		{
    			vp = &is->pictq[i];
    			if (vp->dest_overlay) {
            		SDL_FreeYUVOverlay(vp->dest_overlay);
            		vp->dest_overlay = NULL;
        		}
        		if(vp->dest_surface)
        		{
            		SDL_FreeSurface(vp->dest_surface);
        			vp->dest_surface=NULL;
        		}
    		}
		SDL_DestroyMutex(is->dest_mutex);
     	SDL_DestroyMutex(is->subpq_mutex);
        //SDL_DestroyCond(is->subpq_cond);
    	if(is->img_convert_ctx)
    	{
    		sws_freeContext(is->img_convert_ctx);
    		is->img_convert_ctx=NULL;
    	}
    }
    /* close each stream */
    if (is->audio_stream >= 0)
    {
        stream_component_close(is, is->audio_stream);
    }
    if (is->video_stream >= 0)
    {
        stream_component_close(is, is->video_stream);
    }
    if (is->subtitle_stream >= 0)
    {
        stream_component_close(is, is->subtitle_stream);
    }
    if (is->ic) {
        av_close_input_file(is->ic);
        is->ic = NULL; /* safety */
    }
    
    if(is->ob_refcnt!=0)
    { 
    	GRABGIL
    	Py_DECREF(is);
    	RELEASEGIL
    }
}

void stream_cycle_channel(PyMovie *is, int codec_type)
{
    AVFormatContext *ic = is->ic;
    int start_index, stream_index;
    AVStream *st;
	DECLAREGIL
	GRABGIL
	Py_INCREF(is);
	RELEASEGIL
	
    if (codec_type == CODEC_TYPE_VIDEO)
        start_index = is->video_stream;
    else if (codec_type == CODEC_TYPE_AUDIO)
        start_index = is->audio_stream;
    else
        start_index = is->subtitle_stream;
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
    GRABGIL
    Py_DECREF(is);
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
	while((movie->loops>-1||eternity) )
	{
		GRABGIL
		PySys_WriteStdout("Loops: %i\n", movie->loops);
		RELEASEGIL
		movie->loops--;
		movie=stream_open(movie, movie->filename, NULL);
		/*if(movie->audio_st)
			movie->audio_tid = SDL_CreateThread(audio_thread, movie);*/
		movie->paused=0;
		state =decoder(movie);
		stream_component_close(movie, movie->video_st->index);
		//stream_component_close(movie, movie->audio_st->index);
	}
	if(gstate==PyGILState_LOCKED) RELEASEGIL	
	return state;
}

/* this thread gets the stream from the disk or the network */
 int decoder(void *arg)
{
	PyMovie *is = arg;
	DECLAREGIL
	GRABGIL
    Py_INCREF( is);
    RELEASEGIL
    AVFormatContext *ic;
    int ret;
    AVPacket pkt1, *pkt = &pkt1;
	is->stop=0;
	ic=is->ic;
	int co=0;
	is->last_showtime = av_gettime()/1000.0;
	RELEASEGIL
    //video_open(is, is->pictq_windex);
    for(;;) {
		//PySys_WriteStdout("decoder: loop %i.\n", co);
		co++;
		
        if (is->abort_request)
        { 
            break;
        }
        if (is->paused != is->last_paused) {
            is->last_paused = is->paused;
            if (is->paused)
                av_read_pause(ic);
            else
                av_read_play(ic);
        }
        if (is->seek_req) {
            int stream_index= -1;
            int64_t seek_target= is->seek_pos;

            if     (is->   video_stream >= 0)    stream_index= is->   video_stream;
            else if(is->   audio_stream >= 0)    stream_index= is->   audio_stream;
            else if(is->   subtitle_stream >= 0) stream_index= is->   subtitle_stream;

            if(stream_index>=0){
                seek_target= av_rescale_q(seek_target, AV_TIME_BASE_Q, ic->streams[stream_index]->time_base);
            }

            ret = av_seek_frame(is->ic, stream_index, seek_target, is->seek_flags);
            if (ret < 0) {
                PyErr_Format(PyExc_IOError, "%s: error while seeking", is->ic->filename);
            }else{
                if (is->audio_stream >= 0) {
                    packet_queue_flush(&is->audioq);
                    packet_queue_put(&is->audioq, &flush_pkt);
                }
                if (is->subtitle_stream >= 0) {
		            packet_queue_flush(&is->subtitleq);
                    packet_queue_put(&is->subtitleq, &flush_pkt);
                }
                if (is->video_stream >= 0) {
                    packet_queue_flush(&is->videoq);
                    packet_queue_put(&is->videoq, &flush_pkt);
                }
            }
            is->seek_req = 0;
        }
        /* if the queue are full, no need to read more */
        if ((is->audioq.size > MAX_AUDIOQ_SIZE) || //yay for short circuit logic testing
            (is->videoq.size > MAX_VIDEOQ_SIZE )||
            (is->subtitleq.size > MAX_SUBTITLEQ_SIZE)) {
            /* wait 10 ms */
            SDL_Delay(10);
            continue;
        }
        if(url_feof(ic->pb)) {
            av_init_packet(pkt);
            pkt->data=NULL;
            pkt->size=0;
            pkt->stream_index= is->video_stream;
            packet_queue_put(&is->videoq, pkt);
            continue;
        }
		if(is->pictq_size<VIDEO_PICTURE_QUEUE_SIZE)
		{
	        ret = av_read_frame(ic, pkt);
	        if (ret < 0) {
	            if (ret != AVERROR_EOF && url_ferror(ic->pb) == 0) {
	                goto fail;
	            } else
	            {
	                break;
	            }
	        }
	        /*if (pkt->stream_index == is->audio_stream) {
	            packet_queue_put(&is->audioq, pkt);
	        } else*/ if (pkt->stream_index == is->video_stream) {
	            packet_queue_put(&is->videoq, pkt);
	        //} else if (pkt->stream_index == is->subtitle_stream) {
	        //    packet_queue_put(&is->subtitleq, pkt);
	        } else if(pkt) {
	            av_free_packet(pkt);
	        }
		}
		if(is->video_st)
	        video_render(is);
        //audio_thread(is);
        if(co<2)
        	video_refresh_timer(is);
        if(is->timing>0) {
        	double showtime = is->timing+is->last_showtime;
            double now = av_gettime()/1000.0;
            if(now >= showtime) {
            	double temp = is->timing;
                is->timing =0;
                if(!video_display(is))
                {
                	is->timing=temp;
                }
                is->last_showtime = av_gettime()/1000.0;
                
            } else {
                SDL_Delay(10);
            }
        }
        /*
        if(is->video_clock >=4.5)
        {
        	GRABGIL
        	PySys_WriteStdout("PTS: %f\n", is->video_clock);
        	RELEASEGIL
        }*/
    }

    ret = 0;
 fail:
    /* disable interrupting */

	if(ret!=0)
	{
		//throw python error
	}
	is->pictq_size=is->pictq_rindex=is->pictq_windex=0;
	packet_queue_flush(&is->videoq);
	
	GRABGIL		
    Py_DECREF( is);
    RELEASEGIL
    is->stop =1;
    if(is->abort_request)
    {	return -1;}
    return 0;
}

int video_render(PyMovie *movie)
{
    DECLAREGIL
    GRABGIL
    Py_INCREF( movie);
    RELEASEGIL
    AVPacket pkt1, *pkt = &pkt1;
    int len1, got_picture;
    AVFrame *frame= avcodec_alloc_frame();
    double pts;

    do {
    	
        if (movie->paused && !movie->videoq.abort_request) {
            return 0;
        }
        if (packet_queue_get(&movie->videoq, pkt, 0) <=0)
            break;
		
        if(pkt->data == flush_pkt.data){
            avcodec_flush_buffers(movie->video_st->codec);
            continue;
        }

        /* NOTE: ipts is the PTS of the _first_ picture beginning in
           this packet, if any */

        movie->video_st->codec->reordered_opaque= pkt->pts;
        len1 = avcodec_decode_video(movie->video_st->codec,
                                    frame, &got_picture,
                                    pkt->data, pkt->size);
		
        if(   ( pkt->dts == AV_NOPTS_VALUE)
           && frame->reordered_opaque != AV_NOPTS_VALUE)
        {
            pts= frame->reordered_opaque;
        }
        else if(pkt->dts != AV_NOPTS_VALUE)
        {
            pts= pkt->dts;
        }
        else
        {
            pts= 0;
        }
        pts *= av_q2d(movie->video_st->time_base);

        if (got_picture) {
        	update_video_clock(movie, frame, pts);
        	if (queue_picture(movie, frame) < 0)
        	{
        		goto the_end;
        	}
        }
        av_free_packet(pkt);
    }
    while(0);
    
 the_end:
 	GRABGIL
    Py_DECREF(movie);
    RELEASEGIL
    av_free(frame);
    return 0;
}
