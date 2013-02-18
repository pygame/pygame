/*
    pygame - Python Game Library
    Copyright (C) 2000-2001  Pete Shinners

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

    Pete Shinners
    pete@shinners.org

This code based on the sample "ffplay" included with ffmpeg.
The ffmpeg library is released as LGPL software.

*/
#include "pygame.h"
#include "ffmovie.h"

#ifndef MIN
#define MIN(a,b)    (((a) < (b)) ? (a) : (b))
#endif


static int Global_abort_all = 0;
static int Global_num_active = 0;


static void print_error(const char *filename, int err)
{
    switch(err) {
    case AVERROR_NUMEXPECTED:
        fprintf(stderr, "%s: Incorrect image filename syntax.\n"
                "Use '%%d' to specify the image number:\n"
                "  for img1.jpg, img2.jpg, ..., use 'img%%d.jpg';\n"
                "  for img001.jpg, img002.jpg, ..., use 'img%%03d.jpg'.\n",
                filename);
        break;
    case AVERROR_INVALIDDATA:
        fprintf(stderr, "%s: Error while parsing header\n", filename);
        break;
    case AVERROR_NOFMT:
        fprintf(stderr, "%s: Unknown format\n", filename);
        break;
    default:
        fprintf(stderr, "%s: Error while opening file (%d)\n", filename, err);
        break;
    }
}




/* packet queue handling */
static void packet_queue_init(PacketQueue *q)
{
    memset(q, 0, sizeof(PacketQueue));
    q->mutex = SDL_CreateMutex();
    q->cond = SDL_CreateCond();
}

static void packet_queue_end(PacketQueue *q)
{
    AVPacketList *pkt, *pkt1;

    for(pkt = q->first_pkt; pkt != NULL; pkt = pkt1) {
        pkt1 = pkt->next;
        av_free_packet(&pkt->pkt);
    }
    SDL_DestroyMutex(q->mutex);
    SDL_DestroyCond(q->cond);
}

static int packet_queue_put(PacketQueue *q, AVPacket *pkt)
{
    AVPacketList *pkt1;

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
    q->size += pkt1->pkt.size;
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
        if (q->abort_request || Global_abort_all) {
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





static void video_display(FFMovie *movie)
{
/*DECODE THREAD - from video_refresh_timer*/

    SDL_LockMutex(movie->dest_mutex);
    if (movie->dest_overlay) {
        SDL_DisplayYUVOverlay(movie->dest_overlay, &movie->dest_rect);
    }
    SDL_UnlockMutex(movie->dest_mutex);
}


/* get the current audio clock value */
static double get_audio_clock(FFMovie *movie)
{
/*SDL AUDIO THREAD*/
    double pts;
    int hw_buf_size, bytes_per_sec;
    pts = movie->audio_clock;
    hw_buf_size = movie->audio_hw_buf_size - movie->audio_buf_index;
    bytes_per_sec = 0;
    if (movie->audio_st) {
        bytes_per_sec = movie->audio_st->codec.sample_rate *
            2 * movie->audio_st->codec.channels;
    }
    if (bytes_per_sec)
        pts -= (double)hw_buf_size / bytes_per_sec;
    return pts;
}


static double get_master_clock(FFMovie *movie) {
    Uint32 ticks = SDL_GetTicks();
    return (ticks / 1000.0) - movie->time_offset;
}



/* called to display each frame */
static void video_refresh_timer(FFMovie* movie)
{
/*moving to DECODE THREAD, from queue_frame*/
    double actual_delay, delay, sync_threshold, ref_clock, diff;
    int skipframe = 0;

    if (movie->video_st) { /*shouldn't ever even get this far if no video_st*/

        /* update current video pts */
        movie->video_current_pts = movie->video_clock;

        /* compute nominal delay */
        delay = movie->video_clock - movie->frame_last_pts;
        if (delay <= 0 || delay >= 1.0) {
            /* if incorrect delay, use previous one */
            delay = movie->frame_last_delay;
        }
        movie->frame_last_delay = delay;
        movie->frame_last_pts = movie->video_clock;

        /* we try to correct big delays by duplicating or deleting a frame */
        ref_clock = get_master_clock(movie);
        diff = movie->video_clock - ref_clock;

//printf("get_master_clock = %f\n", (float)(ref_clock/1000000.0));
        /* skip or repeat frame. We take into account the delay to compute
           the threshold. I still don't know if it is the best guess */
        sync_threshold = AV_SYNC_THRESHOLD;
        if (delay > sync_threshold)
            sync_threshold = delay;
        if (fabs(diff) < AV_NOSYNC_THRESHOLD) {
            if (diff <= -sync_threshold) {
                skipframe = 1;
                delay = 0;
            } else if (diff >= sync_threshold) {
                delay = 2 * delay;
            }
        }

        movie->frame_timer += delay;
        actual_delay = movie->frame_timer - get_master_clock(movie);
//printf("DELAY: delay=%f, frame_timer=%f, video_clock=%f\n",
//                (float)delay, (float)movie->frame_timer, (float)movie->video_clock);
        if (actual_delay > 0.010) {
            movie->dest_showtime = movie->frame_timer;
        }
        if (skipframe) {
            movie->dest_showtime = 0;
            /*movie->dest_showtime = get_master_clock(movie); this shows every frame*/
        }
    }
}



static int queue_picture(FFMovie *movie, AVFrame *src_frame)
{
/*DECODE LOOP*/
    AVPicture pict;

    SDL_LockMutex(movie->dest_mutex);

    /* if the frame movie not skipped, then display it */

    if (movie->dest_overlay) {
        /* get a pointer on the bitmap */
        SDL_LockYUVOverlay(movie->dest_overlay);

        pict.data[0] = movie->dest_overlay->pixels[0];
        pict.data[1] = movie->dest_overlay->pixels[2];
        pict.data[2] = movie->dest_overlay->pixels[1];
        pict.linesize[0] = movie->dest_overlay->pitches[0];
        pict.linesize[1] = movie->dest_overlay->pitches[2];
        pict.linesize[2] = movie->dest_overlay->pitches[1];

/*
  first fields of AVFrame match AVPicture, so it appears safe to
  cast here (at least of ffmpeg-0.4.8, this is how ffplay does it)
  AVPicture is just a container for 4 pixel pointers and 4 strides
*/
        img_convert(&pict, PIX_FMT_YUV420P,
                    (AVPicture *)src_frame, movie->video_st->codec.pix_fmt,
                    movie->video_st->codec.width, movie->video_st->codec.height);

        SDL_UnlockYUVOverlay(movie->dest_overlay);

        video_refresh_timer(movie);
    }
    SDL_UnlockMutex(movie->dest_mutex);

    return 0;
}


static void update_video_clock(FFMovie *movie, AVFrame* frame, double pts) {
    /* if B frames are present, and if the current picture is a I
       or P frame, we use the last pts */
    if (movie->video_st->codec.has_b_frames &&
        frame->pict_type != FF_B_TYPE) {

        double last_P_pts = movie->video_last_P_pts;
        movie->video_last_P_pts = pts;
        pts = last_P_pts;
    }

    /* update video clock with pts, if present */
    if (pts != 0) {
        movie->video_clock = pts;
    } else {
        movie->video_clock += movie->frame_delay;
        /* for MPEG2, the frame can be repeated, update accordingly */
        if (frame->repeat_pict) {
            movie->video_clock += frame->repeat_pict *
                    (movie->frame_delay * 0.5);
        }
    }
}


static int video_read_packet(FFMovie *movie, AVPacket *pkt)
{
/*DECODE THREAD*/
    unsigned char *ptr;
    int len, len1, got_picture;
    AVFrame frame;
    double pts;

    ptr = pkt->data;
    if (movie->video_st->codec.codec_id == CODEC_ID_RAWVIDEO) {
        avpicture_fill((AVPicture *)&frame, ptr,
                       movie->video_st->codec.pix_fmt,
                       movie->video_st->codec.width,
                       movie->video_st->codec.height);
        if (pkt->pts != AV_NOPTS_VALUE)
            pts = (double)pkt->pts * movie->context->pts_num / movie->context->pts_den;
        else
            pts = 0;
        frame.pict_type = FF_I_TYPE;
        update_video_clock(movie, &frame, pts);
movie->frame_count++; /*this should probably represent displayed frames, not decoded*/
        if (queue_picture(movie, &frame) < 0)
            return -1;
    } else {
        len = pkt->size;
        while (len > 0) {
            if (movie->vidpkt_start) {
                movie->vidpkt_start = 0;
                movie->vidpkt_timestamp = pkt->pts;
            }
            len1 = avcodec_decode_video(&movie->video_st->codec,
                                        &frame, &got_picture, ptr, len);
            if (len1 < 0)
                break;
            if (got_picture) {
movie->frame_count++; /*this should probably represent displayed frames, not decoded*/
                if (movie->vidpkt_timestamp != AV_NOPTS_VALUE)
                    pts = (double)movie->vidpkt_timestamp * movie->context->pts_num / movie->context->pts_den;
                else
                    pts = 0;
                update_video_clock(movie, &frame, pts);
                if (queue_picture(movie, &frame) < 0)
                    return -1;
                movie->vidpkt_start = 1;
            }
            ptr += len1;
            len -= len1;
        }
    }
    return 0;
}





/* return the new audio buffer size (samples can be added or deleted
   to get better sync if video or external master clock) */
static int synchronize_audio(FFMovie *movie, short *samples,
                             int samples_size1, double pts)
{
/*SDL AUDIO THREAD*/
    int n, samples_size;
    double ref_clock;

    double diff, avg_diff;
    int wanted_size, min_size, max_size, nb_samples;


    n = 2 * movie->audio_st->codec.channels;
    samples_size = samples_size1;

    /* try to remove or add samples to correct the clock */
    ref_clock = get_master_clock(movie);
    diff = get_audio_clock(movie) - ref_clock;

    if (diff < AV_NOSYNC_THRESHOLD) {
        movie->audio_diff_cum = diff + movie->audio_diff_avg_coef * movie->audio_diff_cum;
        if (movie->audio_diff_avg_count < AUDIO_DIFF_AVG_NB) {
            /* not enough measures to have a correct estimate */
            movie->audio_diff_avg_count++;
        } else {
            /* estimate the A-V difference */
            avg_diff = movie->audio_diff_cum * (1.0 - movie->audio_diff_avg_coef);

            if (fabs(avg_diff) >= movie->audio_diff_threshold) {
                wanted_size = samples_size + ((int)(diff * movie->audio_st->codec.sample_rate) * n);
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
        movie->audio_diff_avg_count = 0;
        movie->audio_diff_cum = 0;
    }

    return samples_size;
}

/* decode one audio frame and returns its uncompressed size */
static int audio_decode_frame(FFMovie *movie, uint8_t *audio_buf, double *pts_ptr)
{
/*SDL AUDIO THREAD*/
    AVPacket *pkt = &movie->audio_pkt;
    int len1, data_size;
    double pts;

    for(;;) {
        if (movie->paused || movie->audioq.abort_request || Global_abort_all) {
            return -1;
        }
        while (movie->audio_pkt_size > 0) {
            len1 = avcodec_decode_audio(&movie->audio_st->codec,
                                        (int16_t *)audio_buf, &data_size,
                                        movie->audio_pkt_data, movie->audio_pkt_size);
            if (len1 < 0)
                break;
            movie->audio_pkt_data += len1;
            movie->audio_pkt_size -= len1;
            if (data_size > 0) {
                pts = 0;
                if (movie->audio_pkt_ipts != AV_NOPTS_VALUE)
                    pts = (double)movie->audio_pkt_ipts * movie->context->pts_num / movie->context->pts_den;
                /* if no pts, then compute it */
                if (pts != 0) {
                    movie->audio_clock = pts;
                } else {
                    int n;
                    n = 2 * movie->audio_st->codec.channels;
                    movie->audio_clock += (double)data_size / (double)(n * movie->audio_st->codec.sample_rate);
                }
                *pts_ptr = movie->audio_clock;
                movie->audio_pkt_ipts = AV_NOPTS_VALUE;
                /* we got samples : we can exit now */
                return data_size;
            }
        }

        /* free previous packet if any */
        if (pkt->destruct)
            av_free_packet(pkt);

        /* read next packet */
        if (packet_queue_get(&movie->audioq, pkt, 1) < 0)
            return -1;
        movie->audio_pkt_data = pkt->data;
        movie->audio_pkt_size = pkt->size;
        movie->audio_pkt_ipts = pkt->pts;
    }
}


/* prepare a new audio buffer */
void sdl_audio_callback(void *opaque, Uint8 *stream, int len)
{
/*SDL AUDIO THREAD*/
    FFMovie *movie = opaque;
    int audio_size, len1;
    double pts;

    while (len > 0) {
        if (movie->audio_buf_index >= movie->audio_buf_size) {
           audio_size = audio_decode_frame(movie, movie->audio_buf, &pts);
           if (audio_size < 0) {
                /* if error, just output silence */
               movie->audio_buf_size = 1024;
               memset(movie->audio_buf, 0, movie->audio_buf_size);
           } else {
               audio_size = synchronize_audio(movie, (int16_t*)movie->audio_buf, audio_size, pts);
               movie->audio_buf_size = audio_size;
           }
           movie->audio_buf_index = 0;
        }
        len1 = movie->audio_buf_size - movie->audio_buf_index;
        if (len1 > len)
            len1 = len;
        memcpy(stream, (uint8_t *)movie->audio_buf + movie->audio_buf_index, len1);
        len -= len1;
        stream += len1;
        movie->audio_buf_index += len1;
    }
}



static void ffmovie_cleanup(FFMovie *movie) {
    if(!movie)
        return;

    if(movie->audio_st) {
        packet_queue_abort(&movie->audioq);
        SDL_CloseAudio();
        packet_queue_end(&movie->audioq);
        avcodec_close(&movie->audio_st->codec);
        movie->audio_st = NULL;
    }

    if(movie->video_st) {
        avcodec_close(&movie->video_st->codec);
        movie->video_st = NULL;
    }

    if (movie->context) {
        av_close_input_file(movie->context);
        movie->context = NULL;
    }

    if(movie->dest_mutex) {
        SDL_DestroyMutex(movie->dest_mutex);
        movie->dest_mutex = NULL;
    }

    if (movie->dest_overlay) {
        SDL_FreeYUVOverlay(movie->dest_overlay);
        movie->dest_overlay = NULL;
    }

    Global_num_active--;
}


/* this thread gets the stream from the disk or the network */
static int decode_thread(void *arg)
{
/* DECODE THREAD */
    FFMovie *movie = arg;
    int status;
    AVPacket pkt1, *pkt = &pkt1;

    while(!movie->abort_request && !Global_abort_all) {
        /* read if the queues have room */
        if (movie->audioq.size < MAX_AUDIOQ_SIZE &&
            !movie->dest_showtime) {

            if (av_read_packet(movie->context, pkt) < 0) {
                break;
            }
            if (movie->audio_st &&
                        pkt->stream_index == movie->audio_st->index) {
                packet_queue_put(&movie->audioq, pkt);
            } else if (movie->video_st &&
                    pkt->stream_index == movie->video_st->index) {
                status = video_read_packet(movie, pkt);
                av_free_packet(pkt);
                if(status < 0) {
                    break;
                }
            } else {
                av_free_packet(pkt);
            }
        }

        if(movie->dest_showtime) {
            double now = get_master_clock(movie);
            if(now >= movie->dest_showtime) {
                video_display(movie);
                movie->dest_showtime = 0;
            } else {
//                printf("showtime not ready, waiting... (%.2f,%.2f)\n",
//                            (float)now, (float)movie->dest_showtime);
                SDL_Delay(10);
            }
        }

        
        if(movie->paused) {
            double endpause, startpause = SDL_GetTicks() / 1000.0;
            while(movie->paused && !movie->abort_request && !Global_abort_all) {
                SDL_Delay(100);
            }
            endpause = SDL_GetTicks() / 1000.0;
            movie->dest_showtime = 0;
            movie->time_offset += endpause - startpause;
        }
    }

    ffmovie_cleanup(movie);
    return 0;
}


static int audiostream_init(FFMovie *movie, AVStream *stream)
{
/* MAIN THREAD */
    AVCodec *codec;
    SDL_AudioSpec wanted_spec, spec;

    codec = avcodec_find_decoder(stream->codec.codec_id);
    if (!codec || avcodec_open(&stream->codec, codec) < 0) {
        return -1;
    }

    /* init sdl audio output */
    wanted_spec.freq = stream->codec.sample_rate;
    wanted_spec.format = AUDIO_S16SYS;
    wanted_spec.channels = stream->codec.channels;
    if(wanted_spec.channels > 2)
        wanted_spec.channels = 2;
    wanted_spec.silence = 0;
    wanted_spec.samples = SDL_AUDIO_BUFFER_SIZE;
    wanted_spec.callback = sdl_audio_callback;
    wanted_spec.userdata = movie;
    if (SDL_OpenAudio(&wanted_spec, &spec) < 0) {
        fprintf(stderr, "SDL_OpenAudio: %s\n", SDL_GetError());
        return -1;
    }

    movie->audio_st = stream;
    movie->audio_hw_buf_size = spec.size;
    movie->audio_buf_size = 0;
    movie->audio_buf_index = 0;
    movie->audio_pkt_size = 0;

    /* init averaging filter */
    movie->audio_diff_avg_coef = exp(log(0.01) / AUDIO_DIFF_AVG_NB);
    movie->audio_diff_avg_count = 0;
    /* since we do not have a precise anough audio fifo fullness,
       we correct audio sync only if larger than this threshold */
    movie->audio_diff_threshold = 2.0 *
            SDL_AUDIO_BUFFER_SIZE / stream->codec.sample_rate;
    /* should this be spec.freq instead of codec.sample_rate ?? */

    memset(&movie->audio_pkt, 0, sizeof(movie->audio_pkt));
    packet_queue_init(&movie->audioq);
    SDL_PauseAudio(0);

    return 0;
}


static int videostream_init(FFMovie *movie, AVStream *stream)
{
/* MAIN THREAD */
    AVCodec *codec;

    codec = avcodec_find_decoder(stream->codec.codec_id);
    if (!codec || avcodec_open(&stream->codec, codec) < 0)
        return -1;

    movie->video_st = stream;
    movie->frame_last_delay = 40e-3;
    movie->frame_timer = SDL_GetTicks() / 1000.0;

    movie->frame_delay = (double)movie->video_st->codec.frame_rate_base /
                (double)movie->video_st->codec.frame_rate;

    movie->vidpkt_start = 1;

    movie->dest_mutex = SDL_CreateMutex();

    return 0;
}



static int ffmovie_initialized = 0;


FFMovie *ffmovie_open(const char *filename)
{
/* MAIN THREAD */
    FFMovie *movie;
    int err, i;
    AVFormatParameters params = {0};

    if(!ffmovie_initialized) {
        ffmovie_initialized = 1;
        av_register_all();
    }
    
    movie = av_mallocz(sizeof(FFMovie));
    if (!movie)
        return NULL;

    err = av_open_input_file(&movie->context, filename, NULL, 0, &params);
    if (err < 0) {
        print_error(filename, err);
        return NULL;
    }

    err = av_find_stream_info(movie->context);
    if (err < 0) {
        av_free(movie);
        fprintf(stderr, "%s: could not find codec parameters\n", filename);
        return NULL;
    }

    /*find and open streams*/
    for(i = 0; i < movie->context->nb_streams; i++) {
        AVStream *stream = movie->context->streams[i];
        switch(stream->codec.codec_type) {
            case CODEC_TYPE_AUDIO:
                if (!movie->audio_st && !movie->audio_disable)
                    audiostream_init(movie, stream);
                break;
            case CODEC_TYPE_VIDEO:
                if (!movie->video_st)
                    videostream_init(movie, stream);
                break;
            default: break;
        }
    }

    if (!movie->video_st && !movie->audio_st) {
        fprintf(stderr, "%s: could not open codecs\n", filename);
        ffmovie_cleanup(movie);
        return NULL;
    }

    movie->frame_count = 0;
    movie->time_offset = 0.0;
    movie->paused = 1;
    movie->sourcename = strdup(filename);

    Global_num_active++;
    movie->decode_thread = SDL_CreateThread(decode_thread, movie);
    if (!movie->decode_thread) {
        ffmovie_cleanup(movie);
        return NULL;
    }
    return movie;
}


void ffmovie_close(FFMovie *movie)
{
/*MAIN THREAD*/
    movie->abort_request = 1;
    SDL_WaitThread(movie->decode_thread, NULL);
    if(movie->sourcename) {
        free((void*)movie->sourcename);
    }
    av_free(movie);
}

void ffmovie_play(FFMovie *movie) {
    movie->paused = 0;
}

void ffmovie_stop(FFMovie *movie) {
    movie->paused = 1;
    /*should force blit of current frame to source*/
    /*even better, to rgb not just yuv*/
}

void ffmovie_pause(FFMovie *movie) {
    if(movie->paused) {
        ffmovie_play(movie);
    } else {
        ffmovie_stop(movie);
    }
}

int ffmovie_finished(FFMovie *movie) {
    return movie->context == NULL;
}


void ffmovie_setdisplay(FFMovie *movie, SDL_Surface *dest, SDL_Rect *rect)
{
/*MAIN THREAD*/

    if(!movie->video_st || movie->abort_request || movie->context==NULL) {
        /*This movie has no video stream, or finished*/
        return;
    }

    SDL_LockMutex(movie->dest_mutex);

    if(movie->dest_overlay) {
        /*clean any existing overlay*/
        SDL_FreeYUVOverlay(movie->dest_overlay);
        movie->dest_overlay = NULL;
    }

    if(!dest) {
        /*no destination*/
        movie->dest_overlay = NULL;
    } else {
        if(rect) {
            movie->dest_rect.x = rect->x;
            movie->dest_rect.y = rect->y;
            movie->dest_rect.w = rect->w;
            movie->dest_rect.h = rect->h;
        } else {
            movie->dest_rect.x = 0;
            movie->dest_rect.y = 0;
            movie->dest_rect.w = 0;
            movie->dest_rect.h = 0;
        }
        if(movie->dest_rect.w == 0) {
            movie->dest_rect.w = MIN(movie->video_st->codec.width, dest->w);
        }
        if(movie->dest_rect.h == 0) {
            movie->dest_rect.h = MIN(movie->video_st->codec.height, dest->h);
        }

#if 0
        /* XXX: use generic function */
        /* XXX: disable overlay if no hardware acceleration or if RGB format */
        switch(movie->video_st->codec.pix_fmt) {
        case PIX_FMT_YUV420P:
        case PIX_FMT_YUV422P:
        case PIX_FMT_YUV444P:
        case PIX_FMT_YUV422:
        case PIX_FMT_YUV410P:
        case PIX_FMT_YUV411P:
            is_yuv = 1;
            break;
        default:
            is_yuv = 0;
            break;
        }
#endif
        movie->dest_surface = dest;
        movie->dest_overlay = SDL_CreateYUVOverlay(
                movie->video_st->codec.width,
                movie->video_st->codec.height,
                SDL_YV12_OVERLAY, dest);
    }

    SDL_UnlockMutex(movie->dest_mutex);
    
    /*set display time to now, force redraw*/
    movie->dest_showtime = get_master_clock(movie);
   
}

void ffmovie_setvolume(FFMovie *movie, int volume) {
    if(movie->audio_st) {
        movie->audio_volume = volume;
        /*note, i'll need to multiply the sound data myself*/
    }
}



void ffmovie_abortall() {
    Global_abort_all = 1;
    while(Global_num_active > 0) {
        SDL_Delay(200);
    }
    Global_abort_all = 0;
}


FFMovie *ffmovie_reopen(FFMovie *movie) {
    const char* filename;
    SDL_Overlay *dest_overlay;
    SDL_Surface *dest_surface;
    SDL_Rect dest_rect;
    int waspaused = movie->paused;

    filename = movie->sourcename;
    movie->sourcename = NULL;
    if(!filename) {
        return NULL;
    }

    SDL_LockMutex(movie->dest_mutex);
    dest_overlay = movie->dest_overlay;
    dest_surface = movie->dest_surface;
    dest_rect = movie->dest_rect;
    movie->dest_overlay = NULL;
    movie->dest_surface = NULL;
    SDL_UnlockMutex(movie->dest_mutex);

    ffmovie_close(movie);
    
    movie = ffmovie_open(filename);
    free((void*)filename);
    
    if(movie) {
        if(dest_overlay) {
            SDL_LockMutex(movie->dest_mutex);
            movie->dest_overlay = dest_overlay;
            movie->dest_surface = dest_surface;
            movie->dest_rect = dest_rect;
            SDL_UnlockMutex(movie->dest_mutex);
        }
        if(!waspaused) {
            ffmovie_play(movie);
        }
    }
   
    return movie;
}

