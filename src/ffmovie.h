#include <SDL.h>
#include <SDL_thread.h>
#include <ffmpeg/avformat.h>



#define MAX_SOURCENAME 1024
#define MAX_AUDIOQ_SIZE (5 * 16 * 1024)

/* SDL audio buffer size, in samples. Should be small to have precise
   A/V sync as SDL does not have hardware buffer fullness info. */
#define SDL_AUDIO_BUFFER_SIZE 1024

/* no AV sync correction is done if below the AV sync threshold */
#define AV_SYNC_THRESHOLD 0.08
/* no AV correction is done if too big error */
#define AV_NOSYNC_THRESHOLD 10.0

/* maximum audio speed change to get correct sync */
#define SAMPLE_CORRECTION_PERCENT_MAX 10

/* we use about AUDIO_DIFF_AVG_NB A-V differences to make the average */
#define AUDIO_DIFF_AVG_NB   20

/* NOTE: the size must be big enough to compensate the hardware audio buffersize size */
#define SAMPLE_ARRAY_SIZE (2*65536)





typedef struct PacketQueue {
    AVPacketList *first_pkt, *last_pkt;
    int nb_packets;
    int size;
    int abort_request;
    SDL_mutex *mutex;
    SDL_cond *cond;
} PacketQueue;



typedef struct FFMovie {
    SDL_Thread *decode_thread;
    int abort_request;
    int paused;
    AVFormatContext *context;

    double external_clock; /* external clock base */
    int64_t external_clock_time;

    double audio_clock;
    double audio_diff_cum; /* used for AV difference average computation */
    double audio_diff_avg_coef;
    double audio_diff_threshold;
    int audio_diff_avg_count;
    AVStream *audio_st;
    PacketQueue audioq;
    int audio_hw_buf_size;
    /* samples output by the codec. we reserve more space for avsync compensation */
    uint8_t audio_buf[(AVCODEC_MAX_AUDIO_FRAME_SIZE * 3) / 2];
    int audio_buf_size; /* in bytes */
    int audio_buf_index; /* in bytes */
    AVPacket audio_pkt;
    uint8_t *audio_pkt_data;
    int audio_pkt_size;
    int64_t audio_pkt_ipts;
    int audio_volume; /*must self implement*/

    int16_t sample_array[SAMPLE_ARRAY_SIZE];
    int sample_array_index;

    int frame_count;
    double frame_timer;
    double frame_last_pts;
    double frame_last_delay;
    double frame_delay; /*display time of each frame, based on fps*/
    double video_clock; /*seconds of video frame decoded*/
    AVStream *video_st;
    int64_t vidpkt_timestamp;
    int vidpkt_start;
    double video_last_P_pts; /* pts of the last P picture (needed if B
                                frames are present) */
    double video_current_pts; /* current displayed pts (different from
                                 video_clock if frame fifos are used) */

    SDL_mutex *dest_mutex;
    double dest_showtime; /*when to next show the dest_overlay*/
    SDL_Overlay *dest_overlay;
    SDL_Surface *dest_surface;
    SDL_Rect dest_rect;

    double time_offset; /*track paused time*/

    int audio_disable;

    const char *sourcename;
} FFMovie;




FFMovie *ffmovie_open(const char *filename);
FFMovie *ffmovie_reopen(FFMovie *movie);
void ffmovie_close(FFMovie *movie);
void ffmovie_play(FFMovie *movie);
void ffmovie_stop(FFMovie *movie);
void ffmovie_pause(FFMovie *movie);
void ffmovie_setvolume(FFMovie *movie, int volume);
void ffmovie_setdisplay(FFMovie *movie, SDL_Surface *dest, SDL_Rect *rect);
void ffmovie_abortall(void);


