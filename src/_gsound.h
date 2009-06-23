#ifndef _GSOUND_H_
#define _GSOUND_H_
#include <SDL/SDL_mixer.h>
#include <SDL.h>
#include <SDL_thread.h>

typedef struct BufferNode{
	uint8_t *buf;
	int      len;
	struct BufferNode *next;
} BufferNode;

typedef struct BufferQueue{
	BufferNode *first, *last;
	int size;
} BufferQueue;

int soundInit  (int freq, int size, int channels, int chunksize);
int soundQuit  (void);
int playBuffer (uint8_t *buf, uint32_t len, int channel);
int stopBuffer (int channel);
int pauseBuffer(int channel);
int getPaused  (int channel);
int seekBuffer (uint8_t *buf, uint32_t len, int channel );
int setCallback(void (*callback) (int channel));

#endif /*_GSOUND_H_*/
