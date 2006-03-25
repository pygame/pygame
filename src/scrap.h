
/* Handle clipboard text and data in arbitrary formats */

/* Miscellaneous defines */
#define T(A, B, C, D)	(int)((A<<24)|(B<<16)|(C<<8)|(D<<0))

extern int init_scrap(void);
extern int lost_scrap(void);
extern void put_scrap(int type, int srclen, char *src);
extern void get_scrap(int type, int *dstlen, char **dst);
