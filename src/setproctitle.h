/* undocumented, may break at any time.. */
extern char **NXArgv;
extern int    NXArgc;

/* arbitrary maximum, I got it from the <broken> libutils darwin version */
#define SPT_BUFSIZE 2048

#include <stdarg.h>
#include <string.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/sysctl.h>
#include <sys/vmparam.h>
#include <unistd.h>
void setproctitle(const char *fmt, ...);