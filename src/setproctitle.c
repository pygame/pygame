/* 
   I don't know if it's a good thing to do NXArgc or NXArgv, 
   but it makes finding the beginning a few steps simpler and
   resetting their values might be a good thing for WindowServer
   and the like? 

   bob@redivi.com
*/


#include "setproctitle.h"
void setproctitle(const char *fmt, ...)
{
  int olen,mib[4];
  struct kinfo_proc kp;
  size_t bufSize=sizeof(kp);
  static char newargs[SPT_BUFSIZE];
  char *p1,*minpos,*endorig;
  va_list ap;

  /* write out the formatted string, or quit */
  va_start(ap,fmt);
  if (fmt) {
    newargs[sizeof(newargs)-1] = 0;
    (void)vsnprintf(newargs,sizeof(newargs),fmt,ap);
  } else {
    mib[0]=CTL_KERN;
    mib[1]=KERN_PROC;
    mib[2]=KERN_PROC_PID;
    mib[3]=getpid();
    if (sysctl(mib,4,&kp,&bufSize,NULL,0)) { printf("setproctitle: i dont know my own pid!\n"); return; }
    strcpy(newargs,kp.kp_proc.p_comm);
  }  
  va_end(ap);

  /* find the end of the original string cause we're stackbackwards! */
  endorig = NXArgv[NXArgc-1]+strlen(NXArgv[NXArgc-1]);

  /* kill the original */
  bzero(NXArgv[0],(unsigned int)(endorig-NXArgv[0]));
  for (p1=NXArgv[0]-2;*p1;--p1) *p1=0;

  /* new length (all args) */
  olen=strlen(newargs);

  /* new NXArgv[0] */
  minpos = endorig-olen;
  NXArgv[0] = minpos;

  /* copy the new string to the old place */
  strcpy(NXArgv[0],newargs);

  /* search for spaces, replace with nulls and increment the argc */
  NXArgc=1;
  for (p1=NXArgv[0];*p1;++p1) 
    if (*p1==' ') { *p1=0; NXArgv[NXArgc++] = p1+1; }
  NXArgv[NXArgc]=NULL;

  /* why this is here or what for is beyond me.. theres a copy of the executable name before NXArgv[0] */
  strcpy(NXArgv[0]-strlen(NXArgv[0])-2,NXArgv[0]);

  /* is this even necessary? */
  p1=endorig;
  olen=NXArgc;
  while (++p1<(char *)(USRSTACK-4)) if (!*p1) NXArgv[++olen]=p1+1;  
/*  while (1) {} */
}
