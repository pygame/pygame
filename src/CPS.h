// CPS.h

#ifdef __cplusplus
extern "C" {
#endif

#if PRAGMA_STRUCT_ALIGN
    #pragma options align=mac68k
#elif PRAGMA_STRUCT_PACKPUSH
    #pragma pack(push, 2)
#elif PRAGMA_STRUCT_PACK
    #pragma pack(2)
#endif


struct CPSProcessSerNum
{
	UInt32		lo;
	UInt32		hi;
};

typedef struct CPSProcessSerNum	CPSProcessSerNum;

enum
{
	kCPSNoProcess		=	0,
	kCPSSystemProcess	=	1,
	kCPSCurrentProcess	=	2
};


enum
{
	bfCPSIntraProcessSwitch =	1,
	bfCPSDeathBySignal	=	2
};

typedef UInt16	CPSEventFlags;


enum
{
	kCPSBlueApp	=	0,
	kCPSBlueBox	=	1,
	kCPSCarbonApp	=	2,
	kCPSYellowApp	=	3,
	kCPSUnknownApp	=	4
};

typedef UInt32	CPSAppFlavour;


enum
{
	kCPSBGOnlyAttr		=	1024,
	kCPSUIElementAttr	=	65536,
	kCPSHiddenAttr		=	131072,
	kCPSNoConnectAttr	=	262144,
	kCPSFullScreenAttr	=	524288,
	kCPSClassicReqAttr	=	1048576,
	kCPSNativeReqAttr	=	2097152
};

typedef UInt32	CPSProcAttributes;


struct CPSProcessInfoRec
{
	CPSProcessSerNum 	Parent;
	UInt64			LaunchDate;
	CPSAppFlavour		Flavour;
	CPSProcAttributes	Attributes;
	UInt32			ExecFileType;
	UInt32			ExecFileCreator;
	UInt32			UnixPID;
};

typedef struct CPSProcessInfoRec	CPSProcessInfoRec;


enum
{
	kCPSNotifyChildDeath	=	1,
	kCPSNotifyNewFront	=	2,
	kCPSNotifyAppBirth	=	4,
	kCPSNotifyAppDeath	=	8,
	kCPSNotifyLaunch	=	9,
	kCPSNotifyServiceReq	=	16,
	kCPSNotifyAppHidden	=	32,
	kCPSNotifyAppRevealed	=	64,
	kCPSNotifyFGEnabled	=	128,
	kCPSNotifyLaunchStart	=	256,
	kCPSNotifyAppReady	=	512,
	kCPSNotifyLaunchFail	=	1024,
	kCPSNotifyAppDeathExt	=	2048,
	kCPSNotifyLostKeyFocus	=	4096
};

typedef UInt32	CPSNotificationCodes;


enum
{
	bfCPSLaunchInhibitDaemon	=	128,
	bfCPSLaunchDontSwitch		=	512,
	bfCPSLaunchNoProcAttr		=	2048,
	bfCPSLaunchAsync		=	65536,
	bfCPSLaunchStartClassic		=	131072,
	bfCPSLaunchInClassic		=	262144,
	bfCPSLaunchInstance		=	524288,
	bfCPSLaunchAndHide		=	1048576,
	bfCPSLaunchAndHideOthers	=	2097152
};

typedef UInt32	CPSLaunchOptions;


typedef	UInt8	*CPSLaunchRefcon;


typedef	UInt8	*CPSLaunchData;


enum
{
	bfCPSExtLaunchWithData	=	2,
	bfCPSExtLaunchByParent	=	4,
	bfCPSExtLaunchAsUidGid	=	8
};

typedef UInt32	CPSLaunchPBFields;


struct CPSLaunchPB
{
	CPSLaunchPBFields	Contents;
	CPSLaunchData		pData;
	UInt32			DataLen;
	UInt32			DataTag;
	UInt32			RefCon;
	CPSProcessSerNum	Parent;
	UInt32			ChildUID;
	UInt32			ChildGID;
};

typedef struct CPSLaunchPB	CPSLaunchPB;


enum
{
	bfCPSKillHard		=	1,
	bfCPSKillAllClassicApps	=	2
};

typedef UInt32	CPSKillOptions;


enum
{
	kCPSLaunchService	=	0,
	kCPSKillService		=	1,
	kCPSHideService		=	2,
	kCPSShowService		=	3,
	kCPSPrivService		=	4,
	kCPSExtDeathNoteService	=	5
};

typedef UInt32	CPSServiceReqType;


struct CPSLaunchRequest
{
	CPSProcessSerNum	TargetPSN;
	CPSLaunchOptions 	Options;
	CPSProcAttributes 	ProcAttributes;
	UInt8			*pUTF8TargetPath;
	UInt32			PathLen;
};

typedef struct CPSLaunchRequest	CPSLaunchRequest;


struct CPSKillRequest
{
	CPSProcessSerNum	TargetPSN;
	CPSKillOptions		Options;
};

typedef struct CPSKillRequest	CPSKillRequest;


struct CPSHideRequest
{
	CPSProcessSerNum 	TargetPSN;
};

typedef struct CPSHideRequest	CPSHideRequest;


struct CPSShowRequest
{
	CPSProcessSerNum 	TargetPSN;
};

typedef struct CPSShowRequest	CPSShowRequest;


struct CPSExtDeathNotice
{
	CPSProcessSerNum 	DeadPSN;
	UInt32			Flags;
	UInt8			*pUTF8AppPath;
	UInt32			PathLen;
};

typedef struct CPSExtDeathNotice	CPSExtDeathNotice;


union CPSRequestDetails
{
	CPSLaunchRequest 	LaunchReq;
	CPSKillRequest 		KillReq;
	CPSHideRequest 		HideReq;
	CPSShowRequest 		ShowReq;
	CPSExtDeathNotice 	DeathNotice;
};

typedef union CPSRequestDetails	CPSRequestDetails;


struct CPSServiceRequest
{
	CPSServiceReqType 	Type;
	SInt32			ID;
	CPSRequestDetails 	Details;
};

typedef struct CPSServiceRequest	CPSServiceRequest;


enum
{
	kCPSProcessInterruptKey	=	0,
	kCPSAppSwitchFwdKey	=	1,
	kCPSAppSwitchBackKey	=	2,
	kCPSSessionInterruptKey	=	3,
	kCPSScreenSaverKey	=	4,
	kCPSDiskEjectKey	=	5,
	kCPSSpecialKeyCount	=	6
};

typedef SInt32	CPSSpecialKeyID;


extern Boolean	CPSEqualProcess( CPSProcessSerNum *psn1, CPSProcessSerNum *psn2);

extern OSErr	CPSGetCurrentProcess( CPSProcessSerNum *psn);

extern OSErr	CPSGetFrontProcess( CPSProcessSerNum *psn);

extern OSErr	CPSGetNextProcess( CPSProcessSerNum *psn);

extern OSErr	CPSGetNextToFrontProcess( CPSProcessSerNum *psn);

extern OSErr	CPSGetProcessInfo( CPSProcessSerNum *psn, CPSProcessInfoRec *info, char *path, int maxPathLen, int *len, char *name, int maxNameLen);

extern OSErr	CPSPostHideMostReq( CPSProcessSerNum *psn);

extern OSErr	CPSPostHideReq( CPSProcessSerNum *psn);

extern OSErr	CPSPostKillRequest( CPSProcessSerNum *psn, CPSKillOptions options);

extern OSErr	CPSPostShowAllReq( CPSProcessSerNum *psn);

extern OSErr	CPSPostShowReq( CPSProcessSerNum *psn);

extern OSErr	CPSSetFrontProcess( CPSProcessSerNum *psn);

extern OSErr	CPSReleaseKeyFocus( CPSProcessSerNum *psn);

extern OSErr	CPSStealKeyFocus( CPSProcessSerNum *psn);

extern OSErr 	CPSSetProcessName ( CPSProcessSerNum *psn, char *processname);

extern OSErr 	CPSEnableForegroundOperation( CPSProcessSerNum *psn, UInt32 _arg2, UInt32 _arg3, UInt32 _arg4, UInt32 _arg5);




#if PRAGMA_STRUCT_ALIGN
    #pragma options align=reset
#elif PRAGMA_STRUCT_PACKPUSH
    #pragma pack(pop)
#elif PRAGMA_STRUCT_PACK
    #pragma pack()
#endif

#ifdef __cplusplus
}
#endif
