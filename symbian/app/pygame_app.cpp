/** pygame S60 application */
#include <stddef.h>
#include <coecntrl.h>
#include <aknappui.h>
#include <aknapp.h>
#include <akndoc.h>
#include <sdlepocapi.h>
#include <aknnotewrappers.h>
#include <eikstart.h>
#include "pygame.hrh"

#include <BACLINE.H>

#include "logmanutils.h"

TInt Ticks;
TInt Frames;
TInt Done;

class MExitWait
{
public:
	virtual void DoExit(TInt aErr) = 0;
};

class CExitWait: public CActive
{
public:
	CExitWait(MExitWait& aWait);
	void Start();
	~CExitWait();
private:
	void RunL();
	void DoCancel();
private:
	MExitWait& iWait;
	TRequestStatus* iStatusPtr;
};

class CSDLWin: public CCoeControl
{
public:
	void ConstructL(const TRect& aRect);
	RWindow& GetWindow() const;
	void SetNoDraw();
private:
	void Draw(const TRect& aRect) const;
};

class CSdlApplication: public CAknApplication
{
private:
	// from CApaApplication
	CApaDocument* CreateDocumentL();
	TUid AppDllUid() const;
};

class CSdlAppDocument: public CAknDocument
{
public:
	CSdlAppDocument(CEikApplication& aApp) :
		CAknDocument(aApp)
	{
	}
private:
	CEikAppUi* CreateAppUiL();
};

class CSdlAppUi: public CAknAppUi, public MExitWait
{
public:
	void ConstructL();
	~CSdlAppUi();
private:
	void HandleCommandL(TInt aCommand);
	void HandleResourceChangeL(TInt aType);
	void StartTestL(TInt aCmd);
	void DoExit(TInt aErr);
	void HandleWsEventL(const TWsEvent& aEvent, CCoeControl* aDestination);
private:
	CExitWait* iWait;
	CSDLWin* iSDLWin;
	CSDL* iSdl;
	TBool iExit;
};

CExitWait::CExitWait(MExitWait& aWait) :
	CActive(CActive::EPriorityStandard), iWait(aWait)
{
	CActiveScheduler::Add(this);
}

CExitWait::~CExitWait()
{
	Cancel();
}

void CExitWait::RunL()
{
	if (iStatusPtr != NULL)
		iWait.DoExit(iStatus.Int());
}

void CExitWait::DoCancel()
{
	if (iStatusPtr != NULL)
	{
		User::RequestComplete(iStatusPtr, KErrCancel);
	}
}

void CExitWait::Start()
{
	SetActive();
	iStatusPtr = &iStatus;
}

void CSDLWin::ConstructL(const TRect& aRect)
{
	CreateWindowL();
	SetRect(aRect);
	ActivateL();
}

RWindow& CSDLWin::GetWindow() const
{
	return Window();
}

void CSDLWin::Draw(const TRect& /*aRect*/) const
{
	// Draw black( it will be white otherwise and that's even worse looking )
	// TODO: Take a screenshot and maybe do some kind of fade thingy
	CWindowGc& gc = SystemGc();
	gc.SetDrawMode(CGraphicsContext::EDrawModeWriteAlpha);
	gc.Clear();

	gc.SetPenStyle(CGraphicsContext::ENullPen);
	gc.SetPenColor(0x000000);
	gc.SetBrushStyle(CGraphicsContext::ESolidBrush);
	gc.SetBrushColor(0x000000);
	gc.DrawRect(Rect());

}
/*
 TKeyResponse CSDLWin::OfferKeyEventL(const TKeyEvent &aKeyEvent, TEventCode aType)
 {
 TKeyResponse result = EKeyWasNotConsumed;
 if ( aKeyEvent.iScanCode == 164 )
 {
 result = EKeyWasConsumed;
 }
 return result;
 }
 */

void CSdlAppUi::ConstructL()
{
	BaseConstructL(CAknAppUi::EAknEnableSkin /* | ENoScreenFurniture*/);

	iSDLWin = new (ELeave) CSDLWin;
	iSDLWin->ConstructL(ApplicationRect());

	iWait = new (ELeave) CExitWait(*this);

	StartTestL(0);
}

void CSdlAppUi::HandleCommandL(TInt aCommand)
{
}

extern "C"
{
#include <SDL_events.h>
}
void CSdlAppUi::HandleResourceChangeL(TInt aType)
{

	//User::InfoPrint(_L("rect.Height()"));
	if (aType == KEikDynamicLayoutVariantSwitch)
	{
		// Create SDL resize event
		TRect rect;
		AknLayoutUtils::LayoutMetricsRect(AknLayoutUtils::EApplicationWindow,
				rect);

		SDL_Event event;
		event.type = SDL_VIDEORESIZE;
		event.resize.w = rect.Width();
		event.resize.h = rect.Height();
		//if ( (SDL_EventOK == NULL) || (*SDL_EventOK)(&event) ) {
		SDL_PushEvent(&event);

		iSDLWin->SetRect(rect);
		/*
		 iSdl->SetContainerWindowL(
		 iSDLWin->GetWindow(),
		 iEikonEnv->WsSession(),
		 *iEikonEnv->ScreenDevice());
		 */
	}

	CAknAppUi::HandleResourceChangeL(aType);

}

void CSdlAppUi::StartTestL(TInt aCmd)
{

	//TInt flags = CSDL::EDrawModeGdi | CSDL::EEnableFocusStop
	//		| CSDL::EMainThread;// | CSDL::EAutoOrientation;

	TInt flags = 0;

	//flags |= CSDL::EDrawModeDSB | CSDL::EDrawModeDSBDoubleBuffer;
	flags |= CSDL::EDrawModeGdi;
	// Don't draw when in background.
	//flags |= CSDL::EEnableFocusStop;
	flags |= CSDL::EAutoOrientation;
	// This should be on by default anyway
	flags |= CSDL::EMainThread;

	//Create CommandLine Arguments and read it.
	CDesC8ArrayFlat *arr = new (ELeave) CDesC8ArrayFlat(1);
	CleanupStack::PushL(arr);

	CCommandLineArguments* args = CCommandLineArguments::NewLC();
	// The real args we are interested in start at the 2nd arg
	for (TInt i = 1; i < args->Count(); i++)
	{
		TBuf8<256> arg;
		arg.Copy(args->Arg(i));

		arr->AppendL(arg);
		//TPtrC argumentPrt(args->Arg(i));
		//console->Printf(_L("Arg %d == %S\n"), i, &argumentPrt);
	}

	iSdl = CSDL::NewL(flags);

	iSdl->SetContainerWindowL(iSDLWin->GetWindow(), iEikonEnv->WsSession(),
			*iEikonEnv->ScreenDevice());
	iSdl->CallMainL(iWait->iStatus, *arr, flags, 0x14000);
	iWait->Start();

	arr->Reset();
	CleanupStack::PopAndDestroy(2); // command line and arr
}

void CSdlAppUi::DoExit(TInt aErr)
{
	if (aErr != KErrNone)
	{
		CAknErrorNote* err = new (ELeave) CAknErrorNote(ETrue);
		TBuf<64> buf;
		if (aErr == 1)
		{
			buf.Copy(_L("Python run-time error."));
		}
		else
		{
			buf.Format(_L("SDL Error %d"), aErr);
		}
		err->ExecuteLD(buf);
	}
	else
	{
		/*
		 CAknInformationNote* info = new (ELeave) CAknInformationNote(ETrue);
		 info->SetTimeout(CAknNoteDialog::ENoTimeout);
		 TBuf<64> buf;
		 const TReal ticks = TReal(Ticks) / 1000.0;
		 const TReal fps = TReal(Frames) / ticks;
		 buf.Format(_L("Fps %f, %dms %d frames"), fps, Ticks, Frames);
		 info->ExecuteLD(buf);
		 */
	}
	delete iSdl;
	iSdl = NULL;

	// Exits after main script has completed
	Exit();
}

void CSdlAppUi::HandleWsEventL(
		const TWsEvent& aEvent, CCoeControl* aDestination)
{
	if (iSdl != NULL)
		iSdl->AppendWsEvent(aEvent);
	CAknAppUi::HandleWsEventL(aEvent, aDestination);
}

CSdlAppUi::~CSdlAppUi()
{
	if (iWait != NULL)
		iWait->Cancel();
	delete iSdl;
	delete iWait;
	delete iSDLWin;
}

CEikAppUi* CSdlAppDocument::CreateAppUiL()
{
	return new (ELeave) CSdlAppUi();
}

TUid CSdlApplication::AppDllUid() const
{
    // Get the uid from process	
	return  RProcess().SecureId();
}

CApaDocument* CSdlApplication::CreateDocumentL()
{
	CSdlAppDocument* document = new (ELeave) CSdlAppDocument(*this);
	return document;
}

LOCAL_C CApaApplication* NewApplication()
{
	return new CSdlApplication;
}

GLDEF_C TInt E32Main()
{
	// TODO: Is this the only way to set heap size on emulator?
	//#ifdef __WINS__
	RHeap *heap = UserHeap::ChunkHeap(0, 100000, 10000000, 100000);
	User::SwitchHeap(heap);
	//#endif
	TInt result = EikStart::RunApplication(NewApplication);
	//#ifdef __WINS__
	heap->Close();
	//#endif
	return result;
}

