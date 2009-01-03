#include <coecntrl.h>
#include <aknappui.h>
#include <aknapp.h>
#include <akndoc.h>
#include <sdlepocapi.h>
#include <aknnotewrappers.h>
#include <eikstart.h>

#include "pygame.hrh"

const TUid KUidPygameApp =
{ __UID3__ };

TInt Ticks;
TInt Frames;
TInt Done;

class MExitWait
{
public:
	virtual void DoExit(TInt aErr) = 0;
};

class CExitWait : public CActive
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

class CSDLWin : public CCoeControl
{
public:
	void ConstructL(const TRect& aRect);
	RWindow& GetWindow() const;
	void SetNoDraw();
private:
	void Draw(const TRect& aRect) const;
};

class CSdlApplication : public CAknApplication
{
private:
	// from CApaApplication
	CApaDocument* CreateDocumentL();
	TUid AppDllUid() const;
};

class CSdlAppDocument : public CAknDocument
{
public:
	CSdlAppDocument(CEikApplication& aApp) :
		CAknDocument(aApp)
	{
	}
private:
	CEikAppUi* CreateAppUiL();
};

class CSdlAppUi : public CAknAppUi, public MExitWait
{
public:
	void ConstructL();
	~CSdlAppUi();
private:
	void HandleCommandL(TInt aCommand);
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
		User::RequestComplete(iStatusPtr, KErrCancel);
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
	CWindowGc& gc = SystemGc();
	gc.SetPenStyle(CGraphicsContext::ESolidPen);
	gc.SetPenColor(0x000000);
	gc.SetBrushStyle(CGraphicsContext::ESolidBrush);
	gc.SetBrushColor(0x000000);
	gc.DrawRect(Rect());
	 
}

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
	
	switch(aCommand)
	{
		//case EAknCmdExit:
		case EAknSoftkeyExit:
		//case EEikCmdExit:
		Done = 1;
		iExit = ETrue;
		if(iWait == NULL || !iWait->IsActive())
		Exit();
		
		break; 
	}
	
	
	//if(iSdl == NULL)
	//	StartTestL(aCommand);
}

void CSdlAppUi::StartTestL(TInt aCmd)
{
	TInt flags = CSDL::EDrawModeDSBDoubleBuffer;;
 
	iSdl = CSDL::NewL(flags);

	iSdl->SetContainerWindowL(iSDLWin->GetWindow(), 
	iEikonEnv->WsSession(), *iEikonEnv->ScreenDevice());	
	iSdl->CallMainL(iWait->iStatus);
	iWait->Start();
}
 
void CSdlAppUi::DoExit(TInt aErr)
{
	if (aErr != KErrNone)
	{
		CAknErrorNote* err = new (ELeave) CAknErrorNote(ETrue);
		TBuf<64> buf;
		buf.Format(_L("SDL Error %d"), aErr);
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

void CSdlAppUi::HandleWsEventL(const TWsEvent& aEvent, CCoeControl* aDestination)
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
	return new(ELeave) CSdlAppUi();
}

TUid CSdlApplication::AppDllUid() const
{
	return KUidPygameApp;
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
	RHeap *heap = UserHeap::ChunkHeap(0,100000,10000000,100000);
	User::SwitchHeap(heap);
	//#endif	
	TInt result = EikStart::RunApplication(NewApplication);
	//#ifdef __WINS__	
	heap->Close();
	//#endif	
	return result;
}

