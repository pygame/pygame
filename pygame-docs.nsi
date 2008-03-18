#  NSIS Install script for Pygame Documents and Examples.
#  NSIS (Nullsoft Scriptable Install System) can be downloaded at
#  <http://nsis.sourceforge.net/Main_Page>.
#  Information is take from header pygame-docs.nsh, which
#  contains the constants APP_VERSION, SOURCE_DIRECTORY.

!include "MUI2.nsh"

!include "pygame-docs.nsh"

!define APP_NAME "Pygame ${APP_VERSION} Documents and Examples"
!define SHORT_APP_NAME "pygame-${APP_VERSION}-docs"
!define INSTDIR_REG_ROOT "HKLM"
!define INSTDIR_REG_KEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\${SHORT_APP_NAME}"

!include "AdvUninstLog.nsh"

Name "${APP_NAME}"
OutFile "${SHORT_APP_NAME}-setup.exe"
InstallDir "$PROGRAMFILES\${APP_NAME}"
ShowInstDetails show
ShowUninstDetails show
InstallDirRegKey ${INSTDIR_REG_ROOT} "${INSTDIR_REG_KEY}" "InstallDir"
RequestExecutionLevel admin  ; For Vista, because alters registry.

Var StartMenuFolder
Var InstalledSomething

;!insertmacro UNATTENDED_UNINSTALL
!insertmacro INTERACTIVE_UNINSTALL

!define MUI_WELCOMEPAGE_TITLE "${APP_NAME}"
!define MUI_WELCOMEPAGE_TEXT \
    "This will install the Pygame ${APP_VERSION} HTML documents and tutorials and the example programs."
!define MUI_WELCOMEFINISHPAGE_BITMAP "welcome.bmp"
!insertmacro MUI_PAGE_WELCOME
!define MUI_COMPONENTSPAGE_NODESC
!insertmacro MUI_PAGE_COMPONENTS
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_STARTMENU "ProgramFolder" $StartMenuFolder
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_WELCOME
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_UNPAGE_FINISH

!insertmacro MUI_LANGUAGE "English"

Section "HTML Documents and Tutorials"
    SetOutPath '$INSTDIR'
 
    !insertmacro UNINSTALL.LOG_OPEN_INSTALL

    File /r /x .svn /x *.*~ "${SOURCE_DIRECTORY}\docs"
    File "${SOURCE_DIRECTORY}\readme.html"
    File "${SOURCE_DIRECTORY}\install.html"
    File "${SOURCE_DIRECTORY}\LGPL"
 
    !insertmacro UNINSTALL.LOG_CLOSE_INSTALL

    StrCpy $InstalledSomething 'true'
SectionEnd

Section "Example Programs"
    SetOutPath '$INSTDIR'
 
    !insertmacro UNINSTALL.LOG_OPEN_INSTALL

    File /r /x .svn /x *.*~ /x *.pyc "${SOURCE_DIRECTORY}\examples"

    !insertmacro UNINSTALL.LOG_CLOSE_INSTALL

    StrCpy $InstalledSomething 'true'
SectionEnd

Function .onInit
    !insertmacro UNINSTALL.LOG_PREPARE_INSTALL
FunctionEnd

Function .onInstSuccess
    Push $0
    Push $1

    StrCmp $InstalledSomething 'true' +1 InstallationEnd
        !insertmacro MUI_STARTMENU_WRITE_BEGIN ProgramFolder

            StrCpy $1 "$SMPROGRAMS\$StartMenuFolder"
            WriteRegStr ${INSTDIR_REG_ROOT} ${INSTDIR_REG_KEY} 'StartMenuPath' "$1"
            CreateDirectory "$1"
            StrCpy $0 "$INSTDIR\docs\index.html"
            IfFileExists "$0" +1 +2
                CreateShortcut "$1\Documents.lnk" "$0"
            StrCpy $0 "$INSTDIR\examples"
            IfFileExists "$0" +1 +2
                CreateShortCut "$1\Examples.lnk" "$0"
            CreateShortCut "$1\Uninstall.lnk" "$INSTDIR\Uninstall.exe"
  
        !insertmacro MUI_STARTMENU_WRITE_END

        WriteRegStr ${INSTDIR_REG_ROOT} "${INSTDIR_REG_KEY}" "InstallDir" "$INSTDIR"
        WriteRegStr ${INSTDIR_REG_ROOT} "${INSTDIR_REG_KEY}" "DisplayName" "${APP_NAME}"
        WriteRegStr ${INSTDIR_REG_ROOT} "${INSTDIR_REG_KEY}" "UninstallString" "${UNINST_EXE}"
    InstallationEnd:

    !insertmacro UNINSTALL.LOG_UPDATE_INSTALL

    Pop $1
    Pop $0
FunctionEnd

Section UnInstall
    push $0

    !insertmacro UNINSTALL.LOG_UNINSTALL "$INSTDIR"
    !insertmacro UNINSTALL.LOG_UNINSTALL "$APPDATA\${APP_NAME}"
    !insertmacro UNINSTALL.LOG_END_UNINSTALL

    ReadRegStr $0 ${INSTDIR_REG_ROOT} ${INSTDIR_REG_KEY} 'StartMenuPath'
    IfErrors StartMenuEnd
        Delete "$0\Documents.lnk"
        Delete "$0\Examples.lnk"
        Delete "$0\Uninstall.lnk"
        RMDir "$0"
        DeleteRegValue ${INSTDIR_REG_ROOT} ${INSTDIR_REG_KEY} 'StartMenuPath'
    StartMenuEnd:
    ClearErrors
    DeleteRegKey /ifempty ${INSTDIR_REG_ROOT} "${INSTDIR_REG_KEY}"

    pop $0
SectionEnd

Function UN.onInit
    !insertmacro UNINSTALL.LOG_BEGIN_UNINSTALL
    StrCpy $InstalledSomething 'false'
FunctionEnd
