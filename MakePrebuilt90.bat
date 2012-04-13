@echo off
rem Make the Win32 prebuilt file for Pythons 2.6 and up.
rem Takes an optional argument, the prebuilt directory path.
rem Requires Visual Studio, with VCVARS32.BAT is on the
rem executable search path. Python must also be on the search
rem path. msys_link_VC_2008_dlls.py requires pexports.exe 0.43.

set DESTDIR=%1
if "%DESTDIR%" == "" set DESTDIR=prebuilt

python make_prebuilt.py %DESTDIR%
if errorlevel 1 goto aborted

copy /Y prebuilt-template\readme90.html "%DESTDIR%\readme.html"

set DESTDIR=%DESTDIR%\lib
deltree /Y %DESTDIR%\msvcr71
python msys_link_VC_2008_dlls.py -d %DESTDIR% --all
if errorlevel 1 goto aborted

CALL VCVARS32.BAT
cd %DESTDIR%
CALL MakeLibs.bat
if errorlevel 1 goto aborted

echo '
echo =====================================
echo Prebuilt directory built successfully
goto done

:aborted
echo '
echo *** Error: failed to complete

:done
