@echo off
rem Make the prebuilt package from the libraries built with
rem msys_build_deps.py. Takes one optional argument, the
rem name of the output directory.
rem
rem This batch file needs python.exe, pexports.exe
rem (found in altbinutils-pe at SourceForge,
rem http://sourceforge.net/projects/mingwrep/) and
rem Visual C's VCVARS32.BAT in the executables search path.
rem Otherwise run make_prebuilt.py first, then the batch
rem files MakeDefs.bat and MakeLibs.bat afterward.

python.exe make_prebuilt.py %1
if errorlevel 1 goto done
if "%1"=="" goto useprebuilt
copy /Y prebuilt-template\readme71.html "%1\readme.html"
cd "%1"
goto domake
:useprebuilt
copy /Y prebuilt-template\readme71.html prebuilt\readme.html
cd prebuilt
:domake
cd lib
CALL Make32.bat
cd ..\..
:done


