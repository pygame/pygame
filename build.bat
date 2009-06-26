rem In case you are using the VC++ Toolkit
rem --------------------------------------
rem Set the PATH, INCLUDE and LIB to include the Platform SDK and VC++ Toolkit,
rem and .NET Framework SDK 1.1 VC++ library paths of your system, in case they
rem are not already.
rem

Set MSSdk=C:\Program Files\Microsoft Platform SDK
Set NETSdk=C:\Program Files\Microsoft Visual Studio .NET 2003

Set PATH=%VCToolkitInstallDir%\bin;%MSSdk%\Bin;%MSSdk%\Bin\win64;%PATH%
Set INCLUDE=%VCToolkitInstallDir%\include;%MSSdk%\Include;%INCLUDE%
Set LIB=%NETSdk%\Vc7\lib;%VCToolkitInstallDir%\lib;%MSSdk%\Lib;%LIB%

rem Delete the previous builds
del /S /Q build

rem Build anything.
python setup.py build install

pause
