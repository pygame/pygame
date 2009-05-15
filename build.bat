rem In case you are using the VC++ Toolkit
rem --------------------------------------
rem Set the PATH, INCLUDE and LIB to include the Platform SDK and VC++ Toolkit,
rem and .NET Framework SDK 1.1 VC++ library paths of your system, in case they
rem  are not already.
rem

Set PATH=%VCToolkitInstallDir%\bin;%MSSdk%\Bin;%PATH%
Set INCLUDE=%VCToolkitInstallDir%\include;%MSSdk%\Include;%INCLUDE%

rem German defaults below
rem Set LIB=C:\Programme\Microsoft Visual Studio .NET 2003\Vc7\lib;%VCToolkitInstallDir%\lib;%MSSdk%\Lib;%LIB%

rem English defaults below
Set LIB=C:\Program files\Microsoft Visual Studio .NET 2003\Vc7\lib;%VCToolkitInstallDir%\lib;%MSSdk%\Lib;%LIB%

rem Delete the previous builds
del /S /Q build

rem Build anything.
python setup.py build install

pause
