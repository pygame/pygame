rem If you are using the VC++ Toolkit, set the PATH, INCLUDE and LIB to include
rem the Platform SDK and VC++ Toolkit paths of your system.
rem
rem Note that this requires a slightly changed MSVCCompiler class, see
rem http://www.vrplumber.com/programming/mstoolkit/
rem for details
rem 
rem Set PATH=C:\Program files\Microsoft Visual C++ Toolkit 2003\bin;C:\Program files\Microsoft Platform SDK\Bin;%PATH%
rem Set INCLUDE=C:\Program files\Microsoft Visual C++ Toolkit 2003\include;C:\Program files\Microsoft Platform SDK\Include;%INCLUDE%
rem Set LIB=C:\Program files\Microsoft Visual Studio .NET 2003\Vc7\lib;C:\Program files\Microsoft Visual C++ Toolkit 2003\lib;C:\Program files\Microsoft Platform SDK\Lib;%LIB%

rem Set up the python exe path
Set PATH=C:\Python25;%PATH%

rem Delete the previous builds
del /S /Q build
rem Build and install anything.
python setup.py build install

pause
