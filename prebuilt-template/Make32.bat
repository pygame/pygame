@echo off
rem Builds the Visual C import 32 bit libraries.
rem Requires pexports (see MakeDefs.bat) and VCVARS32.BAT on
rem the executable search path.

CALL VCVARS32.BAT
CALL MakeDefs.bat
CALL MakeLibs.bat
