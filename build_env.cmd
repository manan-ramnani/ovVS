@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=amd64 >nul
call "C:\Program Files (x86)\Intel\oneAPI\compiler\latest\env\vars.bat" >nul
call "C:\Program Files (x86)\Intel\oneAPI\mkl\latest\env\vars.bat" >nul
call "C:\Program Files (x86)\Intel\oneAPI\tbb\latest\env\vars.bat" >nul
call "C:\Program Files (x86)\Intel\oneAPI\ocloc\latest\env\vars.bat" >nul 2>&1
cd /d C:\Personal\ioVS
%*
