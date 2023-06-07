cms:: Locations of things
::
set TERASTITCHER=C:\terastitcher\terastitcher
set Parastitcher=C:\terastitcher\parastitcher.py


:: Set active directory to path of batch file parent folder
cd /d %~dp0

SET WORKINGDIR=%CD%
for %%* in (.) do SET DIRNAME=%%~n*
SET OUTPUTDIR=%DIRNAME%_Stitched
mkdir "..\%OUTPUTDIR%"


call activate parastitcher

%TERASTITCHER% -1 --volin="%WORKINGDIR%" --ref1=H --ref2=V --ref3=D --vxl1=0.420 --vxl2=0.420 --vxl3=1 --projout=xml_import.xml --sparse_data 

mpiexec -np 12 python -m mpi4py %Parastitcher% -2 --projin=.\xml_import.xml --projout=.\xml_displcomp.xml --subvoldim=100 --sV=25 --sH=25 --sD=25

%TERASTITCHER% -3 --projin=.\xml_displcomp.xml

%TERASTITCHER% -4 --projin=.\xml_displproj.xml --threshold=0.7

%TERASTITCHER% -5 --projin=.\xml_displthres.xml

mpiexec -np 12 python -m mpi4py %Parastitcher% -6 --projin=.\xml_merging.xml --volout="..\%OUTPUTDIR%" --volout_plugin="TiledXY|2Dseries" --slicewidth=100000 --sliceheight=150000 

pause