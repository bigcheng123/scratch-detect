 @echo off
 cd /d d:/code/scratch-detect
 python -u main.py > log.txt 2>&1
 ::python main.py
 
 move log.txt d:/code/scratch-detect/log
 cd d:/code/scratch-detect/log
 set NOW_TIME=%date:~0,4%-%date:~5,2%-%date:~8,2%-%time:~0,2%-%time:~3,2%-%time:~6,2%
 chcp 65001 
 ren log.txt %NOW_TIME%log.txt
 timeout /t 1

 cd /d d:/code/scratch-detect
 start restart.bat
 exit
 ::pause