@echo off
cd /d "%~dp0"
python program.py
if errorlevel 1 pause
