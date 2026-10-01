@echo off
cd /d "%~dp0"
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)
call venv\Scripts\activate
echo Installing requirements...
python -m pip install -r requirements.txt
echo Starting app...
python -m streamlit run app.py
pause
