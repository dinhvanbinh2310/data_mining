@echo off
REM Script chạy Streamlit app trên Windows

REM Kích hoạt virtual environment
call .venv\Scripts\activate.bat

REM Chạy Streamlit app
streamlit run src/app/app.py

pause

