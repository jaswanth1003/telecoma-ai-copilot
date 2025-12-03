@echo off

echo Installing dependencies...
pip install -r requirements.txt

echo Starting MCP Server...
start cmd /k python MCP_server.py

echo Waiting for backend to launch...
timeout /t 5 >nul

echo Starting Gradio frontend...
python app.py

pause
