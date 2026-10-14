@echo off
cd /d "%~dp0"

echo Criando ambiente virtual...
if not exist .venv (
    python -m venv .venv
)

echo Atualizando pip...
.venv\Scripts\python.exe -m pip install --upgrade pip

echo Instalando bibliotecas...
.venv\Scripts\python.exe -m pip install -r requirements.txt

echo.
echo INSTALACAO CONCLUIDA
pause
