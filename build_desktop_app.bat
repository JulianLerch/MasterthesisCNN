@echo off
REM Build script for TIFF Simulator V4.0 Desktop App (Windows)
REM ===========================================================

echo.
echo 🔬 Building TIFF Simulator V4.0 Desktop Application
echo ====================================================
echo.

REM Check if PyInstaller is installed
pyinstaller --version >nul 2>&1
if errorlevel 1 (
    echo ❌ PyInstaller not found!
    echo    Installing PyInstaller...
    pip install pyinstaller
)

REM Clean previous builds
echo 🧹 Cleaning previous builds...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

REM Build the application
echo.
echo 🔨 Building application with PyInstaller...
pyinstaller build_app.spec

REM Check if build was successful
if exist "dist\TIFF_Simulator_V4.exe" (
    echo.
    echo ✅ Build successful!
    echo.
    echo 📦 Application location: dist\TIFF_Simulator_V4.exe
    echo.
    echo 🚀 You can now run the application without Python installed!
) else (
    echo.
    echo ❌ Build failed! Check the output above for errors.
    pause
    exit /b 1
)

pause
