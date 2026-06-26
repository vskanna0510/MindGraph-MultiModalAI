@echo off
REM Initialize Flutter platform folders when SDK is available
cd /d "%~dp0..\flutter_app"
where flutter >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
  echo Flutter SDK not found. Install from https://flutter.dev
  exit /b 1
)
flutter create --org com.mindgraph --project-name mindgraph_plus_plus .
flutter pub get
echo Flutter project initialized.
