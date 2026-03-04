@echo off
echo ============================================================
echo Setting up React Frontend
echo ============================================================
echo.

cd frontend

echo Installing dependencies...
call npm install

echo.
echo ============================================================
echo Setup Complete!
echo ============================================================
echo.
echo To start the frontend:
echo   cd frontend
echo   npm run dev
echo.
echo The app will be available at http://localhost:3000
echo.
echo Make sure the backend API is running on http://localhost:8000
echo ============================================================
pause
