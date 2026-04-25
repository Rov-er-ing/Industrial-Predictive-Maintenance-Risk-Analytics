Write-Host "Starting Industrial Predictive Maintenance System (v2.1)..." -ForegroundColor Cyan

# Check if venv exists
if (-not (Test-Path "venv")) {
    Write-Host "Error: venv not found. Please run: python -m venv venv" -ForegroundColor Red
    exit
}

# Run API in a new window
Write-Host "Launching Prediction Engine (FastAPI) on http://localhost:8000..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "& {.\venv\Scripts\activate; python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000; if (!$?) { pause }}"

# Wait a few seconds for API to warm up
Start-Sleep -s 3

# Run Dashboard
Write-Host "Launching Control Center (Streamlit) on http://localhost:8501..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "& {.\venv\Scripts\activate; python -m streamlit run src/ui/dashboard.py; if (!$?) { pause }}"

Write-Host "System Active. Check your browser for the Control Center." -ForegroundColor Cyan
