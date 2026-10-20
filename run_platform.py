"""
DMCLM Research Platform Launcher
Starts the FastAPI Backend (Port 8000) and the React Frontend (Port 5173).
"""

import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT_DIR / "frontend"
BACKEND_DIR = ROOT_DIR / "backend"


def main():
    print("=" * 70)
    print("DMCLM Research Intelligence Platform")
    print("Adaptive Privacy-Preserving Cognitive Memory System")
    print("Member 4: Kavishka | SLIIT ID: IT23208230")
    print("=" * 70)

    # 1. Start FastAPI Backend Server
    print("\n[1/2] Starting FastAPI Backend on http://localhost:8000 ...")
    backend_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "src.api_server:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000"
    ]
    backend_env = os.environ.copy()
    backend_python_path = str(BACKEND_DIR)
    if backend_env.get("PYTHONPATH"):
        backend_python_path += os.pathsep + backend_env["PYTHONPATH"]
    backend_env["PYTHONPATH"] = backend_python_path
    backend_proc = subprocess.Popen(
        backend_cmd,
        cwd=ROOT_DIR,
        env=backend_env,
        shell=False
    )

    # Allow backend to bind socket
    time.sleep(1.5)

    # 2. Start Vite Frontend Server
    print("\n[2/2] Starting Vite Frontend on http://localhost:5173 ...")
    npm_cmd = "npm.cmd" if sys.platform.startswith("win") else "npm"
    frontend_cmd = [npm_cmd, "run", "dev"]
    frontend_proc = subprocess.Popen(
        frontend_cmd,
        cwd=FRONTEND_DIR,
        shell=True
    )

    time.sleep(2)
    print("\n" + "=" * 70)
    print("[SUCCESS] DMCLM Platform is LIVE!")
    print("  -> Frontend: http://localhost:5173/")
    print("  -> Backend API: http://localhost:8000/docs")
    print("  -> Collector DB: C:\\Users\\Admin\\AppData\\Local\\DesktopResearchCollector\\collector.db (Read-Only)")
    print("=" * 70)
    print("\nPress Ctrl+C to terminate both servers.")

    try:
        backend_proc.wait()
        frontend_proc.wait()
    except KeyboardInterrupt:
        print("\nStopping DMCLM platform...")
        backend_proc.terminate()
        frontend_proc.terminate()
        print("Shutdown complete.")


if __name__ == "__main__":
    main()
