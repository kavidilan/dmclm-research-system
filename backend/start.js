#!/usr/bin/env node
/**
 * DMCLM Backend Launcher
 *
 * Starts the FastAPI/Uvicorn server using the root .venv.
 * Designed to be invoked as:
 *   cd backend
 *   npm start
 *
 * The script changes cwd to the project root (one level up from backend/)
 * with backend/ on PYTHONPATH so the relocated src package resolves while
 * keeping the repository root as cwd for existing relative-path behavior.
 */

const path = require("path");
const { spawn } = require("child_process");

const ROOT_DIR = path.resolve(__dirname, "..");
const BACKEND_DIR = path.resolve(__dirname);
const PYTHON = path.join(ROOT_DIR, ".venv", "Scripts", "python.exe");

console.log("=".repeat(70));
console.log("DMCLM Research Intelligence Platform — Python Backend");
console.log("Adaptive Privacy-Preserving Cognitive Memory System");
console.log("=".repeat(70));
console.log(`Root    : ${ROOT_DIR}`);
console.log(`Python  : ${PYTHON}`);
console.log(`Server  : http://127.0.0.1:8000`);
console.log(`Docs    : http://127.0.0.1:8000/docs`);
console.log("=".repeat(70));

const child = spawn(
  PYTHON,
  ["-m", "uvicorn", "src.api_server:app", "--host", "127.0.0.1", "--port", "8000"],
  {
    cwd: ROOT_DIR,
    env: {
      ...process.env,
      PYTHONPATH: [BACKEND_DIR, process.env.PYTHONPATH].filter(Boolean).join(path.delimiter),
    },
    stdio: "inherit",
    windowsHide: false,
  }
);

child.on("error", (err) => {
  console.error(`[DMCLM Backend] Failed to start Python process: ${err.message}`);
  console.error(`Make sure .venv exists at: ${ROOT_DIR}\\.venv`);
  process.exit(1);
});

child.on("close", (code) => {
  if (code !== 0) {
    console.error(`[DMCLM Backend] Python process exited with code ${code}`);
  }
  process.exit(code ?? 0);
});
