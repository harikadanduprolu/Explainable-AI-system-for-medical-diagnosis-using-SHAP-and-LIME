"""
Start the Explainable Medical AI Web Application
================================================

This utility orchestrates the final steps required to run the full-stack
system:

1. Verifies that trained model bundles exist
2. Builds the React SPA if the static assets are missing or stale
3. Launches the FastAPI backend with configurable host/port/logging

Usage examples:
    python start_web_application.py
    python start_web_application.py --host 127.0.0.1 --port 9000
    python start_web_application.py --force-frontend-build
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import uvicorn

PROJECT_ROOT = Path(__file__).parent.resolve()
TRAINED_MODELS_DIR = PROJECT_ROOT / "trained_models"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
FRONTEND_NODE_MODULES = FRONTEND_DIR / "node_modules"
FRONTEND_BUILD_SENTINEL = PROJECT_ROOT / "backend" / "static" / "assets"


def print_banner(host: str, port: int) -> None:
    """Display startup banner with connection information."""
    print("=" * 78)
    print("🚀  Explainable Medical AI - Unified Web Application")
    print("=" * 78)
    print(f"🖥  API Docs : http://{host}:{port}/docs")
    print(f"🌐  Frontend : http://{host}:{port}/app")
    print(f"🩺  Clinical : http://{host}:{port}/clinical")
    print("-" * 78)


def run_command(command, cwd=None, env=None, label: str | None = None) -> bool:
    """Run an external command and report failures with context."""
    printable = label or " ".join(command)
    try:
        subprocess.run(command, check=True, cwd=cwd, env=env)
        return True
    except FileNotFoundError:
        print(f"❌ Command not found while executing '{printable}'. "
              f"Ensure the required tool is installed and on PATH.")
        return False
    except subprocess.CalledProcessError as exc:
        print(f"❌ Command '{printable}' exited with status {exc.returncode}.")
        return False


def ensure_models_present() -> bool:
    """Check that trained model artifacts exist."""
    if not TRAINED_MODELS_DIR.exists():
        print("❌ No trained_models directory found.")
        print("   ➤ Train models with: python train_advanced_models.py")
        return False

    models = sorted(TRAINED_MODELS_DIR.glob("*.pkl"))
    if not models:
        print("❌ No .pkl files detected in trained_models/.")
        print("   ➤ Train models with: python train_advanced_models.py")
        return False

    print(f"✅ Detected {len(models)} trained model bundle(s).")
    return True


def frontend_build_exists() -> bool:
    """Determine if a built React bundle is available."""
    if not FRONTEND_BUILD_SENTINEL.exists():
        return False
    js_candidates = list(FRONTEND_BUILD_SENTINEL.glob("*.js"))
    css_candidates = list(FRONTEND_BUILD_SENTINEL.glob("*.css"))
    return bool(js_candidates and css_candidates)


def ensure_frontend_built(force: bool = False) -> bool:
    """Build the React SPA if assets are missing or a rebuild was requested."""
    if not force and frontend_build_exists():
        print("✅ Frontend bundle already present.")
        return True

    if not FRONTEND_DIR.exists():
        print("❌ Frontend directory missing; ensure the repository was cloned completely.")
        return False

    if shutil.which("npm") is None:
        print("❌ npm is not available on PATH. Install Node.js to build the frontend.")
        return False

    if not FRONTEND_NODE_MODULES.exists():
        print("📦 Installing frontend dependencies (npm install)...")
        if not run_command(["npm", "install"], cwd=FRONTEND_DIR):
            return False

    print("🛠  Building React frontend (npm run build)...")
    return run_command(["npm", "run", "build"], cwd=FRONTEND_DIR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the Explainable Medical AI full-stack web application."
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host interface for Uvicorn (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port for Uvicorn (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (development only)")
    parser.add_argument("--log-level", default="info", help="Uvicorn log level (default: info)")
    parser.add_argument("--skip-model-check", action="store_true", help="Skip trained model verification")
    parser.add_argument("--skip-frontend-build", action="store_true", help="Skip frontend build step")
    parser.add_argument("--force-frontend-build", action="store_true", help="Always rebuild the frontend bundle")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print_banner(args.host, args.port)

    if not args.skip_model_check and not ensure_models_present():
        sys.exit(1)

    if not args.skip_frontend_build and not ensure_frontend_built(force=args.force_frontend_build):
        sys.exit(1)

    try:
        uvicorn.run(
            "backend.main:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level=args.log_level,
        )
    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
    except Exception as exc:  # pragma: no cover - defensive
        print(f"\n❌ Error starting server: {exc}")
        print("   Review the stack trace above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
