"""
deploy_backend.py — Deploy Aloo Sahayak backend to Ubuntu server at 10.3.1.204
STRATEGY: Zip the project locally first, upload ONE zip file, unzip on server.
          This is dramatically faster than SFTP-ing thousands of tiny images.

Usage:
    pip install paramiko
    python deploy_backend.py
"""

import os
import sys
import time
import zipfile
import io
import stat

try:
    import paramiko
except ImportError:
    print("[ERROR] paramiko not installed. Run: pip install paramiko")
    sys.exit(1)

# ─── Config ──────────────────────────────────────────────────────────────────
SERVER_IP   = "10.3.1.204"
SERVER_PORT = 22
USERNAME    = "pdm"
PASSWORD    = "aipddm"

LOCAL_PROJECT_DIR  = os.path.dirname(os.path.abspath(__file__))  # script lives inside the project
REMOTE_BASE_DIR    = "/home/pdm/potato_backend"
REMOTE_PROJECT_DIR = f"{REMOTE_BASE_DIR}/Potato-RAG-ChatBot-main"
VENV_DIR           = f"{REMOTE_BASE_DIR}/venv"
SERVICE_NAME       = "potato-backend"

# Files/folders to skip completely during zip creation
SKIP_DIRS  = {".git", "__pycache__", ".ipynb_checkpoints", "Notebooks", ".vscode"}
SKIP_EXTS  = {".pyc", ".pyo", ".ipynb"}
SKIP_FILES = {"chat_history.db"}  # DB is created fresh on server

# NOTE: reference_images/ is INCLUDED (needed for CLIP matching)
# NOTE: data/ PDFs are INCLUDED (needed for RAG)
# NOTE: faiss_index_multimodal/ is INCLUDED (pre-built vector store)


# ─── Environment Variables ─────────────────────────────────────────────────
# Read secrets from the local .env file — never hardcode keys in source!
def load_env_content() -> str:
    """Read the project .env and return its content for uploading to server."""
    env_path = os.path.join(LOCAL_PROJECT_DIR, ".env")
    if not os.path.exists(env_path):
        print(f"[ERROR] .env not found at: {env_path}")
        print("  Create it with your OPENAI_API_KEY, HUGGINGFACEHUB_API_TOKEN, etc.")
        sys.exit(1)
    with open(env_path, "r", encoding="utf-8") as f:
        base = f.read()
    # Append server-side config overrides
    extra = """
# Server-side overrides (appended by deploy script)
DATABASE_TYPE=sqlite
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=False
CORS_ORIGINS=*
MALLOC_ARENA_MAX=2
CUDA_VISIBLE_DEVICES=
TOKENIZERS_PARALLELISM=false
"""
    return base.rstrip() + "\n" + extra


# ─── Systemd Service ──────────────────────────────────────────────────────
SYSTEMD_SERVICE = f"""[Unit]
Description=Aloo Sahayak FastAPI Backend
After=network.target

[Service]
User=pdm
WorkingDirectory={REMOTE_PROJECT_DIR}
Environment="PATH={VENV_DIR}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
EnvironmentFile={REMOTE_PROJECT_DIR}/.env

# Memory Optimizations
Environment="MALLOC_ARENA_MAX=2"
Environment="CUDA_VISIBLE_DEVICES="
Environment="TOKENIZERS_PARALLELISM=false"

ExecStart={VENV_DIR}/bin/python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 1 --log-level warning

# Auto-restart on crash
Restart=always
RestartSec=10

# Hard memory cap - prevents runaway memory leaks
MemoryMax=1500M
MemorySwapMax=0

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=potato-backend

[Install]
WantedBy=multi-user.target
"""

# ─── Helpers ─────────────────────────────────────────────────────────────────

def print_step(msg):
    print(f"\n{'='*65}")
    print(f"  {msg}")
    print(f"{'='*65}")

def run_ssh(client, cmd, ignore_errors=False):
    print(f"  $ {cmd[:120]}")
    stdin, stdout, stderr = client.exec_command(cmd, timeout=600)
    out = stdout.read().decode("utf-8", errors="replace").strip()
    err = stderr.read().decode("utf-8", errors="replace").strip()
    rc  = stdout.channel.recv_exit_status()
    if out:
        for line in out.splitlines()[-20:]:  # show last 20 lines max
            print(f"    {line}")
    if err and rc != 0:
        for line in err.splitlines()[-10:]:
            print(f"    [stderr] {line}")
    if rc != 0 and not ignore_errors:
        print(f"  [WARN] Exit code {rc}")
    return out, err, rc

def run_sudo(client, cmd, password=PASSWORD):
    full_cmd = f"echo '{password}' | sudo -S bash -c \"{cmd}\""
    return run_ssh(client, full_cmd)

def create_zip_in_memory():
    """
    Creates a zip of the project in MEMORY (no disk temp file).
    Skips .git, __pycache__, .pyc files and the sqlite db.
    Returns the bytes of the zip.
    """
    buf = io.BytesIO()
    total_files = 0
    total_size = 0

    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for root, dirs, files in os.walk(LOCAL_PROJECT_DIR):
            # Prune dirs in-place to skip unwanted folders
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

            for fname in files:
                _, ext = os.path.splitext(fname)
                if ext in SKIP_EXTS or fname in SKIP_FILES:
                    continue

                local_path = os.path.join(root, fname)

                # Arc name is relative to parent of LOCAL_PROJECT_DIR
                # so it extracts as Potato-RAG-ChatBot-main/...
                arcname = os.path.relpath(local_path, os.path.dirname(LOCAL_PROJECT_DIR))
                arcname = arcname.replace("\\", "/")  # Zip uses forward slashes

                zf.write(local_path, arcname)
                total_files += 1
                total_size += os.path.getsize(local_path)

                if total_files % 500 == 0:
                    print(f"    ... zipped {total_files} files ({total_size//1024//1024} MB uncompressed)")

    zip_bytes = buf.getvalue()
    print(f"  [OK] Zip complete: {total_files} files | {total_size//1024//1024} MB -> {len(zip_bytes)//1024//1024} MB compressed")
    return zip_bytes

# ─── Main Deployment ──────────────────────────────────────────────────────────

def main():
    print("\n" + "="*65)
    print("  Aloo Sahayak — Backend Deployment Script (Zip Strategy)")
    print(f"  Target: {USERNAME}@{SERVER_IP}:{SERVER_PORT}")
    print("="*65)

    # ── Step 1: Create zip ──
    print_step("1/8  Creating project zip (skipping .git, __pycache__, .db)...")
    zip_bytes = create_zip_in_memory()
    zip_size_mb = len(zip_bytes) / 1024 / 1024

    # ── Step 2: Connect ──
    print_step("2/8  Connecting to server via SSH...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(SERVER_IP, port=SERVER_PORT, username=USERNAME, password=PASSWORD, timeout=30)
    print("  Connected!")

    # ── Step 3: Install unzip & python3-venv ──
    print_step("3/8  Installing system packages (unzip, python3-venv, pip)...")
    run_sudo(client, "apt-get update -qq")
    run_sudo(client, "apt-get install -y python3-venv python3-pip unzip")

    # ── Step 4: Upload zip ──
    print_step(f"4/8  Uploading zip ({zip_size_mb:.1f} MB) via SFTP...")
    sftp = client.open_sftp()
    run_ssh(client, f"mkdir -p {REMOTE_BASE_DIR}")

    remote_zip = f"{REMOTE_BASE_DIR}/project.zip"
    with sftp.file(remote_zip, "wb") as f:
        f.write(zip_bytes)
    print("  [OK] Upload complete!")

    # -- Step 5: Unzip on server --
    print_step("5/8  Unzipping on server...")
    # Remove old version first if exists, then unzip fresh
    run_ssh(client, f"rm -rf {REMOTE_PROJECT_DIR}")
    run_ssh(client, f"cd {REMOTE_BASE_DIR} && unzip -q project.zip")
    run_ssh(client, f"rm {remote_zip}")  # Clean up zip
    print("  Unzip done!")

    # ── Step 6: Write .env ──
    print_step("6/8  Writing .env file (loaded from local .env)...")
    env_content = load_env_content()
    with sftp.file(f"{REMOTE_PROJECT_DIR}/.env", "w") as f:
        f.write(env_content)
    sftp.close()
    print("  .env written.")

    # ── Step 7: Create venv and install deps ──
    print_step("7/8  Creating venv and installing Python dependencies...")
    print("  (This takes 3-10 minutes depending on server speed...)")
    
    # Create venv (reuse if already exists)
    run_ssh(client, f"python3 -m venv {VENV_DIR} --upgrade-deps 2>&1 | tail -5")

    # Upgrade pip + wheel for faster builds
    run_ssh(client, f"{VENV_DIR}/bin/pip install --upgrade pip wheel setuptools --quiet 2>&1 | tail -5")

    # Install CPU-only PyTorch wheel first (much smaller than default)
    print("  Installing PyTorch (CPU only, ~250MB)...")
    out, err, rc = run_ssh(client,
        f"{VENV_DIR}/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet 2>&1 | tail -5"
    )

    # Install main requirements
    print("  Installing main requirements.txt...")
    run_ssh(client,
        f"{VENV_DIR}/bin/pip install -r {REMOTE_PROJECT_DIR}/requirements.txt --quiet 2>&1 | tail -10"
    )
    
    # Install backend requirements
    print("  Installing backend/requirements.txt...")
    run_ssh(client,
        f"{VENV_DIR}/bin/pip install -r {REMOTE_PROJECT_DIR}/backend/requirements.txt --quiet 2>&1 | tail -5"
    )

    print("  [OK] Dependencies installed!")

    # ── Step 8: Systemd service ──
    print_step("8/8  Installing and starting systemd service...")
    
    # Write service file via echo
    service_escaped = SYSTEMD_SERVICE.replace('"', '\\"').replace('\n', '\\n')
    run_ssh(client,
        f"printf '{SYSTEMD_SERVICE.replace(chr(39), chr(39)+chr(92)+chr(39)+chr(39))}' > /tmp/{SERVICE_NAME}.service"
    )
    run_sudo(client, f"cp /tmp/{SERVICE_NAME}.service /etc/systemd/system/{SERVICE_NAME}.service")
    run_sudo(client, "systemctl daemon-reload")
    run_sudo(client, f"systemctl enable {SERVICE_NAME}")
    run_sudo(client, f"systemctl restart {SERVICE_NAME}")

    # Wait for service to start
    print("  Waiting 8 seconds for service startup...")
    time.sleep(8)

    # ── Health check ──
    out, _, _ = run_ssh(client, "curl -s http://127.0.0.1:8000/api/health", ignore_errors=True)
    run_ssh(client, f"systemctl status {SERVICE_NAME} --no-pager -l", ignore_errors=True)
    run_ssh(client, f"journalctl -u {SERVICE_NAME} -n 30 --no-pager", ignore_errors=True)
    
    # RAM usage
    run_ssh(client, "free -m", ignore_errors=True)

    client.close()

    print("\n" + "="*65)
    if "healthy" in out.lower():
        print("  [SUCCESS] DEPLOYMENT SUCCESSFUL!")
        print("  The backend is running and healthy.")
    else:
        print("  [WARN] Service started - health check pending.")
        print("      RAG chain loads lazily on first request (~30s warmup).")
        print(f"      Test: curl http://{SERVER_IP}:8000/api/health")

    print(f"\n  API Base URL:  http://{SERVER_IP}:8000/api")
    print(f"  Docs:          http://{SERVER_IP}:8000/api/docs")
    print(f"  Health:        http://{SERVER_IP}:8000/api/health")
    print(f"  View logs:     journalctl -u {SERVICE_NAME} -f")
    print(f"  Memory usage:  curl http://{SERVER_IP}:8000/api/debug/memory")
    print("="*65)


if __name__ == "__main__":
    main()
