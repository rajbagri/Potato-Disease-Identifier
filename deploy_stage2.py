"""
deploy_stage2.py - Continue deployment from where we left off.
Steps 1-6 are already done (zip uploaded, unzipped, .env written).
This script:
  - Runs pip installs via nohup (survives SSH disconnects)
  - Polls until install completes
  - Sets up systemd service
  - Verifies health endpoint
"""

import os
import sys
import time

try:
    import paramiko
except ImportError:
    print("[ERROR] paramiko not installed. Run: pip install paramiko")
    sys.exit(1)

# --- Config ---
SERVER_IP   = "10.3.1.204"
SERVER_PORT = 22
USERNAME    = "pdm"
PASSWORD    = "aipddm"

REMOTE_PROJECT_DIR = "/home/pdm/potato_backend/Potato-RAG-ChatBot-main"
VENV_DIR           = "/home/pdm/potato_backend/venv"
SERVICE_NAME       = "potato-backend"
INSTALL_LOG        = "/home/pdm/potato_backend/pip_install.log"

# --- Systemd Service ---
SYSTEMD_SERVICE = f"""[Unit]
Description=Aloo Sahayak FastAPI Backend
After=network.target

[Service]
User=pdm
WorkingDirectory={REMOTE_PROJECT_DIR}
Environment="PATH={VENV_DIR}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
EnvironmentFile={REMOTE_PROJECT_DIR}/.env
Environment="MALLOC_ARENA_MAX=2"
Environment="CUDA_VISIBLE_DEVICES="
Environment="TOKENIZERS_PARALLELISM=false"
ExecStart={VENV_DIR}/bin/python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 1 --log-level warning
Restart=always
RestartSec=10
MemoryMax=1500M
MemorySwapMax=0
StandardOutput=journal
StandardError=journal
SyslogIdentifier=potato-backend

[Install]
WantedBy=multi-user.target
"""


def print_step(msg):
    print(f"\n{'='*65}")
    print(f"  {msg}")
    print(f"{'='*65}")


def new_client():
    """Create a fresh SSH connection (reconnect after drops)."""
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(SERVER_IP, port=SERVER_PORT, username=USERNAME, password=PASSWORD,
              timeout=30, banner_timeout=30)
    return c


def run(client, cmd, ignore_errors=False, timeout=60):
    """Run command, print output, return (out, err, rc)."""
    print(f"  $ {cmd[:120]}")
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace").strip()
    err = stderr.read().decode("utf-8", errors="replace").strip()
    rc  = stdout.channel.recv_exit_status()
    if out:
        for line in out.splitlines()[-15:]:
            print(f"    {line}")
    if err and rc != 0 and not ignore_errors:
        for line in err.splitlines()[-8:]:
            print(f"    [err] {line}")
    return out, err, rc


def sudo(client, cmd):
    return run(client, f"echo '{PASSWORD}' | sudo -S bash -c \"{cmd}\"")


def main():
    print("\n" + "="*65)
    print("  Aloo Sahayak - Deployment Stage 2 (Resume from pip install)")
    print(f"  Target: {USERNAME}@{SERVER_IP}:{SERVER_PORT}")
    print("="*65)

    # ── Connect ──
    print_step("1/4  Connecting to server...")
    client = new_client()
    print("  Connected!")

    # ── Pip install via nohup (survives SSH drops) ──
    print_step("2/4  Running pip installs via nohup (safe against disconnects)...")

    install_script = f"""#!/bin/bash
set -e
LOG={INSTALL_LOG}
echo "=== Starting pip installs $(date) ===" > $LOG

echo "--- Upgrading pip/wheel ---" >> $LOG
{VENV_DIR}/bin/pip install --upgrade pip wheel setuptools --quiet >> $LOG 2>&1

echo "--- Installing PyTorch CPU ---" >> $LOG
{VENV_DIR}/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet >> $LOG 2>&1

echo "--- Installing main requirements ---" >> $LOG
{VENV_DIR}/bin/pip install -r {REMOTE_PROJECT_DIR}/requirements.txt --quiet >> $LOG 2>&1

echo "--- Installing backend requirements ---" >> $LOG
{VENV_DIR}/bin/pip install -r {REMOTE_PROJECT_DIR}/backend/requirements.txt --quiet >> $LOG 2>&1

echo "=== INSTALL COMPLETE $(date) ===" >> $LOG
"""

    # Write install script to server
    sftp = client.open_sftp()
    script_path = "/home/pdm/potato_backend/install_deps.sh"
    with sftp.file(script_path, "w") as f:
        f.write(install_script)
    sftp.close()
    run(client, f"chmod +x {script_path}")

    # Kill any old install if running
    run(client, "pkill -f install_deps.sh", ignore_errors=True)
    run(client, f"rm -f {INSTALL_LOG}")

    # Launch with nohup so it survives SSH drops
    run(client, f"nohup bash {script_path} > /dev/null 2>&1 &")
    print("  [OK] Install running in background. Polling for completion...")
    print("       (Expected time: 5-15 minutes for PyTorch + all deps)")

    # Poll the log file every 30 seconds
    dots = 0
    while True:
        time.sleep(30)
        dots += 1
        try:
            # Reconnect if session dropped
            try:
                out, _, _ = run(client, f"tail -5 {INSTALL_LOG}", ignore_errors=True)
            except Exception:
                print("  [reconnect] SSH session dropped, reconnecting...")
                client.close()
                client = new_client()
                out, _, _ = run(client, f"tail -5 {INSTALL_LOG}", ignore_errors=True)

            print(f"\n  [Poll {dots}] Log tail:")
            for line in out.splitlines():
                print(f"    {line}")

            if "INSTALL COMPLETE" in out:
                print("\n  [OK] Dependencies installed successfully!")
                break

            # Check if install script is still running
            ps_out, _, _ = run(client, "pgrep -f install_deps.sh || echo NOTRUNNING", ignore_errors=True)
            if "NOTRUNNING" in ps_out and "INSTALL COMPLETE" not in out:
                print("  [ERROR] Install script died unexpectedly. Check log:")
                run(client, f"cat {INSTALL_LOG}")
                print("\n  Exiting. Fix the error above and re-run this script.")
                client.close()
                sys.exit(1)

        except KeyboardInterrupt:
            print("\n  [INTERRUPTED] Install is still running on server.")
            print(f"  SSH in and run: tail -f {INSTALL_LOG}")
            client.close()
            sys.exit(0)

    # ── Systemd service ──
    print_step("3/4  Installing systemd service...")

    service_path = f"/tmp/{SERVICE_NAME}.service"
    sftp = client.open_sftp()
    with sftp.file(service_path, "w") as f:
        f.write(SYSTEMD_SERVICE)
    sftp.close()

    sudo(client, f"cp {service_path} /etc/systemd/system/{SERVICE_NAME}.service")
    sudo(client, "systemctl daemon-reload")
    sudo(client, f"systemctl enable {SERVICE_NAME}")
    sudo(client, f"systemctl restart {SERVICE_NAME}")
    print("  [OK] Service enabled and started!")

    # Wait for startup
    print("  Waiting 10 seconds for startup...")
    time.sleep(10)

    # ── Health check ──
    print_step("4/4  Verifying deployment...")
    out, _, _ = run(client, "curl -s --max-time 10 http://127.0.0.1:8000/api/health", ignore_errors=True)
    
    print("\n  --- Service Status ---")
    run(client, f"systemctl status {SERVICE_NAME} --no-pager -l", ignore_errors=True)
    
    print("\n  --- Last 30 Journal Lines ---")
    run(client, f"journalctl -u {SERVICE_NAME} -n 30 --no-pager", ignore_errors=True)
    
    print("\n  --- Memory Usage (free -m) ---")
    run(client, "free -m", ignore_errors=True)

    client.close()

    print("\n" + "="*65)
    if "healthy" in out.lower():
        print("  [SUCCESS] DEPLOYMENT COMPLETE AND HEALTHY!")
    else:
        print("  [OK] Service started. RAG chain warms up on first request (~30s).")
        print(f"       curl http://{SERVER_IP}:8000/api/health")

    print(f"\n  API:     http://{SERVER_IP}:8000/api")
    print(f"  Docs:    http://{SERVER_IP}:8000/api/docs")
    print(f"  Logs:    journalctl -u {SERVICE_NAME} -f")
    print(f"  Memory:  curl http://{SERVER_IP}:8000/api/debug/memory")
    print("="*65)


if __name__ == "__main__":
    main()
