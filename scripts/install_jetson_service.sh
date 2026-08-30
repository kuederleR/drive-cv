#!/usr/bin/env bash
# ==============================================================================
# DriveCV Jetson Orin Nano Super Systemd Service Installer
# Installs and enables a systemd service to run DriveCV Docker container on boot.
# ==============================================================================

set -e

SERVICE_NAME="drivecv-jetson"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "======================================================================"
echo " 🚀 DriveCV Jetson Systemd Service Installer"
echo "======================================================================"
echo "[INFO] Target Project Location: ${PROJECT_DIR}"

# Check for root / sudo permissions
if [ "$EUID" -ne 0 ]; then
    echo "[ERROR] Please run this install script with sudo:"
    echo "  sudo bash scripts/install_jetson_service.sh"
    exit 1
fi

# Detect Docker Compose binary command
DOCKER_COMPOSE_BIN=""
if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
    DOCKER_COMPOSE_BIN="$(which docker) compose"
elif command -v docker-compose >/dev/null 2>&1; then
    DOCKER_COMPOSE_BIN="$(which docker-compose)"
else
    echo "[ERROR] Neither 'docker compose' nor 'docker-compose' command was found."
    echo "Please ensure Docker and Docker Compose are installed on your Jetson."
    exit 1
fi

echo "[INFO] Detected Docker Compose binary: ${DOCKER_COMPOSE_BIN}"

# Create Systemd Service File
echo "[INFO] Creating systemd service file at ${SERVICE_FILE}..."
cat <<EOF > "${SERVICE_FILE}"
[Unit]
Description=DriveCV 3D ADAS Perception Service (NVIDIA Jetson)
After=docker.service network-online.target
Requires=docker.service

[Service]
Type=simple
WorkingDirectory=${PROJECT_DIR}
ExecStart=${DOCKER_COMPOSE_BIN} -f docker-compose.jetson.yml up --build
ExecStop=${DOCKER_COMPOSE_BIN} -f docker-compose.jetson.yml down
Restart=always
RestartSec=5s
Environment=PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

[Install]
WantedBy=multi-user.target
EOF

# Set permissions and reload systemd daemon
chmod 644 "${SERVICE_FILE}"
echo "[INFO] Reloading systemd daemon..."
systemctl daemon-reload

echo "[INFO] Enabling ${SERVICE_NAME}.service to launch automatically on boot..."
systemctl enable "${SERVICE_NAME}.service"

echo "======================================================================"
echo " ✅ DriveCV Jetson Service Installed Successfully!"
echo "======================================================================"
echo " Service Name: ${SERVICE_NAME}.service"
echo ""
echo " Useful Commands:"
echo "   Start Service:   sudo systemctl start ${SERVICE_NAME}"
echo "   Stop Service:    sudo systemctl stop ${SERVICE_NAME}"
echo "   Restart Service: sudo systemctl restart ${SERVICE_NAME}"
echo "   Status:          sudo systemctl status ${SERVICE_NAME}"
echo "   View Logs:       sudo journalctl -u ${SERVICE_NAME} -f"
echo "======================================================================"

# Prompt to start service immediately if run interactively
if [ -t 0 ]; then
    read -p "Would you like to start the DriveCV service now? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "[INFO] Starting ${SERVICE_NAME}.service..."
        systemctl start "${SERVICE_NAME}.service"
        echo "[INFO] Service started! Check status with: sudo systemctl status ${SERVICE_NAME}"
    fi
fi
