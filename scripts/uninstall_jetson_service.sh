#!/usr/bin/env bash
# ==============================================================================
# DriveCV Jetson Systemd Service Uninstaller
# ==============================================================================

set -e

SERVICE_NAME="drivecv-jetson"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

if [ "$EUID" -ne 0 ]; then
    echo "[ERROR] Please run this uninstall script with sudo:"
    echo "  sudo bash scripts/uninstall_jetson_service.sh"
    exit 1
fi

echo "[INFO] Stopping ${SERVICE_NAME}.service..."
systemctl stop "${SERVICE_NAME}.service" || true

echo "[INFO] Disabling ${SERVICE_NAME}.service..."
systemctl disable "${SERVICE_NAME}.service" || true

if [ -f "${SERVICE_FILE}" ]; then
    echo "[INFO] Removing ${SERVICE_FILE}..."
    rm -f "${SERVICE_FILE}"
fi

echo "[INFO] Reloading systemd daemon..."
systemctl daemon-reload

echo "✅ DriveCV Jetson Service uninstalled successfully."
