#!/bin/bash
set -e

# Install systemd user service for auto-starting vLLM on boot

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_NAME="vllm-gemma4-26b.service"
USER_SERVICE_DIR="$HOME/.config/systemd/user"

echo "=================================="
echo "vLLM Systemd Service Installer"
echo "=================================="

mkdir -p "$USER_SERVICE_DIR"

# Generate service file with actual paths
cat > "$USER_SERVICE_DIR/$SERVICE_NAME" << EOF
[Unit]
Description=vLLM Gemma-4 26B Uncensored (NVFP4)
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=$REPO_DIR
ExecStart=$REPO_DIR/scripts/start.sh
ExecStop=-docker stop vllm-gemma4-26b
ExecStop=-docker rm vllm-gemma4-26b

[Install]
WantedBy=default.target
EOF

# Reload systemd
systemctl --user daemon-reload

# Enable service (auto-start on login/boot)
systemctl --user enable "$SERVICE_NAME"

echo ""
echo "✅ Service installed: $SERVICE_NAME"
echo ""
echo "Commands:"
echo "  Start now:   systemctl --user start $SERVICE_NAME"
echo "  Stop:        systemctl --user stop $SERVICE_NAME"
echo "  Status:      systemctl --user status $SERVICE_NAME"
echo "  Disable:     systemctl --user disable $SERVICE_NAME"
echo ""

# Check if lingering is enabled
if ! loginctl show-user "$USER" --property=Linger 2>/dev/null | grep -q "yes"; then
    echo "⚠️  Important: lingering is NOT enabled for user $USER."
    echo "   Without lingering, the service will only start on login, not at boot."
    echo ""
    echo "   To enable boot-time auto-start, run:"
    echo "     sudo loginctl enable-linger $USER"
    echo ""
else
    echo "✅ Linger is enabled — service will auto-start at boot."
fi

echo "To start vLLM immediately, run: systemctl --user start $SERVICE_NAME"
