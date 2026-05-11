#!/bin/bash
set -euo pipefail

RUN_USER="${SUDO_USER:-${USER:-pi}}"
RUN_HOME="$(getent passwd "$RUN_USER" | cut -d: -f6)"
RUN_UID="$(id -u "$RUN_USER")"
PROJECT_DIR="$RUN_HOME/pi-help-bot"

if [[ ! -d "$PROJECT_DIR" ]]; then
    echo "Project directory not found: $PROJECT_DIR" >&2
    exit 1
fi

# Ensure the user's PipeWire/PulseAudio session is running at boot so the
# system service (running as $RUN_USER) can reach the ALSA->PipeWire bridge
# via XDG_RUNTIME_DIR. Without lingering, /run/user/$RUN_UID and PipeWire
# only exist after an interactive login.
sudo loginctl enable-linger "$RUN_USER"

sudo tee /usr/local/bin/pi-help-bot.sh > /dev/null <<EOF
#!/bin/bash
exec $PROJECT_DIR/.venv/bin/python $PROJECT_DIR/pi-help-bot.py
EOF
sudo chmod +x /usr/local/bin/pi-help-bot.sh

sudo tee /etc/systemd/system/pi-help-bot.service > /dev/null <<EOF
[Unit]
Description=Pi Help Bot
After=network-online.target sound.target user@${RUN_UID}.service
Wants=network-online.target user@${RUN_UID}.service

[Service]
Type=simple
User=$RUN_USER
Group=$RUN_USER
SupplementaryGroups=audio input plugdev
WorkingDirectory=$PROJECT_DIR
Environment=XDG_RUNTIME_DIR=/run/user/${RUN_UID}
Environment=PULSE_SERVER=unix:/run/user/${RUN_UID}/pulse/native
ExecStart=/usr/local/bin/pi-help-bot.sh
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl unmask pi-help-bot.service 2>/dev/null || true
sudo systemctl enable pi-help-bot.service
sudo systemctl restart pi-help-bot.service
