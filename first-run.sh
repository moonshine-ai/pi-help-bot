#!/bin/bash

cat > /usr/local/bin/pi-help-bot.sh <<'EOF'
#!/bin/bash
/home/pete/pi-help-bot/.venv/bin/python /home/pete/pi-help-bot/pi-help-bot.py
EOF
chmod +x /usr/local/bin/pi-help-bot.sh

cat > /etc/systemd/system/pi-help-bot.service <<'EOF'
[Unit]
Description=Pi Help Bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart=/usr/local/bin/pi-help-bot.sh

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable pi-help-bot.service
systemctl start pi-help-bot.service