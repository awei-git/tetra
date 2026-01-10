# Local Hostnames

This setup maps local hostnames (e.g., `tetra-app`) to local ports (e.g., `127.0.0.1:8000`) without typing a port.

## 1) Define host mappings

Edit `tetra/config/hostmap.txt`:

```
tetra-app=8000
other-app=6000
```

## 2) Add hostnames to /etc/hosts

Run in Terminal (requires sudo):

```
echo "127.0.0.1 tetra-app other-app" | sudo tee -a /etc/hosts
```

## 3) Run the local proxy (user-level)

```
python scripts/local_host_proxy.py --port 8080
```

This listens on `127.0.0.1:8080` and routes by hostname.

## 4) Redirect port 80 to 8080 (so you can type `http://tetra-app`)

Run in Terminal (requires sudo):

```
sudo tee /etc/pf.anchors/tetra-app >/dev/null <<'EOF'
rdr pass on lo0 inet proto tcp from any to 127.0.0.1 port 80 -> 127.0.0.1 port 8080
EOF

sudo sh -c 'grep -q "tetra-app" /etc/pf.conf || printf "\nrdr-anchor \"tetra-app\"\nload anchor \"tetra-app\" from \"/etc/pf.anchors/tetra-app\"\n" >> /etc/pf.conf'

sudo pfctl -f /etc/pf.conf
sudo pfctl -E
```

## Optional: run proxy at login

```
cp config/launchd/com.tetra.local-proxy.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.tetra.local-proxy.plist
```

## Notes

- This only touches port 80 and does not affect other localhost ports.
- If `/etc/pf.conf` contains `set skip on lo0`, the redirect will not apply to loopback.
