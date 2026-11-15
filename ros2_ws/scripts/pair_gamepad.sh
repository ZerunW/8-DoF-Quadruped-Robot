#!/usr/bin/env bash
# Pair, trust and connect a PS5 DualSense over Bluetooth (BlueZ), without input:
# robotdog> pixi run pair_gamepad. Put the controller in pairing mode first (PS + Create).
# Checks the one-time setup (bluez, ClassicBondedOnly=false, modules uhid and hid_playstation)
# and prints the sudo commands for anything missing.
#
#   NAME=...        controller name to look for (default "DualSense Wireless Controller")
#   SCAN_SECONDS=.. how long to scan (default 20)
set -u

NAME="${NAME:-DualSense Wireless Controller}"
SCAN_SECONDS="${SCAN_SECONDS:-20}"

say()  { printf 'pair_gamepad: %s\n' "$*"; }
fail() { printf 'pair_gamepad: ERROR: %s\n' "$*" >&2; exit 1; }

# bluetoothctl with a clean environment, a timeout and no stdin.
btc() { env -i PATH=/usr/bin:/bin HOME="${HOME:-/root}" timeout "${BT_TIMEOUT:-20}" bluetoothctl "$@" 2>&1 </dev/null; }

input_device_exists() { grep -q "$NAME" /proc/bus/input/devices 2>/dev/null; }
js_device() { ls /dev/input/js* 2>/dev/null | head -1; }

# ---- one-time system setup checks ----------------------------------
missing=()
if ! [ -x /usr/bin/bluetoothctl ]; then
  missing+=("sudo apt-get install -y bluez")
fi
if ! grep -Eqi '^[[:space:]]*ClassicBondedOnly[[:space:]]*=[[:space:]]*false' /etc/bluetooth/input.conf 2>/dev/null; then
  missing+=("# BlueZ 5.72 rejects HID from non-bonded devices; the DualSense pairing does not bond
sudo sed -i '/^\\[General\\]/a ClassicBondedOnly=false' /etc/bluetooth/input.conf
sudo systemctl restart bluetooth")
fi
if ! lsmod | grep -q '^uhid\b' || ! lsmod | grep -q '^hid_playstation\b' \
   || ! grep -qs hid_playstation /etc/modules-load.d/gamepad.conf; then
  missing+=("sudo modprobe uhid hid_playstation
printf 'uhid\\nhid_playstation\\n' | sudo tee /etc/modules-load.d/gamepad.conf")
fi
if [ "${#missing[@]}" -gt 0 ]; then
  say "one-time setup missing. Run these on the Pi, then run pair_gamepad again:"
  for c in "${missing[@]}"; do printf '\n%s\n' "$c"; done
  exit 1
fi

# ---- adapter ------------------------------------------------------------------------------
out="$(btc show)"
if [ -z "$out" ] || printf '%s' "$out" | grep -qi "No default controller\|not available\|org.freedesktop.DBus.Error"; then
  fail "no Bluetooth adapter / BlueZ is not reachable ($(printf '%s' "$out" | head -1)). Check: systemctl status bluetooth ; rfkill list"
fi
if printf '%s' "$out" | grep -qi "Powered: no"; then
  say "adapter is off, powering on"
  btc power on | grep -qi "succeeded" || fail "could not power the adapter on (try: rfkill unblock bluetooth)."
fi

find_mac() {
  local m
  m="$(btc devices | grep -F "$NAME" | head -1 | awk '{print $2}')"
  [ -n "$m" ] || m="$(btc devices | grep -F "Wireless Controller" | head -1 | awk '{print $2}')"
  printf '%s' "$m"
}

# ---- already connected and the kernel input device exists ----------------------
mac="$(find_mac)"
if [ -n "$mac" ] && btc info "$mac" | grep -q "Connected: yes" && input_device_exists; then
  say "OK: already connected ($mac), input device present ($(js_device)). Nothing to do."
  exit 0
fi

# ---- forget the old record, scan, pair, trust, connect -----------------------
if [ -n "$mac" ]; then
  say "removing the old record $mac"
  btc remove "$mac" >/dev/null
fi
say "scanning ${SCAN_SECONDS} s for \"$NAME\": hold PS + Create until the light flashes now"
timeout "$((SCAN_SECONDS + 10))" env -i PATH=/usr/bin:/bin HOME="${HOME:-/root}" \
  bluetoothctl --timeout "$SCAN_SECONDS" scan on >/dev/null 2>&1 </dev/null
mac="$(find_mac)"
[ -n "$mac" ] || fail "\"$NAME\" not found. Hold PS + Create until the light bar flashes quickly and run again."
say "found $mac"

say "pairing"
res="$(btc --agent NoInputNoOutput pair "$mac")"
printf '%s' "$res" | grep -qi "Pairing successful\|AlreadyExists" || fail "pairing failed: $(printf '%s' "$res" | tail -2 | tr '\n' ' ')
Put the controller in pairing mode again (PS + Create) and rerun."
btc trust "$mac" | grep -qi "trust succeeded\|Trusted: yes" && say "trusted" || say "WARNING: could not trust the device"
say "connecting"
res="$(btc connect "$mac")"
printf '%s' "$res" | grep -qi "Connection successful" || fail "connect failed: $(printf '%s' "$res" | tail -2 | tr '\n' ' ')
Press PS to wake the controller and rerun."

# ---- verify the kernel input device ----------------------------------------------------------
for _ in 1 2 3 4 5 6 7 8 9 10; do input_device_exists && break; sleep 1; done
if input_device_exists; then
  say "OK: $NAME connected ($mac), input device $(js_device). Next: robotdog> pixi run launch_teleop"
else
  fail "connected, but no \"$NAME\" input device in /proc/bus/input/devices. Is /etc/bluetooth/input.conf ClassicBondedOnly=false (bluetooth restarted) and hid_playstation loaded? (journalctl -u bluetooth | grep -i bonded)"
fi
