#!/usr/bin/env bash
# Idempotent Cloud Agent bootstrap for the film pipelines
# (Middle-earth short and Xiangya marathon promo).
set -euo pipefail

cd "$(dirname "$0")/.."

export DEBIAN_FRONTEND=noninteractive
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  ffmpeg \
  fluidsynth \
  fluid-soundfont-gm \
  sox \
  libegl1 \
  libgl1 \
  libgl1-mesa-dri

# Debian already owns numpy. --ignore-installed places the pinned wheels in
# /usr/local without trying to uninstall distro packages.
sudo python3 -m pip install --break-system-packages --ignore-installed -r requirements.txt
