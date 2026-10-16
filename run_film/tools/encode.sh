#!/usr/bin/env bash
# Encode out/frames2 + music/bgm.wav into the WeChat, HQ and review deliverables.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
ART=${ART:-/opt/cursor/artifacts}
IN=(-framerate 24 -i "$ROOT/out/frames2/f%05d.jpg" -i "$ROOT/music/bgm.wav")
V=(-c:v libx264 -preset slow -profile:v high -level 4.1 -pix_fmt yuv420p -g 48)
A=(-c:a aac -ar 48000 -movflags +faststart -shortest)
WX=(-b:v 2900k -maxrate 3500k -bufsize 5800k -passlogfile /tmp/xiangya_x264)

ffmpeg -y -v error "${IN[@]}" "${V[@]}" "${WX[@]}" -pass 1 -an -f mp4 /dev/null
ffmpeg -y -v error "${IN[@]}" "${V[@]}" "${WX[@]}" -pass 2 "${A[@]}" -b:a 160k "$ART/xiangya_promo_v2_60s_wechat.mp4"
ffmpeg -y -v error "${IN[@]}" "${V[@]}" -b:v 6000k -maxrate 8000k -bufsize 12000k "${A[@]}" -b:a 192k \
  "$ART/xiangya_promo_v2_60s_hq.mp4"
ffmpeg -y -v error "${IN[@]}" -vf scale=540:960 -c:v libx264 -crf 26 -pix_fmt yuv420p "${A[@]}" -b:a 128k \
  "$ROOT/out/review_v2c_540.mp4"
