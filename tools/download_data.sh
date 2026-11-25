#!/usr/bin/env bash
# Fetch the datasets into data/datasets (or $IA3DGS_DATA). Nothing is redistributed by this repository;
# each dataset is downloaded from its original source and stays under its own terms of use.
#
#   bash tools/download_data.sh kodak        # 3 PNGs, < 2 MB           (Tier A)
#   bash tools/download_data.sh mipnerf360   # 12.5 GB zip, keeps garden + bicycle (images_4, sparse) only
#   bash tools/download_data.sh nerf         # prints the manual step for NeRF-Synthetic (Google Drive)
#   bash tools/download_data.sh all
set -euo pipefail
cd "$(dirname "$0")/.."
D="${IA3DGS_DATA:-${IA3DGS_ROOT:+$IA3DGS_ROOT/datasets}}"
D="${D:-data/datasets}"
mkdir -p "$D"
what="${1:-all}"

kodak() {
  # Kodak Lossless True Color Image Suite, https://r0k.us/graphics/kodak/
  mkdir -p "$D/kodak"
  for i in 05 15 23; do
    [ -f "$D/kodak/kodim$i.png" ] || curl -fL -o "$D/kodak/kodim$i.png" "https://r0k.us/graphics/kodak/kodak/kodim$i.png"
  done
  echo "kodak: $(ls "$D/kodak" | tr '\n' ' ')"
}

mipnerf360() {
  # Mip-NeRF 360 (Barron et al., CVPR 2022), https://jonbarron.info/mipnerf360/
  mkdir -p "$D/mipnerf360"
  Z="$D/mipnerf360/360_v2.zip"
  [ -f "$Z" ] || curl -fL -C - -o "$Z" "http://storage.googleapis.com/gresearch/refraw360/360_v2.zip"
  for s in garden bicycle; do
    unzip -n -q "$Z" "$s/images_4/*" "$s/sparse/*" "$s/poses_bounds.npy" -d "$D/mipnerf360"
  done
  echo "mipnerf360: garden, bicycle extracted; delete $Z to free 12.5 GB"
}

nerf() {
  # NeRF-Synthetic (Mildenhall et al., ECCV 2020), https://github.com/bmild/nerf
  if [ -d "$D/nerf_synthetic/lego" ] && [ -d "$D/nerf_synthetic/chair" ] && [ -d "$D/nerf_synthetic/ficus" ]; then
    echo "nerf_synthetic: present"; return
  fi
  cat <<EOF
NeRF-Synthetic is distributed through Google Drive and must be fetched by hand:
  1. open https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1
  2. download nerf_synthetic.zip
  3. unzip nerf_synthetic.zip 'nerf_synthetic/lego/*' 'nerf_synthetic/chair/*' 'nerf_synthetic/ficus/*' -d "$D"
EOF
}

case "$what" in
  kodak) kodak ;;
  mipnerf360) mipnerf360 ;;
  nerf) nerf ;;
  all) kodak; nerf; mipnerf360 ;;
  *) echo "usage: $0 [kodak|mipnerf360|nerf|all]"; exit 1 ;;
esac
