#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  echo "Usage: LEFT_IFACE=... RIGHT_IFACE=... bash scripts/setup_network.sh"
  echo "Defaults match the RM75 workstation; configures the two dedicated robot interfaces."
  exit 0
fi
LEFT_IFACE="${LEFT_IFACE:-enx6c1ff7eccd8e}"
RIGHT_IFACE="${RIGHT_IFACE:-enp89s0}"
LEFT_HOST_IP="${LEFT_HOST_IP:-192.168.1.100}"
RIGHT_HOST_IP="${RIGHT_HOST_IP:-192.168.1.101}"
LEFT_ARM_IP="${LEFT_ARM_IP:-192.168.1.18}"
RIGHT_ARM_IP="${RIGHT_ARM_IP:-192.168.1.19}"
for iface in "${LEFT_IFACE}" "${RIGHT_IFACE}"; do ip link show dev "${iface}" >/dev/null; done
sudo ip link set "${LEFT_IFACE}" up
sudo ip link set "${RIGHT_IFACE}" up
sudo ip addr replace "${LEFT_HOST_IP}/32" dev "${LEFT_IFACE}"
sudo ip addr replace "${RIGHT_HOST_IP}/32" dev "${RIGHT_IFACE}"
sudo ip route replace "${LEFT_ARM_IP}/32" dev "${LEFT_IFACE}" src "${LEFT_HOST_IP}"
sudo ip route replace "${RIGHT_ARM_IP}/32" dev "${RIGHT_IFACE}" src "${RIGHT_HOST_IP}"
ip route get "${LEFT_ARM_IP}"
ip route get "${RIGHT_ARM_IP}"
