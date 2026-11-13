#!/usr/bin/env bash
# Harden a fresh Ubuntu VM before anything runs on it (section 8 of the plan, "VM and network"):
#
#   sudo bash scripts/harden_vm.sh                  # every step
#   sudo STEPS="ssh updates" bash scripts/harden_vm.sh   # only some, as the container test does
#
#   ssh       keys only, no root login, no passwords, short login window
#   updates   automatic security updates, with a reboot at 05:00 UTC when one needs it
#   firewall  the host firewall lets in only the tailnet and the cluster's own pod and service
#             ranges. Oracle's Ubuntu image rejects everything else, which is kept
#   tailscale the admin tunnel: the only way in once Oracle's security list is closed
#
# What only the owner can do, in the Oracle console, after `tailscale ssh` works: delete the
# security list's ingress rule for port 22 from 0.0.0.0/0, leaving no inbound rule at all.
# Then the VM takes no traffic from the internet, and the k3s API, Argo CD, Grafana, Prometheus,
# and Redpanda are reachable only over the tailnet (k8s/README.md).
set -euo pipefail

STEPS="${STEPS:-ssh updates firewall tailscale}"
[ "$(id -u)" -eq 0 ] || { echo "run with sudo" >&2; exit 1; }

step_ssh() {
  # A low number so it's read before cloud-init's own file, which can turn passwords back on:
  # for each setting, sshd keeps the first value it reads.
  install -d -m 0755 /etc/ssh/sshd_config.d
  cat > /etc/ssh/sshd_config.d/10-wikiguard.conf <<'EOF'
# Written by scripts/harden_vm.sh
PermitRootLogin no
PasswordAuthentication no
KbdInteractiveAuthentication no
PubkeyAuthentication yes
AuthenticationMethods publickey
PermitEmptyPasswords no
X11Forwarding no
AllowAgentForwarding no
AllowTcpForwarding no
MaxAuthTries 3
LoginGraceTime 30
ClientAliveInterval 300
ClientAliveCountMax 2
EOF
  chmod 0644 /etc/ssh/sshd_config.d/10-wikiguard.conf
  install -d -m 0755 /run/sshd
  sshd -t
  if command -v systemctl >/dev/null && systemctl is-active --quiet ssh 2>/dev/null; then
    systemctl reload ssh
  fi
  echo "ssh: $(sshd -T 2>/dev/null | grep -E '^(permitrootlogin|passwordauthentication|kbdinteractiveauthentication) ' | tr '\n' ' ')"
}

step_updates() {
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y -qq unattended-upgrades >/dev/null
  cat > /etc/apt/apt.conf.d/20auto-upgrades <<'EOF'
APT::Periodic::Update-Package-Lists "1";
APT::Periodic::Unattended-Upgrade "1";
APT::Periodic::AutocleanInterval "7";
EOF
  # Security updates only, and a reboot when a kernel update needs one. k3s and every pod come
  # back by themselves, and the ingestor repairs the gap from recentchanges (section 2).
  cat > /etc/apt/apt.conf.d/52wikiguard-upgrades <<'EOF'
Unattended-Upgrade::Allowed-Origins {
        "${distro_id}:${distro_codename}-security";
        "${distro_id}ESMApps:${distro_codename}-apps-security";
        "${distro_id}ESM:${distro_codename}-infra-security";
};
Unattended-Upgrade::Automatic-Reboot "true";
Unattended-Upgrade::Automatic-Reboot-Time "05:00";
Unattended-Upgrade::Remove-Unused-Dependencies "true";
EOF
  unattended-upgrade --dry-run >/dev/null 2>&1 && echo "updates: unattended-upgrades configured"
}

step_firewall() {
  # Oracle's Ubuntu image ends INPUT with a REJECT. k3s's pods (10.42.0.0/16) and services
  # (10.43.0.0/16) talk to the node too, and the tailnet is the admin path, so those go in front.
  # Nothing public is added.
  apt-get install -y -qq iptables-persistent >/dev/null 2>&1 || true
  for rule in "-i tailscale0" "-s 10.42.0.0/16" "-s 10.43.0.0/16"; do
    # shellcheck disable=SC2086 # the rule is meant to split into arguments
    iptables -C INPUT $rule -j ACCEPT 2>/dev/null || iptables -I INPUT 1 $rule -j ACCEPT
  done
  command -v netfilter-persistent >/dev/null && netfilter-persistent save >/dev/null
  echo "firewall: tailnet, pod, and service ranges allowed; the image's other rules unchanged"
}

step_tailscale() {
  if ! command -v tailscale >/dev/null; then
    # Tailscale's own signed apt repository, through its documented install script.
    curl -fsSL https://tailscale.com/install.sh | sh
  fi
  echo "tailscale: installed. Now run: sudo tailscale up --ssh  (it prints a login link)"
}

for step in $STEPS; do
  "step_${step}"
done
