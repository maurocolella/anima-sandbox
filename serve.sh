#!/usr/bin/env bash
set -euo pipefail

# Serve the web app via trunk
# Usage: ./serve.sh [trunk args]

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/web"

if ! command -v trunk >/dev/null 2>&1; then
  echo "Error: trunk is not installed." >&2
  echo "Install with: cargo install trunk" >&2
  exit 127
fi

# Ensure wasm32 target exists (optional soft check)
if ! rustup target list --installed | grep -q '^wasm32-unknown-unknown$'; then
  echo "Error: wasm32-unknown-unknown target not installed." >&2
  echo "Install with: rustup target add wasm32-unknown-unknown" >&2
  exit 127
fi

# Forward all args to trunk serve
exec trunk serve --open "$@"
