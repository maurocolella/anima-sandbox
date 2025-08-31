#!/usr/bin/env bash
set -euo pipefail

# Run the native desktop app
# Usage: ./run.local.sh [--release] [additional cargo args]

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Forward all args to cargo run
exec cargo run -p desktop "$@"
