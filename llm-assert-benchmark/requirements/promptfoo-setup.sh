#!/usr/bin/env bash
# Install Promptfoo globally via npm.
# Requires Node.js 18+ installed on the system.
set -euo pipefail

if ! command -v npm &> /dev/null; then
    echo "Error: npm not found. Install Node.js 18+ first."
    exit 1
fi

npm install -g promptfoo@latest
echo "Promptfoo installed: $(npx promptfoo --version)"
