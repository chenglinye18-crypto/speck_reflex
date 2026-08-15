#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
spikems_dir="${repo_root}/third_party/SpikeMS"
expected_sha="c449c83313423d62a23d92df32dd8d3180680a36"

if [[ ! -e "${spikems_dir}/.git" ]]; then
    echo "SpikeMS submodule is not initialized." >&2
    echo "Run: git submodule update --init --recursive" >&2
    exit 1
fi

actual_sha="$(git -C "${spikems_dir}" rev-parse HEAD)"
if [[ "${actual_sha}" != "${expected_sha}" ]]; then
    echo "Unexpected SpikeMS commit: ${actual_sha}" >&2
    echo "Expected: ${expected_sha}" >&2
    exit 1
fi

if [[ -n "$(git -C "${spikems_dir}" status --short)" ]]; then
    echo "SpikeMS submodule has local modifications." >&2
    git -C "${spikems_dir}" status --short >&2
    exit 1
fi

echo "SpikeMS reference OK: ${actual_sha}"
