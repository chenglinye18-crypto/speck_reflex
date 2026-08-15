#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
base_python="${SPIKEMS_BASE_PYTHON:-${repo_root}/.venv/bin/python}"
reference_env="${SPIKEMS_REFERENCE_ENV:-/home/speck/.venvs/speck_reflex_spikems_reference}"
slayer_root="${SPIKEMS_SLAYER_ROOT:-/home/speck/.cache/spikems_reference/slayerPytorch}"
slayer_commit="01beeeb6a181546d6c6830382ce6086bfc587836"

if [[ ! -x "${base_python}" ]]; then
    echo "Base Python not found: ${base_python}" >&2
    exit 1
fi

if [[ ! -x "${reference_env}/bin/python" ]]; then
    python3 -m venv "${reference_env}"
fi

base_site="$(${base_python} -c 'import site; print(site.getsitepackages()[0])')"
reference_site="$(${reference_env}/bin/python -c 'import site; print(site.getsitepackages()[0])')"
printf '%s\n' "${base_site}" > "${reference_site}/speck_reflex_base_torch.pth"
"${reference_env}/bin/python" -m pip install --disable-pip-version-check ninja==1.13.0

if [[ ! -d "${slayer_root}/.git" ]]; then
    mkdir -p "$(dirname "${slayer_root}")"
    git clone https://github.com/bamsumit/slayerPytorch.git "${slayer_root}"
fi
git -C "${slayer_root}" fetch origin "${slayer_commit}"
git -C "${slayer_root}" checkout --detach "${slayer_commit}"

echo "Reference environment: ${reference_env}"
echo "Official SLAYER source: ${slayer_root} @ ${slayer_commit}"
