#!/usr/bin/env bash
#
# fix_lightgbm_openmp.sh
# -----------------------------------------------------------------------------
# Make the conda LightGBM dylib resolve its OpenMP runtime (libomp) from the
# CURRENT conda environment instead of an external copy (e.g. Homebrew/MacPorts).
#
# Why: some `defaults`-channel LightGBM builds leak an absolute rpath such as
# `/opt/homebrew/opt/libomp/lib` that ranks ABOVE the env's own libomp. If that
# external libomp exists, LightGBM loads it while numpy/scikit-learn load the
# env's libomp -> TWO OpenMP runtimes in one process. With OMP_WAIT_POLICY=active
# both busy-wait and oversubscribe the cores, making many-tiny-fit workloads
# (e.g. retrain-mode Shapley benchmarks) up to ~20x slower.
#
# This script removes any absolute LC_RPATH that points OUTSIDE the active conda
# env AND actually contains a libomp*.dylib, then re-signs the binary ad-hoc.
# It is idempotent (a clean dylib is left untouched) and macOS-only.
#
# Usage:
#   conda activate <env>
#   bash fix_lightgbm_openmp.sh           # apply the fix
#   bash fix_lightgbm_openmp.sh --revert  # restore from the .bak backup
# -----------------------------------------------------------------------------
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "This script only applies to macOS (dyld/rpath specific). Nothing to do."
    exit 0
fi

for tool in otool install_name_tool codesign python; do
    command -v "$tool" >/dev/null 2>&1 || { echo "ERROR: '$tool' not found in PATH." >&2; exit 1; }
done

: "${CONDA_PREFIX:?Activate the target conda environment first (CONDA_PREFIX is unset).}"

# --- locate the compiled LightGBM dylib (handles both packaging layouts) ------
LIB="$(python - <<'PY'
import glob, os
try:
    import lightgbm
except ImportError:
    print(""); raise SystemExit
roots = [os.path.join(os.path.dirname(lightgbm.__file__), "lib"),
         os.path.join(os.environ.get("CONDA_PREFIX", ""), "lib")]
hits = []
for r in roots:
    hits += glob.glob(os.path.join(r, "lib_lightgbm*.dylib"))
    hits += glob.glob(os.path.join(r, "liblightgbm*.dylib"))
print(hits[0] if hits else "")
PY
)"

if [[ -z "$LIB" || ! -f "$LIB" ]]; then
    echo "ERROR: could not locate the LightGBM dylib (is lightgbm importable in this env?)." >&2
    exit 1
fi
echo "LightGBM dylib: $LIB"

# --- revert mode --------------------------------------------------------------
if [[ "${1:-}" == "--revert" ]]; then
    if [[ -f "$LIB.bak" ]]; then
        cp -p "$LIB.bak" "$LIB"
        codesign --force --sign - "$LIB" >/dev/null 2>&1 || true
        echo "Reverted from $LIB.bak and re-signed."
    else
        echo "No backup found at $LIB.bak; nothing to revert."
    fi
    exit 0
fi

# --- collect LC_RPATH entries -------------------------------------------------
# (avoid `mapfile`: it is a bash 4+ builtin and macOS ships bash 3.2)
RPATHS=()
while IFS= read -r _rp; do
    [ -n "$_rp" ] && RPATHS+=("$_rp")
done < <(otool -l "$LIB" | awk '/^ *cmd LC_RPATH/{f=1;next} f&&/^ *path /{print $2; f=0}')

echo "Current rpaths:"
for p in "${RPATHS[@]}"; do echo "    $p"; done

# --- decide which rpaths are offending ----------------------------------------
# Offending = absolute path, NOT under $CONDA_PREFIX, and contains a libomp dylib.
OFFENDING=()
for p in "${RPATHS[@]}"; do
    [[ "$p" == /* ]] || continue                 # skip @loader_path/@rpath/etc.
    [[ "$p" == "$CONDA_PREFIX"/* ]] && continue   # keep env-internal paths
    if compgen -G "$p/libomp*.dylib" >/dev/null 2>&1; then
        OFFENDING+=("$p")
    fi
done

if [[ ${#OFFENDING[@]} -eq 0 ]]; then
    echo "No external OpenMP rpath found. The dylib is already clean -- nothing to do."
    exit 0
fi

echo "Offending external OpenMP rpaths to remove:"
for p in "${OFFENDING[@]}"; do echo "    $p"; done

# --- backup, patch, re-sign ---------------------------------------------------
if [[ ! -f "$LIB.bak" ]]; then
    cp -p "$LIB" "$LIB.bak"
    echo "Backup written: $LIB.bak"
else
    echo "Backup already exists: $LIB.bak (keeping it)"
fi

for p in "${OFFENDING[@]}"; do
    install_name_tool -delete_rpath "$p" "$LIB"
    echo "Deleted rpath: $p"
done

codesign --force --sign - "$LIB"
echo "Re-signed (ad-hoc): $LIB"

# --- verify -------------------------------------------------------------------
echo "Rpaths after:"
otool -l "$LIB" | awk '/^ *cmd LC_RPATH/{f=1;next} f&&/^ *path /{print "    "$2; f=0}'

echo "Verifying loaded OpenMP runtimes (expect exactly one, inside the env)..."
python - <<'PY'
import ctypes, os, sys
libc = ctypes.CDLL(None)
libc._dyld_image_count.restype = ctypes.c_uint32
libc._dyld_get_image_name.restype = ctypes.c_char_p
libc._dyld_get_image_name.argtypes = [ctypes.c_uint32]
import numpy as np, lightgbm as lgb
lgb.LGBMRegressor(verbosity=-1).fit(np.random.rand(80, 4), np.random.rand(80))
names = [libc._dyld_get_image_name(i).decode() for i in range(libc._dyld_image_count())]
omp = sorted({n for n in names if any(k in n.lower() for k in ("libomp.dylib", "libgomp", "libiomp"))})
print("  loaded OpenMP runtimes:")
for n in omp:
    print("   ", n)
prefix = os.environ.get("CONDA_PREFIX", "")
ok = len(omp) == 1 and omp[0].startswith(prefix)
print("  RESULT:", "OK -- single env-internal runtime" if ok else "STILL MULTIPLE / external -- check above")
sys.exit(0 if ok else 2)
PY
echo "Done."
