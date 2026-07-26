#!/usr/bin/env bash
# Deploy custom Synthea modules and extensions to the main Synthea repository.
#
# Usage: ./deploy.sh [--synthea-root /path/to/synthea]
#
# This copies all JSON modules from synthea_extensions/modules/ into
# synthea/src/main/resources/modules/ so they are picked up by Synthea's
# module loader at runtime.
#
# The modified Generator.java (with mimicPriorDir support) must be compiled
# separately via: cd <synthea_root> && ./gradlew uberJar

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTENSIONS_DIR="${SCRIPT_DIR}"

# Default Synthea root: two levels up from this script
DEFAULT_SYNTHEA="${SCRIPT_DIR}/../../synthea"
SYNTHEA_ROOT="${1:-${DEFAULT_SYNTHEA}}"

if [[ "$#" -ge 2 && "$1" == "--synthea-root" ]]; then
    SYNTHEA_ROOT="$2"
fi

MODULES_SRC="${EXTENSIONS_DIR}/modules"
MODULES_DST="${SYNTHEA_ROOT}/src/main/resources/modules"
JAVA_SRC="${EXTENSIONS_DIR}/java"
JAVA_DST="${SYNTHEA_ROOT}/src/main/java"

if [[ ! -d "${MODULES_DST}" ]]; then
    echo "ERROR: Synthea modules directory not found: ${MODULES_DST}"
    echo "Check --synthea-root path or run from the correct directory."
    exit 1
fi

echo "Deploying custom modules from ${MODULES_SRC} → ${MODULES_DST}"

# List of custom modules to deploy
MODULES=(
    "hyperlipidemia_general.json"
    "major_depressive_disorder.json"
    "anxiety_disorder.json"
    "alcohol_use_disorder.json"
    "hepatitis_c.json"
    "liver_cirrhosis.json"
    "bipolar_disorder.json"
    "ptsd_general.json"
    "gerd.json"
    "hypertension_general.json"
    "diabetes_type2_general.json"
    "copd_general.json"
    "ckd_general.json"
    "heart_failure_general.json"
)

for module in "${MODULES[@]}"; do
    src="${MODULES_SRC}/${module}"
    dst="${MODULES_DST}/${module}"
    if [[ -f "${src}" ]]; then
        cp "${src}" "${dst}"
        echo "  ✓ deployed: ${module}"
    else
        echo "  ✗ missing:  ${module}"
    fi
done

echo "Deploying conditioned-generation Java extensions"
cp "${JAVA_SRC}/App.java" "${JAVA_DST}/App.java"
cp "${JAVA_SRC}/org/mitre/synthea/engine/Generator.java" \
  "${JAVA_DST}/org/mitre/synthea/engine/Generator.java"
echo "  ✓ deployed: App.java"
echo "  ✓ deployed: Generator.java"

echo ""
echo "Synthea extensions deployed."
echo ""
echo "Next step: build the modified Synthea JAR:"
echo "  cd ${SYNTHEA_ROOT}"
echo "  ./gradlew uberJar"
echo ""
echo "Then run conditioned generation from Python:"
echo "  from ClinVerse.pipeline.case_builder import build_cases_conditioned_batch"
