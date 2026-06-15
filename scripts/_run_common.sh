#!/usr/bin/env bash

moer_repo_root() {
	local script_dir
	script_dir="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"
	cd "${script_dir}/.." && pwd
}

moer_setup_ld_library_path() {
	export MUSA_HOME="${MUSA_HOME:-/home/o_mabin/.local/musa_toolkits/musa_toolkits_4.2.0}"
	export MUSA_PATH="${MUSA_PATH:-${MUSA_HOME}}"
	export PATH="${MUSA_HOME}/bin:${PATH}"
	local extra_paths=()
	local candidate
	for candidate in \
		"${MUSA_HOME}/lib" \
		"/home/o_mabin/.local/gfortran/usr/lib/x86_64-linux-gnu" \
		"/home/o_mabin/.local/openblas/usr/lib/x86_64-linux-gnu/openblas-pthread" \
		"/home/o_mabin/.local/mudnn/mudnn/lib" \
		"/usr/local/musa/lib"
	do
		if [[ -d "${candidate}" ]]; then
			extra_paths+=("${candidate}")
		fi
	done

	if [[ ${#extra_paths[@]} -gt 0 ]]; then
		local joined
		joined="$(IFS=:; echo "${extra_paths[*]}")"
		export LD_LIBRARY_PATH="${joined}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
	fi
}

moer_prepare_run_dir() {
	local repo_root="$1"
	local run_name="$2"
	local stamp
	stamp="$(date -u +%Y%m%dT%H%M%SZ)"
	export MOER_RESULTS_ROOT="${MOER_RESULTS_ROOT:-${repo_root}/results}"
	export MOER_RUN_DIR="${MOER_RESULTS_ROOT}/${run_name}/${stamp}"
	export MOER_ARTIFACT_ROOT="${MOER_RUN_DIR}/artifacts"
	export MOER_ARTIFACT_DIR="${MOER_ARTIFACT_ROOT}"
	export MOER_LATEST_TP_ARTIFACT_FILE="${MOER_RUN_DIR}/latest_tp_artifact.txt"
	mkdir -p "${MOER_ARTIFACT_ROOT}"
	exec > >(tee -a "${MOER_RUN_DIR}/run.log") 2>&1
	echo "repo_root=${repo_root}"
	echo "run_name=${run_name}"
	echo "run_dir=${MOER_RUN_DIR}"
	echo "artifact_root=${MOER_ARTIFACT_ROOT}"
	echo "MUSA_HOME=${MUSA_HOME:-}"
	echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
}
