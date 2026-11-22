# Pin the sourcing shell (and children) to the runner lane's OMNI_CI_CPUSET;
# no-op when unset (forks, local runs).
# note (Jiaxin Deng): docker container options cannot read the runner-local
# .env value, so the reachable layer is the script shell, not --cpuset-cpus.
if [ -n "${OMNI_CI_CPUSET:-}" ]; then
  if ! taskset -pc "${OMNI_CI_CPUSET}" $$ > /dev/null; then
    echo "::error::failed to pin to OMNI_CI_CPUSET='${OMNI_CI_CPUSET}'"
    exit 2
  fi
  # note (Jiaxin Deng): readback catches the kernel silently narrowing the
  # mask (e.g. offline CPUs in the lane spec), mirroring the perf fixture.
  (
    cpu_ids() {
      local IFS=, range first last stride cpu
      local -a ranges ids=()
      read -r -a ranges <<< "$1"
      for range in "${ranges[@]}"; do
        [[ $range =~ ^([0-9]+)(-([0-9]+)(:([0-9]+))?)?$ ]] || return 1
        first=$((10#${BASH_REMATCH[1]}))
        last=$((10#${BASH_REMATCH[3]:-${BASH_REMATCH[1]}}))
        stride=$((10#${BASH_REMATCH[5]:-1}))
        ((first <= last && stride > 0)) || return 1
        for ((cpu=first; cpu<=last; cpu+=stride)); do
          ids[cpu]=1
        done
      done
      printf '%s\n' "${!ids[*]}"
    }

    effective=$(taskset -pc $$) || exit 2
    effective=${effective##* }
    if ! requested_ids=$(cpu_ids "$OMNI_CI_CPUSET") ||
       ! effective_ids=$(cpu_ids "$effective") ||
       [[ $requested_ids != "$effective_ids" ]]; then
      echo "::error::pinned mask '${effective}' != requested OMNI_CI_CPUSET='${OMNI_CI_CPUSET}'"
      exit 2
    fi
  ) || exit 2
fi
