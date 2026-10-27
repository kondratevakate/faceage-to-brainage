#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
  echo "usage: $0 MARKDOWN_FILE [...]" >&2
  exit 2
fi

if command -v rg >/dev/null 2>&1 && rg --version >/dev/null 2>&1; then
  extract_urls=(rg -o 'https?://[^)">, ]+')
else
  extract_urls=(grep -Eho 'https?://[^)">, ]+')
fi

"${extract_urls[@]}" "$@" \
  | sort -u \
  | tr '\n' '\0' \
  | xargs -0 -r -P "${LINK_JOBS:-8}" -n 1 bash -c '
      url=$1
      code=$(curl -L -A "Mozilla/5.0" --connect-timeout 8 --max-time 25 \
        -s -o /dev/null -w "%{http_code}" "$url")
      printf "%s\t%s\n" "$code" "$url"
      case "$code" in
        000|404|410|5??) exit 1 ;;
      esac
    ' _
