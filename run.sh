#!/usr/bin/env bash
# 로컬 서버 실행 스크립트
#
# 목적: 워크트리(별도 작업 폴더)에는 자체 .venv(부품 창고)가 없어서
#       그냥 'python main.py' 하면 부품을 못 찾는 일이 반복됨.
#       이 스크립트가 알아서 적절한 .venv를 골라 서버를 띄운다.
#
# 선택 순서:
#   1) 현재 폴더에 .venv 가 있으면 그걸 사용 (워크트리가 자체 venv를 가진 경우)
#   2) 없으면 메인 저장소의 .venv 를 빌려 사용 (워크트리 기본 상황)
#   3) 둘 다 없으면 친절히 안내하고 종료
set -euo pipefail

# 이 스크립트(=프로젝트 루트)가 있는 폴더로 이동
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

pick_python() {
  # 1) 현재 폴더 자체 .venv
  if [ -x "$HERE/.venv/bin/python" ]; then
    echo "$HERE/.venv/bin/python"
    return 0
  fi
  # 2) 메인 저장소(git worktree 목록의 첫 항목)의 .venv
  local main_dir
  main_dir="$(git worktree list --porcelain 2>/dev/null | awk '/^worktree /{print $2; exit}')"
  if [ -n "${main_dir:-}" ] && [ -x "$main_dir/.venv/bin/python" ]; then
    echo "$main_dir/.venv/bin/python"
    return 0
  fi
  return 1
}

if PY="$(pick_python)"; then
  echo "[run.sh] 사용할 파이썬: $PY"
  exec "$PY" main.py
else
  echo "[run.sh] 오류: 사용할 .venv 를 찾지 못했습니다." >&2
  echo "         메인 저장소에서 아래를 한 번 실행한 뒤 다시 시도하세요:" >&2
  echo "         python -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi
