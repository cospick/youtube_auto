"""pytest 공통 fixture — 테스트용 임시 환경변수와 stub."""

import os
import sys

# 프로젝트 루트를 sys.path에 추가 (tests/ 디렉토리에서 실행 시)
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# JWT_SECRET 누락 시 settings 로드가 실패하므로 import 전에 주입.
os.environ.setdefault("JWT_SECRET", "test-secret-" + "x" * 32)
