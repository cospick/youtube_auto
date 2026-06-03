# 프로젝트 규칙

## 실행 명령

```bash
./run.sh                                                   # 로컬 서버 (http://localhost:8000) — venv 자동 선택(워크트리는 메인 .venv 사용)
python create_admin.py admin@example.com 닉네임 비밀번호     # 관리자 계정 생성
docker build -t ai-shorts . && docker run -p 8000:8000 --env-file .env ai-shorts  # Docker
```

## 필수 환경변수

```
GEMINI_API_KEY    # 없으면 이미지/텍스트 생성 불가
JWT_SECRET        # 없으면 서버 시작 거부 — python -c "import secrets; print(secrets.token_hex(32))"
```

## 코드 수정 규칙

- 수정 전 관련 파일을 모두 읽고 호출 관계와 영향 범위를 파악한 뒤 진행한다.
- 수정 후 변경된 부분과 연결된 전체 데이터 흐름을 추적하여 놓친 수정이 없는지 검토한다.
- 검토 결과 문제가 있으면 즉시 수정하고, 없으면 "정합성 검토 완료"로 보고한다.

## 답변 스타일

- 모든 설명은 비전공자도 이해할 수 있게 작성한다. 기술 용어를 쓰면 바로 이어서 "즉, ~한다는 뜻" 같은 1줄 풀이를 붙인다.
- 복잡한 개념은 일상 비유(요리·택배·편지 등)로 풀어낸다.
- 코드 수정 보고서에도 "무슨 코드가 바뀌었나"뿐 아니라 "사용자 입장에서 뭐가 달라지는가"를 함께 설명한다.

## 비직관적 규칙 (코드만 봐서는 모르는 것)

- 새 DB 컬럼 추가 시 `db/database.py`의 `_MIGRATIONS` 딕셔너리에 반드시 등록해야 한다. `Base.metadata.create_all()`만으로는 기존 테이블에 컬럼이 추가되지 않는다.
- 사용자 API 키가 설정되어 있으면 서버 기본 키로 폴백하지 않는다. 키가 틀려도 사용자 키만 사용한다.
- SSE 엔드포인트(`/api/jobs/{id}/stream`)는 FastAPI Depends를 사용할 수 없어서 쿠키에서 직접 토큰을 검증한다.
- Railway 배포 시 영상 조립 중인 FFmpeg subprocess는 SIGTERM을 받지 못한다. 배포 전 진행 중 작업 확인 필요.
- `R2_BUCKET_NAME`이 빈 값이면 R2 관련 코드가 전부 no-op으로 동작한다 (로컬 전용 모드).
