# Railway 배포 가이드

이 문서를 위에서 아래로 순서대로 따라하면 배포가 완료됩니다.

---

## 1단계: 사전 준비 (로컬에서)

### 1-1. GitHub에 push

Railway는 GitHub 리포지토리에서 코드를 가져옵니다.

```bash
git add -A
git commit -m "배포 준비: Dockerfile, R2, 사용자별 API 키, BGM 업로드 등"
git push origin main
```

아직 GitHub 리포지토리가 없다면:
1. https://github.com/new 에서 새 리포지토리 생성 (Private 추천)
2. `git remote add origin https://github.com/본인계정/youtube_auto.git`
3. `git push -u origin main`

> BGM 파일은 git에 포함하지 않습니다. 배포 후 사용자가 직접 BGM을 업로드하면 R2에 저장됩니다.

---

## 2단계: Railway 가입 + 프로젝트 생성

### 2-1. Railway 가입

1. https://railway.app 접속
2. **"Login"** 클릭 → **"Login with GitHub"** 선택 (가장 간편)
3. GitHub 계정으로 로그인

### 2-2. Pro 플랜 전환 (필수)

Railway 무료 플랜은 제한이 심합니다. Pro 플랜 필요.

1. 로그인 후 왼쪽 하단 **프로필 아이콘** 클릭
2. **"Settings"** → **"Billing"**
3. **"Upgrade to Pro"** 클릭 ($20/월, 크레딧 $20 포함)
4. 카드 등록

### 2-3. 새 프로젝트 생성

1. Railway 대시보드에서 **"New Project"** 클릭
2. **"Empty Project"** 선택

---

## 3단계: PostgreSQL 추가

1. 프로젝트 화면에서 **"+ New"** 클릭
2. **"Database"** → **"Add PostgreSQL"** 클릭
3. PostgreSQL 서비스가 생성됨 (30초 정도 소요)
4. 생성된 PostgreSQL 서비스 클릭 → **"Variables"** 탭
5. `DATABASE_URL`이 자동 생성된 것 확인 (나중에 앱에 연결할 것)

---

## 4단계: 앱 서비스 추가

### 4-1. GitHub 리포지토리 연결

1. 프로젝트 화면에서 **"+ New"** 클릭
2. **"GitHub Repo"** 선택
3. 본인의 `youtube_auto` 리포지토리 선택
4. Railway가 자동으로 `Dockerfile`을 감지하고 빌드 시작 (아직 환경변수가 없어서 실패할 수 있음 — 정상)

### 4-2. 환경변수 설정

**앱 서비스 클릭** → **"Variables"** 탭 → 아래 변수들을 하나씩 추가:

#### 필수 (이것 없으면 서버 시작 불가)

| 변수 | 값 | 설명 |
|------|-----|------|
| `DATABASE_URL` | **"Add Reference"로 PostgreSQL 연결** (아래 설명) | DB 연결 |
| `JWT_SECRET` | 터미널에서 `python -c "import secrets; print(secrets.token_hex(32))"` 실행한 결과 | 보안 키 |

> `GEMINI_API_KEY`, `TYPECAST_API_KEY`, `FAL_KEY`는 설정하지 않습니다. 외부 생성 API는 서버 기본 키 없이 사용자별 저장 키만 사용합니다.

**DATABASE_URL 연결 방법:**
1. Variables 탭에서 **"Add Reference Variable"** 클릭 (또는 "+" 버튼)
2. Source: PostgreSQL 서비스 선택
3. Variable: `DATABASE_URL` 선택
4. 이렇게 하면 Railway가 자동으로 PostgreSQL 주소를 연결해줌

#### 서비스 URL (필수)

배포 후 Railway가 부여하는 도메인을 확인해야 합니다.

1. 앱 서비스 → **"Settings"** 탭 → **"Networking"** 섹션
2. **"Generate Domain"** 클릭 → `xxxx.up.railway.app` 형태의 도메인 생성
3. 이 도메인을 복사해서 아래 변수에 사용:

| 변수 | 값 | 설명 |
|------|-----|------|
| `BASE_URL` | `https://xxxx.up.railway.app` | 서비스 URL |
| `GOOGLE_REDIRECT_URI` | `https://xxxx.up.railway.app/api/auth/google/callback` | Google OAuth |
| `KAKAO_REDIRECT_URI` | `https://xxxx.up.railway.app/api/auth/kakao/callback` | Kakao OAuth |

#### 선택 (기능별로 필요할 때 추가)

| 변수 | 필요 시점 | 발급처 |
|------|-----------|--------|
| `INVITE_CODE` | 회원가입 제한하고 싶을 때 | 본인이 정한 코드 |
| `GOOGLE_CLIENT_ID` | Google 로그인 사용 시 | [Google Cloud Console](https://console.cloud.google.com/apis/credentials) |
| `GOOGLE_CLIENT_SECRET` | Google 로그인 사용 시 | 위와 동일 |
| `KAKAO_CLIENT_ID` | 카카오 로그인 사용 시 | [Kakao Developers](https://developers.kakao.com) |
| `KAKAO_CLIENT_SECRET` | 카카오 로그인 사용 시 | 위와 동일 |
| `SMTP_USER` | 비밀번호 재설정 이메일 사용 시 | Gmail 주소 |
| `SMTP_PASSWORD` | 비밀번호 재설정 이메일 사용 시 | [Gmail 앱 비밀번호](https://myaccount.google.com/apppasswords) |
| `R2_ENDPOINT_URL` | Railway/PostgreSQL 운영 시 필수 | Cloudflare R2 대시보드 |
| `R2_ACCESS_KEY_ID` | Railway/PostgreSQL 운영 시 필수 | Cloudflare R2 API Token |
| `R2_SECRET_ACCESS_KEY` | Railway/PostgreSQL 운영 시 필수 | 위와 동일 |
| `R2_BUCKET_NAME` | Railway/PostgreSQL 운영 시 필수 | 본인이 만든 버킷 이름 |

### 4-3. 재배포

환경변수를 모두 설정한 후:
1. 앱 서비스 → **"Deployments"** 탭
2. 가장 최근 배포 옆 **"Redeploy"** 클릭
3. 빌드 로그 확인 (2~5분 소요)
4. **"AI 쇼츠 자동 제작 웹앱 시작!"** 메시지가 보이면 성공

---

## 5단계: Google OAuth 설정 (선택)

Google 로그인을 사용하려면:

1. [Google Cloud Console](https://console.cloud.google.com) 접속
2. 프로젝트 선택 (또는 새 프로젝트 생성)
3. **"API 및 서비스"** → **"사용자 인증 정보"** → **"+ 사용자 인증 정보 만들기"** → **"OAuth 클라이언트 ID"**
4. 애플리케이션 유형: **"웹 애플리케이션"**
5. **승인된 리디렉션 URI**에 추가:
   - `https://xxxx.up.railway.app/api/auth/google/callback`
6. 생성된 **클라이언트 ID**와 **클라이언트 보안 비밀번호**를 Railway 환경변수에 입력

---

## 6단계: Kakao OAuth 설정 (선택)

1. [Kakao Developers](https://developers.kakao.com) 접속 → 로그인
2. **"내 애플리케이션"** → **"애플리케이션 추가하기"**
3. 앱 이름 입력 → 생성
4. **앱 키** → **REST API 키** 복사 → Railway `KAKAO_CLIENT_ID`에 입력
5. **제품 설정** → **카카오 로그인** → **활성화 ON**
6. **Redirect URI** 추가: `https://xxxx.up.railway.app/api/auth/kakao/callback`
7. **동의항목** → **개인정보** → **카카오계정(이메일)** → **필수 동의**로 설정
8. **앱 키** → **보안** → **Client Secret** 생성 → Railway `KAKAO_CLIENT_SECRET`에 입력

---

## 7단계: Cloudflare R2 설정 (필수)

R2를 설정하면:
- 생성된 이미지/영상이 재배포 후에도 보존됩니다
- 사용자가 업로드한 BGM이 영구 보관됩니다
- 30일 히스토리로 재다운로드가 가능합니다

> Railway/PostgreSQL 운영 환경에서는 R2 4개 환경변수가 모두 설정되어야 이미지/영상 생성과 업로드가 시작됩니다.

### 7-1. R2 버킷 생성

1. [Cloudflare 대시보드](https://dash.cloudflare.com) 접속 → 로그인
2. 왼쪽 메뉴 → **"R2 Object Storage"**
3. **"Create Bucket"** 클릭
4. 버킷 이름: `ai-shorts` (또는 원하는 이름)
5. 위치: **APAC** (아시아-태평양, 한국 사용자에게 빠름)
6. 생성 완료

### 7-2. API 토큰 생성

1. R2 페이지 → 오른쪽 상단 **"Manage R2 API Tokens"**
2. **"Create API Token"** 클릭
3. 권한: **Object Read & Write**
4. 특정 버킷 지정: `ai-shorts`
5. 생성 후 표시되는 값 복사:
   - **Access Key ID** → Railway `R2_ACCESS_KEY_ID`
   - **Secret Access Key** → Railway `R2_SECRET_ACCESS_KEY`
6. **Account ID** 확인 (R2 메인 페이지 오른쪽에 표시됨)
   - Railway `R2_ENDPOINT_URL` = `https://{Account_ID}.r2.cloudflarestorage.com`
7. Railway `R2_BUCKET_NAME` = `ai-shorts`

### 7-3. 30일 자동 삭제 설정

1. 버킷 클릭 → **"Settings"** 탭
2. **"Object lifecycle rules"** → **"Add rule"**
3. Prefix: `jobs/` (영상/이미지만 삭제, BGM은 제외)
4. Action: **Delete** after **30 days**
5. 저장

> 중요: prefix를 `jobs/`로 설정하면 BGM(`bgm/`)은 삭제되지 않습니다. BGM은 사용자의 에셋이므로 영구 보관됩니다.

---

## 8단계: 관리자 계정 생성

배포가 완료된 후, Railway에서 관리자 계정을 생성합니다.

### 방법 A: Railway CLI (추천)

```bash
# Railway CLI 설치 (한 번만)
npm install -g @railway/cli

# Railway 로그인
railway login

# 프로젝트 연결
railway link

# 관리자 생성 명령 실행
railway run python create_admin.py admin@example.com 관리자닉네임 비밀번호
```

### 방법 B: Railway 웹 콘솔

1. 앱 서비스 클릭 → **"Settings"** 탭
2. **"Railway Shell"** 또는 **"Execute Command"**
3. `python create_admin.py admin@example.com 관리자닉네임 비밀번호` 입력

---

## 9단계: 배포 확인 테스트

브라우저에서 `https://xxxx.up.railway.app` 접속 후 순서대로 확인:

| # | 테스트 | 확인 사항 |
|---|--------|-----------|
| 1 | 헬스체크 | `https://xxxx.up.railway.app/health` → `{"status":"ok"}` |
| 2 | 로그인 페이지 | 메인 URL 접속 → 로그인 페이지로 리다이렉트 |
| 3 | 회원가입 | 이메일+비밀번호 가입 (초대코드 설정했으면 코드 입력) |
| 4 | 로그인 | 가입한 계정으로 로그인 → 메인 페이지 표시 |
| 5 | API 키 설정 | 상단 "설정" 클릭 → Gemini API 키 입력 → 저장 |
| 6 | BGM 업로드 | BGM 단계에서 "+" 클릭 → MP3 업로드 → 목록에 표시 |
| 7 | 제목 생성 | 주제 입력 → 제목 3~4개 생성 |
| 8 | 이미지 생성 | 워크플로우 진행 → 이미지 미리보기 |
| 9 | 영상 생성 | 확인 → TTS + 영상 조립 → 다운로드 |
| 10 | 작업 이력 | "작업 이력" 페이지 → D-day 표시 확인 |

### 문제가 생기면

1. Railway 앱 → **"Deployments"** → 최근 배포 클릭 → **빌드 로그** 확인
2. 앱 서비스 → **"Logs"** 탭 → **런타임 에러** 확인
3. 흔한 문제:
   - `JWT_SECRET` 미설정 → "JWT_SECRET이 설정되지 않았습니다" 에러
   - `DATABASE_URL` 미연결 → SQLite로 동작 (PostgreSQL이 아님)
   - 폰트 못 찾음 → 자막 없는 영상 생성 (Dockerfile에 fonts-noto-cjk 확인)
   - BGM 업로드 실패 → R2 환경변수 확인 (4개 모두 설정 필요)

---

## 배포 후 운영 팁

| 작업 | 방법 |
|------|------|
| 코드 업데이트 | `git push origin main` → Railway 자동 재배포 |
| 로그 확인 | Railway 앱 → Logs 탭 |
| DB 직접 확인 | Railway PostgreSQL → Data 탭 또는 `railway run python -c "..."` |
| 30일 만료 처리 | `railway run python cleanup_old_jobs.py` (아래 cron 자동화 권장) |
| 환경변수 변경 | Railway 앱 → Variables 탭 → 값 수정 → 자동 재배포 |

---

## 30일 자동 정리 (cron) — 카드 B 사용 시 권장

카드 B 사용자의 "영상 다운로드" 또는 "새 영상 만들기" 클릭 시 중간 작업 파일은 즉시 정리되지만, 사용자가 두 버튼 모두 누르지 않은 채 영상이 남아 있는 경우엔 R2 용량이 계속 점유됩니다. `cleanup_old_jobs.py`는 완료된 지 30일 지난 영상을 자동으로 R2에서 제거합니다.

Railway에서 자동화하려면:

1. **Cron Plugin 추가** — Railway 프로젝트 → "New" → "Database / Plugin" → **"Cron"** 선택
2. **스케줄 설정** — 매일 새벽 4시(KST)에 실행하려면:
   - Schedule: `0 19 * * *` (UTC, KST 04:00에 해당)
   - Command: `python cleanup_old_jobs.py --days 30`
   - 동일 서비스 컨테이너를 reuse하므로 환경변수(`DATABASE_URL`, `R2_*` 등)가 자동 주입됩니다
3. **로그 확인** — Cron 서비스의 Logs 탭에서 매일 실행 결과 확인. 만료 처리된 job 수가 출력됩니다.

cron을 설정하지 않으면 30일이 지나도 R2의 영상 파일이 그대로 남아 무료 한도(10GB)를 점차 잠식합니다. 사용자 수가 늘면 점유량이 빠르게 증가하므로 반드시 설정하길 권합니다.

로컬 환경(개발용)에서는 필요할 때만 수동으로 `python cleanup_old_jobs.py` 실행하면 됩니다.

---

## 비용 요약

| 항목 | 월 비용 | 비고 |
|------|---------|------|
| Railway Pro | $20 (크레딧 $20 포함) | 웹앱+DB 합쳐 ~$10~17 → 크레딧 내 |
| Cloudflare R2 | $0 | 무료 티어 10GB, 100명이면 충분 |
| Google OAuth | $0 | 무료 |
| Kakao OAuth | $0 | 무료 |
| Gmail SMTP | $0 | 무료 (일 500통 제한) |
| **합계** | **~$20/월** | API 비용은 수강생 본인 부담 |
