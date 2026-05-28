/* 인증 유틸리티 - 모든 페이지에서 공유 */

/* ── 테마 (다크/라이트) ── */
const ICON_SUN = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="20" height="20" style="pointer-events:none"><circle cx="12" cy="12" r="4"/><path d="M12 2v2"/><path d="M12 20v2"/><path d="m4.93 4.93 1.41 1.41"/><path d="m17.66 17.66 1.41 1.41"/><path d="M2 12h2"/><path d="M20 12h2"/><path d="m6.34 17.66-1.41 1.41"/><path d="m19.07 4.93-1.41 1.41"/></svg>';
const ICON_MOON = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="20" height="20" style="pointer-events:none"><path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/></svg>';

function setThemeIcon() {
    const btn = document.getElementById('theme-toggle-btn');
    if (!btn) return;
    btn.innerHTML = document.body.classList.contains('dark') ? ICON_MOON : ICON_SUN;
}

function toggleTheme() {
    const isDark = document.body.classList.toggle('dark');
    localStorage.setItem('theme', isDark ? 'dark' : 'light');
    setThemeIcon();
}

// 이벤트 위임: 버튼이 다시 렌더링되어도 동작
document.addEventListener('click', (e) => {
    if (!e.target.closest) return;
    const btn = e.target.closest('#theme-toggle-btn');
    if (btn) {
        e.preventDefault();
        toggleTheme();
        return;
    }

    // 로고/홈 클릭 시 진행 중 작업 보호
    const homeAnchor = e.target.closest('.navbar-logo, a.navbar-link[href="/"]');
    if (homeAnchor && hasUnsavedWork()) {
        const ok = confirm('진행 중인 작업이 모두 사라집니다.\n정말 처음 화면으로 돌아갈까요?');
        if (!ok) {
            e.preventDefault();
            e.stopPropagation();
        }
    }
});

function hasUnsavedWork() {
    // index.html(app.js)에서만 _generationMode가 설정됨. 다른 페이지엔 영향 없음.
    return !!window._generationMode;
}

let currentUser = null;
let _authReady;
const authReady = new Promise(resolve => { _authReady = resolve; });

/**
 * fetch 래퍼: credentials 자동 포함 + 401 시 토큰 갱신 또는 로그인 리다이렉트
 */
let _refreshing = null; // 토큰 갱신 중복 방지

async function authFetch(url, options = {}) {
    options.credentials = 'same-origin';
    let resp = await fetch(url, options);

    // 401이면 토큰 갱신 시도 (동시 요청 시 1번만 갱신)
    if (resp.status === 401) {
        if (!_refreshing) {
            _refreshing = fetch('/api/auth/refresh', {
                method: 'POST',
                credentials: 'same-origin',
            }).finally(() => { _refreshing = null; });
        }
        const refreshResp = await _refreshing;
        if (refreshResp && refreshResp.ok) {
            resp = await fetch(url, options);
        } else {
            window.location.href = '/static/login.html';
            return new Response('{}', { status: 401, headers: { 'Content-Type': 'application/json' } });
        }
    }

    // 미승인 사용자가 서비스 API 호출 시 pending 페이지로
    if (resp.status === 403) {
        try {
            const clone = resp.clone();
            const data = await clone.json();
            if (data.detail && data.detail.includes('승인 대기')) {
                window.location.href = '/static/pending.html';
                return new Response('{}', { status: 403, headers: { 'Content-Type': 'application/json' } });
            }
        } catch { /* not JSON or other 403, pass through */ }
    }

    return resp;
}

/**
 * 페이지 로드 시 인증 확인
 */
async function checkAuth() {
    try {
        const resp = await fetch('/api/auth/me', { credentials: 'same-origin' });
        if (resp.ok) {
            const data = await resp.json();
            currentUser = data.user;

            if (!currentUser.approved) {
                window.location.href = '/static/pending.html';
                return false;
            }

            _authReady(currentUser);
            updateUserUI(data.user);
            return true;
        }

        // 토큰 갱신 시도
        const refreshResp = await fetch('/api/auth/refresh', {
            method: 'POST',
            credentials: 'same-origin',
        });
        if (refreshResp.ok) {
            const data = await refreshResp.json();
            currentUser = data.user;

            if (!currentUser.approved) {
                window.location.href = '/static/pending.html';
                return false;
            }

            _authReady(currentUser);
            updateUserUI(data.user);
            return true;
        }

        window.location.href = '/static/login.html';
        return false;
    } catch {
        window.location.href = '/static/login.html';
        return false;
    }
}

/**
 * 뒤로가기 — 같은 origin에서 진입했으면 history.back, 외부/북마크/새 탭이면 href="/" 기본 동작.
 */
function goBackOrHome(event) {
    try {
        if (document.referrer) {
            const ref = new URL(document.referrer);
            if (ref.origin === window.location.origin) {
                event.preventDefault();
                history.back();
            }
        }
    } catch (e) {
    }
}

/**
 * 네비바 우측 메뉴 렌더링
 */
function updateUserUI(user) {
    const nav = document.getElementById('navbar-right');
    if (!nav || !user) return;
    nav.innerHTML = '';

    // Home
    const homeLink = document.createElement('a');
    homeLink.href = '/';
    homeLink.className = 'navbar-link';
    homeLink.innerHTML = '<i data-lucide="home"></i><span>Home</span>';
    nav.appendChild(homeLink);

    // 새소식
    const newsLink = document.createElement('a');
    newsLink.href = '/static/changelog.html';
    newsLink.className = 'navbar-link';
    newsLink.innerHTML = '<i data-lucide="bell"></i><span>새소식</span>';
    nav.appendChild(newsLink);

    // 작업이력
    const historyLink = document.createElement('a');
    historyLink.href = '/static/history.html';
    historyLink.className = 'navbar-link';
    historyLink.innerHTML = '<i data-lucide="clock"></i><span>작업이력</span>';
    nav.appendChild(historyLink);

    // API 키 입력 (미설정 시)
    const ctaBtn = document.createElement('a');
    ctaBtn.href = '/static/settings.html';
    ctaBtn.className = 'navbar-cta';
    ctaBtn.innerHTML = '<i data-lucide="key-round"></i><span>API 키 입력</span>';
    nav.appendChild(ctaBtn);

    // 다크/라이트 모드 전환 (클릭은 document 위임으로 처리)
    const themeBtn = document.createElement('button');
    themeBtn.className = 'navbar-icon-btn';
    themeBtn.id = 'theme-toggle-btn';
    themeBtn.type = 'button';
    themeBtn.innerHTML = document.body.classList.contains('dark') ? ICON_MOON : ICON_SUN;
    themeBtn.title = '테마 전환';
    nav.appendChild(themeBtn);

    // 이메일
    const emailSpan = document.createElement('span');
    emailSpan.className = 'navbar-email';
    emailSpan.textContent = user.email;
    nav.appendChild(emailSpan);

    // 관리 (admin만)
    if (user.role === 'admin') {
        const adminLink = document.createElement('a');
        adminLink.href = '/static/admin.html';
        adminLink.className = 'navbar-link navbar-link-admin';
        adminLink.innerHTML = '<i data-lucide="shield-check"></i><span>관리</span>';
        nav.appendChild(adminLink);
    }

    // 로그아웃
    const logoutBtn = document.createElement('button');
    logoutBtn.className = 'navbar-icon-btn';
    logoutBtn.title = '로그아웃';
    logoutBtn.innerHTML = '<i data-lucide="log-out"></i>';
    logoutBtn.onclick = logout;
    nav.appendChild(logoutBtn);

    if (window.refreshIcons) window.refreshIcons();
}

/**
 * 로그아웃
 */
async function logout() {
    await fetch('/api/auth/logout', { method: 'POST', credentials: 'same-origin' });
    window.location.href = '/static/login.html';
}

// 로그인/리셋/승인대기 페이지가 아니면 인증 확인
document.addEventListener('DOMContentLoaded', () => {
    const path = window.location.pathname;
    if (!path.includes('login') && !path.includes('reset-password') && !path.includes('pending') && !path.includes('terms') && !path.includes('privacy')) {
        checkAuth();
    }
});
