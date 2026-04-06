/* 인증 유틸리티 - 모든 페이지에서 공유 */

/* ── 테마 (다크/라이트) ── */
function toggleTheme() {
    const isDark = document.body.classList.toggle('dark');
    localStorage.setItem('theme', isDark ? 'dark' : 'light');
    const btn = document.getElementById('theme-toggle-btn');
    if (btn) btn.innerHTML = isDark ? '&#127769;' : '&#9728;&#65039;';
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
    homeLink.textContent = 'Home';
    nav.appendChild(homeLink);

    // 새소식
    const newsLink = document.createElement('a');
    newsLink.href = '/static/changelog.html';
    newsLink.className = 'navbar-link';
    newsLink.textContent = '새소식';
    nav.appendChild(newsLink);

    // 작업이력
    const historyLink = document.createElement('a');
    historyLink.href = '/static/history.html';
    historyLink.className = 'navbar-link';
    historyLink.textContent = '작업이력';
    nav.appendChild(historyLink);

    // API 키 입력 (미설정 시)
    if (!user.has_gemini_key) {
        const ctaBtn = document.createElement('a');
        ctaBtn.href = '/static/settings.html';
        ctaBtn.className = 'navbar-cta';
        ctaBtn.textContent = 'API 키를 입력해주세요';
        nav.appendChild(ctaBtn);
    }

    // 다크/라이트 모드 전환
    const themeBtn = document.createElement('button');
    themeBtn.className = 'navbar-icon-btn';
    themeBtn.id = 'theme-toggle-btn';
    themeBtn.innerHTML = document.body.classList.contains('dark') ? '&#127769;' : '&#9728;&#65039;';
    themeBtn.title = '테마 전환';
    themeBtn.onclick = toggleTheme;
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
        adminLink.className = 'navbar-link';
        adminLink.style.color = 'var(--primary)';
        adminLink.textContent = '관리';
        nav.appendChild(adminLink);
    }

    // 로그아웃
    const logoutBtn = document.createElement('button');
    logoutBtn.className = 'navbar-icon-btn';
    logoutBtn.title = '로그아웃';
    logoutBtn.innerHTML = '&#10132;';
    logoutBtn.onclick = logout;
    nav.appendChild(logoutBtn);
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
    if (!path.includes('login') && !path.includes('reset-password') && !path.includes('pending')) {
        checkAuth();
    }
});
