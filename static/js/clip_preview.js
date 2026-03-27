/**
 * AI 클립 미리보기 페이지 - 클립 확인 & 재생성 & 영상 제작 확정
 */

const params = new URLSearchParams(window.location.search);
const jobId = params.get('job');
let clipPreviewData = null;

async function loadClipPreview() {
    if (!jobId) {
        alert('작업 ID가 없습니다');
        return;
    }

    try {
        const resp = await fetch(`/api/jobs/${jobId}/clip-preview`);
        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || '클립 미리보기 로드 실패');
        }

        clipPreviewData = await resp.json();
        document.getElementById('clip-preview-title').textContent = clipPreviewData.title;

        const grid = document.getElementById('clip-preview-grid');
        grid.innerHTML = clipPreviewData.lines.map((line, i) => `
            <div class="preview-card" id="clip-card-${i}">
                <div class="preview-image-wrap" style="position: relative;">
                    <video src="${clipPreviewData.clip_urls[i]}?t=${Date.now()}"
                           class="preview-image" autoplay loop muted playsinline></video>
                </div>
                <div class="preview-info">
                    <span class="line-num">${i + 1}</span>
                    <p class="preview-text">${escapeHtml(line.text)}</p>
                    <span class="line-motion">${line.motion}</span>
                    <button class="btn-small btn-secondary" onclick="regenerateClip(${i})">재생성</button>
                    <button class="btn-small btn-upload" onclick="triggerClipUpload(${i})">영상 업로드</button>
                    <input type="file" id="clip-upload-${i}" accept="video/mp4,video/quicktime,video/webm,video/x-msvideo"
                           style="display:none" onchange="handleClipUpload(${i}, this)">
                </div>
            </div>
        `).join('');
    } catch (e) {
        alert('에러: ' + e.message);
    }
}

async function regenerateClip(index) {
    const card = document.getElementById(`clip-card-${index}`);
    const video = card.querySelector('video');
    const imgWrap = card.querySelector('.preview-image-wrap');

    // 로딩 오버레이 표시
    video.style.opacity = '0.2';
    let overlay = card.querySelector('.regen-overlay');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.className = 'regen-overlay';
        overlay.innerHTML = '<div class="regen-spinner"></div><p class="regen-text">클립 생성 중...</p>';
        imgWrap.appendChild(overlay);
    }
    overlay.style.display = 'flex';

    // 버튼 비활성화
    const btns = card.querySelectorAll('button');
    btns.forEach(btn => btn.disabled = true);

    try {
        const resp = await fetch(`/api/jobs/${jobId}/regenerate-clip/${index}`, {
            method: 'POST',
        });

        if (resp.ok) {
            let elapsed = 0;
            const interval = setInterval(async () => {
                elapsed += 2000;
                const sec = Math.floor(elapsed / 1000);
                const textEl = overlay.querySelector('.regen-text');
                if (textEl) textEl.textContent = `클립 생성 중... ${sec}초`;

                // 5분 타임아웃
                if (elapsed > 300000) {
                    clearInterval(interval);
                    overlay.style.display = 'none';
                    btns.forEach(btn => btn.disabled = false);
                    loadClipPreview();
                    return;
                }

                try {
                    const statusResp = await fetch(`/api/jobs/${jobId}`);
                    const job = await statusResp.json();

                    if (job.status === 'failed') {
                        clearInterval(interval);
                        overlay.style.display = 'none';
                        btns.forEach(btn => btn.disabled = false);
                        alert(job.error || '클립 재생성 실패');
                        loadClipPreview();
                        return;
                    }

                    // 클립 파일이 갱신되었는지 확인
                    if (job.status === 'clips_ready' && elapsed >= 10000) {
                        clearInterval(interval);
                        video.src = `${clipPreviewData.clip_urls[index]}?t=${Date.now()}`;
                        video.style.opacity = '1';
                        overlay.style.display = 'none';
                        btns.forEach(btn => btn.disabled = false);
                    }
                } catch (e) { /* retry */ }
            }, 2000);
        }
    } catch (e) {
        alert('재생성 실패: ' + e.message);
        video.style.opacity = '1';
        overlay.style.display = 'none';
        btns.forEach(btn => btn.disabled = false);
    }
}

// ── 영상 업로드 ──

function triggerClipUpload(index) {
    document.getElementById(`clip-upload-${index}`).click();
}

async function handleClipUpload(index, input) {
    const file = input.files[0];
    if (!file) return;

    if (file.size > 50 * 1024 * 1024) {
        alert('파일 크기는 50MB 이하만 가능합니다');
        input.value = '';
        return;
    }

    const card = document.getElementById(`clip-card-${index}`);
    const video = card.querySelector('video');
    const imgWrap = card.querySelector('.preview-image-wrap');

    // 로딩 오버레이
    video.style.opacity = '0.2';
    let overlay = card.querySelector('.regen-overlay');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.className = 'regen-overlay';
        overlay.innerHTML = '<div class="regen-spinner"></div><p class="regen-text">업로드 중...</p>';
        imgWrap.appendChild(overlay);
    }
    overlay.querySelector('.regen-text').textContent = '업로드 중...';
    overlay.style.display = 'flex';

    const btns = card.querySelectorAll('button');
    btns.forEach(btn => btn.disabled = true);

    try {
        const formData = new FormData();
        formData.append('file', file);

        const resp = await fetch(`/api/jobs/${jobId}/upload-clip/${index}`, {
            method: 'POST',
            body: formData,
        });

        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || '업로드 실패');
        }

        video.src = `${clipPreviewData.clip_urls[index]}?t=${Date.now()}`;
        video.style.opacity = '1';
        overlay.style.display = 'none';
    } catch (e) {
        alert('업로드 실패: ' + e.message);
        video.style.opacity = '1';
        overlay.style.display = 'none';
    } finally {
        btns.forEach(btn => btn.disabled = false);
        input.value = '';
    }
}

async function confirmClipsAndRender() {
    const btn = document.getElementById('btn-confirm-clips');
    btn.disabled = true;
    btn.textContent = '영상 제작 시작 중...';

    try {
        const resp = await fetch(`/api/jobs/${jobId}/confirm-clips`, {
            method: 'POST',
        });
        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || '확정 실패');
        }

        window.location.href = `/static/status.html?job=${jobId}&phase=render`;
    } catch (e) {
        alert('에러: ' + e.message);
        btn.disabled = false;
        btn.textContent = '영상 제작 시작';
    }
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

loadClipPreview();
