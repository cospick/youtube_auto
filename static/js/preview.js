/**
 * 미리보기 페이지 - 이미지 확인 & 프롬프트 수정/재생성 & 이미지 업로드 & 영상 제작 확정
 */

const params = new URLSearchParams(window.location.search);
const jobId = params.get('job');
let previewData = null;

async function loadPreview() {
    if (!jobId) {
        alert('작업 ID가 없습니다');
        return;
    }

    try {
        const resp = await fetch(`/api/jobs/${jobId}/preview`);
        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || '미리보기 로드 실패');
        }

        previewData = await resp.json();
        document.getElementById('preview-title').textContent = previewData.title;

        const grid = document.getElementById('preview-grid');
        grid.innerHTML = previewData.lines.map((line, i) => `
            <div class="preview-card" id="card-${i}">
                <div class="preview-image-wrap">
                    <img src="${previewData.image_urls[i]}?t=${Date.now()}" alt="이미지 ${i + 1}" class="preview-image">
                </div>
                <div class="preview-info">
                    <span class="line-num">${i + 1}</span>
                    <p class="preview-text">${escapeHtml(line.text)}</p>
                    <span class="line-motion">${line.motion}</span>
                    <button class="btn-small btn-secondary" onclick="toggleRequestInput(${i})">재생성</button>
                    <button class="btn-small btn-upload" onclick="triggerUpload(${i})">이미지 업로드</button>
                    <input type="file" id="upload-input-${i}" accept="image/png,image/jpeg,image/webp"
                           style="display:none" onchange="handleUpload(${i}, this)">
                </div>
                <div class="prompt-edit hidden" id="prompt-edit-${i}">
                    <div class="prompt-current">
                        <label class="prompt-label">현재 프롬프트 (영어)</label>
                        <textarea class="prompt-textarea" id="eng-prompt-${i}" rows="3">${escapeHtml(line.image_prompt)}</textarea>
                        <button class="btn-small btn-primary" onclick="regenerateWithEnglish(${i})">이 프롬프트로 재생성</button>
                    </div>
                    <div class="prompt-divider"><span>또는</span></div>
                    <div class="prompt-korean">
                        <label class="prompt-label">한글로 새 요청</label>
                        <input type="text" class="korean-request" id="korean-req-${i}"
                               placeholder="예: 한국 여성이 거울 앞에서 화장하는 모습">
                        <button class="btn-small btn-primary" onclick="regenerateWithRequest(${i})">한글 요청으로 재생성</button>
                    </div>
                </div>
            </div>
        `).join('');
    } catch (e) {
        alert('에러: ' + e.message);
    }
}

function toggleRequestInput(index) {
    const editDiv = document.getElementById(`prompt-edit-${index}`);
    editDiv.classList.toggle('hidden');
    if (!editDiv.classList.contains('hidden')) {
        document.getElementById(`eng-prompt-${index}`).focus();
    }
}

// ── 프롬프트 수정 재생성 ──

async function regenerateWithEnglish(index) {
    const englishPrompt = document.getElementById(`eng-prompt-${index}`).value.trim();
    if (!englishPrompt) {
        alert('프롬프트를 입력해주세요');
        return;
    }
    await doRegenerate(index, null, englishPrompt);
}

async function regenerateWithRequest(index) {
    const koreanRequest = document.getElementById(`korean-req-${index}`).value.trim();
    if (!koreanRequest) {
        alert('원하는 이미지를 한글로 설명해주세요');
        return;
    }
    await doRegenerate(index, koreanRequest, null);
}

async function doRegenerate(index, koreanRequest, englishPrompt = null) {
    const card = document.getElementById(`card-${index}`);
    const imgWrap = card.querySelector('.preview-image-wrap');
    const img = card.querySelector('.preview-image');

    // 로딩 오버레이 표시
    img.style.opacity = '0.2';
    let overlay = card.querySelector('.regen-overlay');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.className = 'regen-overlay';
        overlay.innerHTML = '<div class="regen-spinner"></div><p class="regen-text">이미지 생성 중...</p>';
        imgWrap.style.position = 'relative';
        imgWrap.appendChild(overlay);
    }
    overlay.querySelector('.regen-text').textContent = '이미지 생성 중...';
    overlay.style.display = 'flex';

    // 버튼 비활성화
    const regenBtns = card.querySelectorAll('button');
    regenBtns.forEach(btn => btn.disabled = true);

    try {
        const bodyData = {};
        if (englishPrompt) {
            bodyData.english_prompt = englishPrompt;
        } else {
            bodyData.korean_request = koreanRequest;
        }

        const resp = await fetch(`/api/jobs/${jobId}/regenerate-image/${index}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(bodyData),
        });

        if (resp.ok) {
            let elapsed = 0;
            const interval = setInterval(async () => {
                elapsed += 1000;
                const sec = Math.floor(elapsed / 1000);
                const textEl = overlay.querySelector('.regen-text');
                if (textEl) textEl.textContent = `이미지 생성 중... ${sec}초`;

                if (elapsed > 60000) {
                    clearInterval(interval);
                    overlay.style.display = 'none';
                    regenBtns.forEach(btn => btn.disabled = false);
                    loadPreview();
                    return;
                }
                try {
                    const statusResp = await fetch(`/api/jobs/${jobId}`);
                    const job = await statusResp.json();
                    if (job.status === 'failed') {
                        clearInterval(interval);
                        overlay.style.display = 'none';
                        regenBtns.forEach(btn => btn.disabled = false);
                        alert(job.error || '이미지 재생성 실패');
                        loadPreview();
                        return;
                    }
                    if (job.status === 'preview_ready' && elapsed >= 3000) {
                        clearInterval(interval);
                        // 이미지 갱신
                        img.src = `${previewData.image_urls[index]}?t=${Date.now()}`;
                        img.style.opacity = '1';
                        overlay.style.display = 'none';
                        regenBtns.forEach(btn => btn.disabled = false);
                        // 프롬프트 데이터 갱신
                        const freshResp = await fetch(`/api/jobs/${jobId}/preview`);
                        if (freshResp.ok) {
                            previewData = await freshResp.json();
                            const textarea = document.getElementById(`eng-prompt-${index}`);
                            if (textarea) textarea.value = previewData.lines[index].image_prompt;
                        }
                    }
                } catch (e) { /* retry */ }
            }, 1000);
        }
    } catch (e) {
        alert('재생성 실패: ' + e.message);
        img.style.opacity = '1';
        overlay.style.display = 'none';
        regenBtns.forEach(btn => btn.disabled = false);
    }
}

// ── 이미지 업로드 ──

function triggerUpload(index) {
    document.getElementById(`upload-input-${index}`).click();
}

async function handleUpload(index, input) {
    const file = input.files[0];
    if (!file) return;

    if (file.size > 10 * 1024 * 1024) {
        alert('파일 크기는 10MB 이하만 가능합니다');
        input.value = '';
        return;
    }

    const card = document.getElementById(`card-${index}`);
    const imgWrap = card.querySelector('.preview-image-wrap');
    const img = card.querySelector('.preview-image');

    // 로딩 오버레이
    img.style.opacity = '0.2';
    let overlay = card.querySelector('.regen-overlay');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.className = 'regen-overlay';
        overlay.innerHTML = '<div class="regen-spinner"></div><p class="regen-text">업로드 중...</p>';
        imgWrap.style.position = 'relative';
        imgWrap.appendChild(overlay);
    }
    overlay.querySelector('.regen-text').textContent = '업로드 중...';
    overlay.style.display = 'flex';

    const regenBtns = card.querySelectorAll('button');
    regenBtns.forEach(btn => btn.disabled = true);

    try {
        const formData = new FormData();
        formData.append('file', file);

        const resp = await fetch(`/api/jobs/${jobId}/upload-image/${index}`, {
            method: 'POST',
            body: formData,
        });

        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || '업로드 실패');
        }

        img.src = `${previewData.image_urls[index]}?t=${Date.now()}`;
        img.style.opacity = '1';
        overlay.style.display = 'none';
    } catch (e) {
        alert('업로드 실패: ' + e.message);
        img.style.opacity = '1';
        overlay.style.display = 'none';
    } finally {
        regenBtns.forEach(btn => btn.disabled = false);
        input.value = '';
    }
}

// ── 영상 모드 선택 & 확정 ──

async function confirmAndRender() {
    const btn = document.getElementById('btn-confirm');
    btn.disabled = true;
    btn.textContent = '처리 중...';

    try {
        const videoMode = getSelectedVideoMode();
        const resp = await fetch(`/api/jobs/${jobId}/confirm`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ video_mode: videoMode }),
        });
        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || '확정 실패');
        }

        const result = await resp.json();

        if (result.next === 'clips') {
            window.location.href = `/static/status.html?job=${jobId}&phase=clips`;
        } else {
            window.location.href = `/static/status.html?job=${jobId}&phase=render`;
        }
    } catch (e) {
        alert('에러: ' + e.message);
        btn.disabled = false;
        btn.textContent = '영상 제작 시작';
    }
}

function selectVideoMode(card) {
    const grid = card.closest('.style-card-grid');
    grid.querySelectorAll('.style-card').forEach(c => c.classList.remove('selected'));
    card.classList.add('selected');
}

function getSelectedVideoMode() {
    const selected = document.querySelector('.style-card-grid .style-card.selected');
    return selected ? selected.dataset.value : 'kenburns';
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

loadPreview();
