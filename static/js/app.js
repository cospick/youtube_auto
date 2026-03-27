/**
 * 메인 페이지 - 멀티스텝 대본 생성
 * Step 1: 주제 설정 → Step 2: 제목 선택 → Step 3: 나레이션 확인
 * → Step 4: 이미지 스타일 → Step 5: 음성 설정 → Step 6: BGM 설정
 * → Step 7: 최종 확인 → Job 생성
 */

// ── TTS 음성 옵션 (엔진별) ──
const VOICE_OPTIONS = {
    edge: [
        { value: "ko-KR-HyunsuMultilingualNeural", label: "현수 (남성, 다국어)" },
        { value: "ko-KR-InJoonNeural", label: "인준 (남성)" },
        { value: "ko-KR-SunHiNeural", label: "선희 (여성)" },
    ],
    typecast: [
        { value: "tc_62e8f21e979b3860fe2f6a24", label: "혜리 (여성)" },
        { value: "tc_611c3f692fac944dff493a04", label: "세희 (여성)" },
        { value: "tc_6568164fe05ddffee8b0e271", label: "시연 (여성)" },
        { value: "tc_622964d6255364be41659078", label: "세나 (여성)" },
        { value: "tc_61659c5818732016a95fe763", label: "류은 (여성)" },
        { value: "tc_632293f759d649937b97f323", label: "진우 (남성)" },
        { value: "tc_668f4f533ea5c6ce5e43fd48", label: "우성 (남성)" },
        { value: "tc_6059dad0b83880769a50502f", label: "창수 (남성)" },
        { value: "tc_61de29497924994f5abd68db", label: "세진 (남성)" },
    ],
};

// ── 스텝 구성 ──
const STEPS = [
    { id: 'step-input',     label: '주제',     summaryFn: () => document.getElementById('topic').value || '' },
    { id: 'step-titles',    label: '제목',     summaryFn: () => selectedTitle || '' },
    { id: 'step-narration', label: '나레이션', summaryFn: () => selectedTitle || '' },
    { id: 'step-style',     label: '스타일',   summaryFn: () => {
        const card = document.querySelector('.style-card.selected .style-card-name');
        return card ? card.textContent : '';
    }},
    { id: 'step-tts',       label: '음성',     summaryFn: () => {
        const sel = document.getElementById('tts-voice');
        return sel && sel.selectedOptions[0] ? sel.selectedOptions[0].text : '';
    }},
    { id: 'step-bgm',       label: 'BGM',      summaryFn: () => selectedBgm ? selectedBgm.replace(/\.(mp3|wav|ogg)$/i, '') : '없음' },
    { id: 'step-confirm',   label: '확인',     summaryFn: () => '' },
];
let currentStepIndex = 0;

// ── 상태 관리 ──
let titleOptions = null;
let selectedTitle = null;
let narrationData = null;
let scriptData = null;
let bgmList = [];
let selectedBgm = null;
let bgmAudio = null;
let bgmDuration = 0;
let previewAudio = null;

// ── 카테고리 필드 토글 ──
function toggleCategoryFields() {
    const category = document.getElementById('category').value;
    const cosmeticsFields = document.getElementById('cosmetics-fields');
    cosmeticsFields.style.display = category === 'cosmetics' ? 'block' : 'none';
}

function onTitleEdited(value) {
    selectedTitle = value;
}

function toggleProductName() {
    const mentionType = document.getElementById('mention-type').value;
    const field = document.getElementById('product-name-field');
    field.style.display = mentionType === 'direct' ? 'block' : 'none';
    if (mentionType !== 'direct') {
        document.getElementById('product-name').value = '';
    }
}

function getCategoryPayload() {
    const category = document.getElementById('category').value;
    const payload = { category };
    if (category === 'cosmetics') {
        const painPoint = document.getElementById('pain-point').value.trim();
        const ingredient = document.getElementById('ingredient').value.trim();
        const mentionType = document.getElementById('mention-type').value;
        if (painPoint) payload.pain_point = painPoint;
        if (ingredient) payload.ingredient = ingredient;
        payload.mention_type = mentionType;
        if (mentionType === 'direct') {
            const productName = document.getElementById('product-name').value.trim();
            if (productName) payload.product_name = productName;
        }
    }
    return payload;
}

// ──────────────────────────────────
// Step 2: 제목 생성
// ──────────────────────────────────
async function generateTitles() {
    const topic = document.getElementById('topic').value.trim();
    if (!topic) {
        alert('주제를 입력해주세요');
        return;
    }

    showLoading('제목 생성 중...');

    try {
        const payload = { topic, ...getCategoryPayload() };
        const resp = await fetch('/api/generate/titles', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });

        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || '제목 생성 실패');
        }

        titleOptions = await resp.json();
        displayTitles(titleOptions);
    } catch (e) {
        alert('에러: ' + e.message);
        hideLoading();
    }
}

function displayTitles(data) {
    hideLoading();
    goToStep(1);

    const container = document.getElementById('title-options');
    container.innerHTML = data.titles.map((opt, i) => `
        <div class="title-option" onclick="selectTitle(${i})">
            <div class="title-text">${escapeHtml(opt.title)}</div>
            <div class="title-hook">${escapeHtml(opt.hook)}</div>
        </div>
    `).join('');
}

function selectTitle(index) {
    selectedTitle = titleOptions.titles[index].title;

    document.querySelectorAll('.title-option').forEach((el, i) => {
        el.classList.toggle('selected', i === index);
    });

    generateNarration();
}

// ──────────────────────────────────
// Step 3: 나레이션 생성
// ──────────────────────────────────
async function generateNarration() {
    if (!selectedTitle) return;

    showLoading('나레이션 생성 중...');

    try {
        const topic = document.getElementById('topic').value.trim();
        const payload = {
            topic,
            selected_title: selectedTitle,
            num_lines: 6,
            ...getCategoryPayload(),
        };

        const resp = await fetch('/api/generate/narration', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });

        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || '나레이션 생성 실패');
        }

        narrationData = await resp.json();
        displayNarration(narrationData);
    } catch (e) {
        alert('에러: ' + e.message);
        hideLoading();
    }
}

function displayNarration(data) {
    hideLoading();
    goToStep(2);

    document.getElementById('selected-title-display').value = selectedTitle;

    const roleLabels = {
        hook: 'Hook', problem: '문제', insight: '핵심',
        solution1: '해결 1', solution2: '해결 2', cta: 'CTA',
    };

    const container = document.getElementById('narration-lines');
    container.innerHTML = data.lines.map((line, i) => {
        const charCount = line.text.replace(/[?,!.~…]/g, '').length;
        const overClass = charCount > 24 ? 'over' : '';
        return `
        <div class="narration-line">
            <div class="line-header">
                <span class="line-num">${i + 1}</span>
                <span class="narration-role">${roleLabels[line.role] || line.role}</span>
                <span class="char-count ${overClass}">${charCount}/24</span>
            </div>
            <input type="text" class="line-text" value="${escapeHtml(line.text)}"
                   data-index="${i}" oninput="updateCharCount(this)">
        </div>
    `}).join('');
}

function updateCharCount(input) {
    const count = input.value.replace(/[?,!.~…]/g, '').length;
    const counter = input.parentElement.querySelector('.char-count');
    counter.textContent = `${count}/24`;
    counter.classList.toggle('over', count > 24);
}

function regenerateNarration() {
    generateNarration();
}

// ──────────────────────────────────
// Step 4: 나레이션 확정 → 이미지 스타일 표시
// ──────────────────────────────────
function approveNarration() {
    // 편집된 나레이션 텍스트 수집 & 검증
    const textInputs = document.querySelectorAll('#narration-lines .line-text');
    const narrationLines = Array.from(textInputs).map(input => input.value.trim());

    if (narrationLines.some(line => !line)) {
        alert('빈 나레이션 줄이 있습니다');
        return;
    }

    // 나레이션 텍스트 저장 (이미지 프롬프트 생성 시 사용)
    window._approvedNarrationLines = narrationLines;

    goToStep(3);
    document.getElementById('image-prompt-result').classList.add('hidden');
}

async function generateImagePrompts() {
    const narrationLines = window._approvedNarrationLines;
    if (!narrationLines) {
        alert('나레이션을 먼저 확정해주세요');
        return;
    }

    showLoading('이미지 프롬프트 생성 중...');

    try {
        const style = document.getElementById('style').value;
        const payload = { narration_lines: narrationLines, style };

        const resp = await fetch('/api/generate/image-prompts', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });

        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || '이미지 프롬프트 생성 실패');
        }

        scriptData = await resp.json();
        displayImagePrompts(scriptData);
    } catch (e) {
        alert('에러: ' + e.message);
        hideLoading();
    }
}

function displayImagePrompts(data) {
    hideLoading();
    goToStep(3);

    document.getElementById('title-text').value = selectedTitle;
    document.getElementById('image-prompt-result').classList.remove('hidden');

    const container = document.getElementById('script-lines');
    container.innerHTML = data.lines.map((line, i) => `
        <div class="script-line">
            <div class="line-header">
                <span class="line-num">${i + 1}</span>
                <span class="line-motion">${motionLabel(line.motion)}</span>
            </div>
            <input type="text" class="line-text" value="${escapeHtml(line.text)}" data-index="${i}">
            <p class="line-prompt">${escapeHtml(line.image_prompt)}</p>
        </div>
    `).join('');
}

// ──────────────────────────────────
// Step 5: 음성 설정
// ──────────────────────────────────
function confirmImagePrompts() {
    if (!scriptData) {
        alert('이미지 프롬프트를 먼저 생성해주세요');
        return;
    }

    // 수정된 텍스트 반영
    const textInputs = document.querySelectorAll('#script-lines .line-text');
    textInputs.forEach((input, i) => {
        scriptData.lines[i].text = input.value;
    });

    goToStep(4);
    updateVoiceOptions();
}

// ── TTS 음성 옵션 ──
function updateVoiceOptions() {
    const engine = document.getElementById('tts-engine').value;
    const voiceSelect = document.getElementById('tts-voice');
    const options = VOICE_OPTIONS[engine] || [];
    voiceSelect.innerHTML = options.map(opt =>
        `<option value="${opt.value}">${opt.label}</option>`
    ).join('');

    const emotionSection = document.getElementById('emotion-section');
    if (engine === 'typecast') {
        emotionSection.classList.remove('hidden');
        loadEmotions(voiceSelect.value);
    } else {
        emotionSection.classList.add('hidden');
        document.getElementById('tts-emotion').value = 'normal';
    }
}

async function loadEmotions(voiceId) {
    const container = document.getElementById('emotion-buttons');
    const hiddenInput = document.getElementById('tts-emotion');
    hiddenInput.value = 'normal';

    if (!voiceId) {
        container.innerHTML = '<p class="text-dim">성우를 선택하세요.</p>';
        return;
    }

    container.innerHTML = '<p class="text-dim">감정 목록 로딩 중...</p>';

    try {
        const resp = await fetch(`/api/tts/emotions?voice_id=${encodeURIComponent(voiceId)}`);
        if (!resp.ok) throw new Error('조회 실패');
        const emotions = await resp.json();

        container.innerHTML = emotions.map(e =>
            `<button type="button" class="emotion-btn${e.value === 'normal' ? ' active' : ''}" data-emotion="${e.value}">${e.label}</button>`
        ).join('');

        container.querySelectorAll('.emotion-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                container.querySelectorAll('.emotion-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                hiddenInput.value = btn.dataset.emotion;
            });
        });
    } catch (e) {
        container.innerHTML = '<p class="text-dim">감정 목록을 불러올 수 없습니다.</p>';
    }
}

// ── TTS 이벤트 리스너 ──
document.getElementById('tts-engine').addEventListener('change', updateVoiceOptions);
document.getElementById('tts-voice').addEventListener('change', function() {
    if (document.getElementById('tts-engine').value === 'typecast') {
        loadEmotions(this.value);
    }
});
document.getElementById('tts-speed').addEventListener('input', function() {
    document.getElementById('speed-val').textContent = this.value;
});

// ── 음성 미리듣기 ──
document.getElementById('voice-preview-btn').addEventListener('click', async function() {
    const btn = this;
    const engine = document.getElementById('tts-engine').value;
    const voiceId = document.getElementById('tts-voice').value;

    if (!voiceId) return;

    if (previewAudio && !previewAudio.paused) {
        previewAudio.pause();
        previewAudio = null;
        btn.textContent = '▶';
        return;
    }

    btn.textContent = '⏳';
    btn.disabled = true;

    try {
        const speed = document.getElementById('tts-speed').value;
        const emotion = engine === 'typecast' ? document.getElementById('tts-emotion').value : 'normal';
        const url = `/api/tts/preview?engine=${encodeURIComponent(engine)}&voice_id=${encodeURIComponent(voiceId)}&speed=${speed}&emotion=${encodeURIComponent(emotion)}`;
        const resp = await fetch(url);
        if (!resp.ok) throw new Error(`미리듣기 실패: ${resp.status}`);

        const blob = await resp.blob();
        const audioUrl = URL.createObjectURL(blob);
        previewAudio = new Audio(audioUrl);
        previewAudio.onended = () => { btn.textContent = '▶'; };
        previewAudio.play();
        btn.textContent = '⏹';
    } catch (e) {
        console.error(e);
        alert('미리듣기 생성에 실패했습니다.');
        btn.textContent = '▶';
    } finally {
        btn.disabled = false;
    }
});

// ──────────────────────────────────
// Step 6: BGM 설정
// ──────────────────────────────────
function confirmTtsSettings() {
    goToStep(5);
    if (bgmList.length === 0) loadBgmList();
}

// ── BGM 이벤트 리스너 ──
document.getElementById('bgm-volume').addEventListener('input', function() {
    document.getElementById('bgm-val').textContent = this.value;
});
document.getElementById('bgm-start').addEventListener('input', function() {
    const val = parseFloat(this.value);
    document.getElementById('bgm-start-val').textContent = formatTime(val);
    document.getElementById('bgm-start-sec').value = val.toFixed(1);
});
document.getElementById('bgm-start-sec').addEventListener('change', function() {
    const val = parseFloat(this.value) || 0;
    const clamped = Math.min(Math.max(val, 0), bgmDuration);
    this.value = clamped.toFixed(1);
    document.getElementById('bgm-start').value = clamped;
    document.getElementById('bgm-start-val').textContent = formatTime(clamped);
});

// ──────────────────────────────────
// Step 7: 최종 확인
// ──────────────────────────────────
function confirmBgmSettings() {
    goToStep(6);

    // 설정 요약 표시
    const engine = document.getElementById('tts-engine').value;
    const voiceLabel = document.getElementById('tts-voice').selectedOptions[0]?.text || '';
    const emotion = document.getElementById('tts-emotion').value;
    const speed = document.getElementById('tts-speed').value;
    const styleCard = document.querySelector('.style-card.selected .style-card-name');
    const style = styleCard ? styleCard.textContent : '';
    const bgm = selectedBgm ? selectedBgm.replace(/\.(mp3|wav|ogg)$/i, '') : '없음';
    const bgmVol = document.getElementById('bgm-volume').value;

    document.getElementById('confirm-summary').innerHTML = `
        <div class="summary-grid">
            <div class="summary-item"><span class="summary-label">제목</span><span>${escapeHtml(document.getElementById('title-text').value)}</span></div>
            <div class="summary-item"><span class="summary-label">이미지 스타일</span><span>${style}</span></div>
            <div class="summary-item"><span class="summary-label">TTS 엔진</span><span>${engine === 'edge' ? 'Edge TTS' : 'Typecast'}</span></div>
            <div class="summary-item"><span class="summary-label">음성</span><span>${voiceLabel}</span></div>
            <div class="summary-item"><span class="summary-label">감정/톤</span><span>${emotion}</span></div>
            <div class="summary-item"><span class="summary-label">속도</span><span>${speed}배</span></div>
            <div class="summary-item"><span class="summary-label">BGM</span><span>${escapeHtml(bgm)}</span></div>
            <div class="summary-item"><span class="summary-label">BGM 볼륨</span><span>${bgmVol}%</span></div>
        </div>
    `;
}

// ──────────────────────────────────
// Job 생성
// ──────────────────────────────────
async function createJob() {
    if (!scriptData) return;

    const payload = {
        topic: document.getElementById('topic').value,
        style: document.getElementById('style').value,
        video_mode: "kenburns",
        tts_engine: document.getElementById('tts-engine').value,
        tts_speed: parseFloat(document.getElementById('tts-speed').value),
        voice_id: document.getElementById('tts-voice').value,
        emotion: document.getElementById('tts-engine').value === 'typecast' ? document.getElementById('tts-emotion').value : null,
        title: document.getElementById('title-text').value,
        lines: scriptData.lines,
        bgm_volume: parseInt(document.getElementById('bgm-volume').value) / 100,
        bgm_filename: selectedBgm || null,
        bgm_start_sec: parseFloat(document.getElementById('bgm-start-sec').value) || 0,
    };

    showLoading('작업 등록 중...');

    try {
        const resp = await fetch('/api/jobs/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });

        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || '작업 생성 실패');
        }

        const job = await resp.json();
        window.location.href = `/static/status.html?job=${job.job_id}&phase=images`;
    } catch (e) {
        alert('에러: ' + e.message);
        hideLoading();
    }
}

// ──────────────────────────────────
// 스텝 관리 (타임라인 + 접기/펼치기)
// ──────────────────────────────────
function goToStep(stepIndex) {
    currentStepIndex = stepIndex;
    updateTimeline(stepIndex);

    STEPS.forEach((step, i) => {
        const section = document.getElementById(step.id);
        if (i < stepIndex) {
            section.classList.remove('hidden');
            section.classList.add('collapsed');
            const summaryEl = document.getElementById('summary-' + i);
            if (summaryEl) summaryEl.textContent = step.summaryFn();
        } else if (i === stepIndex) {
            section.classList.remove('hidden', 'collapsed');
        } else {
            section.classList.add('hidden');
            section.classList.remove('collapsed');
        }
    });

    setTimeout(() => {
        const current = document.getElementById(STEPS[stepIndex].id);
        current.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
}

function updateTimeline(activeIndex) {
    const items = document.querySelectorAll('.timeline-item');
    items.forEach((item, i) => {
        item.classList.remove('completed', 'active');
        if (i < activeIndex) item.classList.add('completed');
        else if (i === activeIndex) item.classList.add('active');
    });

    const track = document.querySelector('.timeline-track');
    const progress = activeIndex === 0 ? 0 : (activeIndex / (STEPS.length - 1)) * 100;
    track.style.setProperty('--timeline-progress', progress + '%');
}

function toggleStepExpand(stepIndex) {
    if (stepIndex >= currentStepIndex) return;
    const section = document.getElementById(STEPS[stepIndex].id);

    if (section.classList.contains('collapsed')) {
        section.classList.remove('collapsed');
    } else {
        section.classList.add('collapsed');
        const summaryEl = document.getElementById('summary-' + stepIndex);
        if (summaryEl) summaryEl.textContent = STEPS[stepIndex].summaryFn();
    }
}

// ──────────────────────────────────
// 스타일 카드 선택
// ──────────────────────────────────
function selectStyle(card) {
    document.querySelectorAll('#step-style .style-card').forEach(c => c.classList.remove('selected'));
    card.classList.add('selected');
    document.getElementById('style').value = card.dataset.value;
}

// ──────────────────────────────────
// 유틸리티
// ──────────────────────────────────
function hideAllSteps() {
    ['step-input', 'step-titles', 'step-narration', 'step-style', 'step-tts', 'step-bgm', 'step-confirm', 'step-loading'].forEach(id => {
        const el = document.getElementById(id);
        el.classList.add('hidden');
        el.classList.remove('collapsed');
    });
}

function showLoading(text) {
    document.getElementById('step-loading').classList.remove('hidden');
    document.getElementById('loading-text').textContent = text;
    document.getElementById('btn-generate').disabled = true;
}

function hideLoading() {
    document.getElementById('step-loading').classList.add('hidden');
    document.getElementById('btn-generate').disabled = false;
}

function motionLabel(motion) {
    const labels = {
        zoom_in: 'Zoom In', zoom_out: 'Zoom Out',
        pan_left: 'Pan Left', pan_right: 'Pan Right',
        pan_up: 'Pan Up', pan_down: 'Pan Down',
    };
    return labels[motion] || motion;
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

// ──────────────────────────────────
// BGM 선택 + 미리듣기
// ──────────────────────────────────
async function loadBgmList() {
    try {
        const resp = await fetch('/api/assets/bgm');
        if (!resp.ok) return;
        bgmList = await resp.json();
        renderBgmList();
    } catch (e) {
        console.error('BGM 목록 로드 실패:', e);
        document.getElementById('bgm-list').innerHTML =
            '<p class="text-dim">BGM 목록을 불러올 수 없습니다</p>';
    }
}

function renderBgmList() {
    const container = document.getElementById('bgm-list');
    if (bgmList.length === 0) {
        container.innerHTML = '<p class="text-dim">bgm/ 폴더에 MP3 파일이 없습니다</p>';
        return;
    }
    container.innerHTML = bgmList.map((bgm, i) => `
        <div class="bgm-card ${selectedBgm === bgm.filename ? 'selected' : ''}"
             onclick="selectBgm(${i})">
            <div class="bgm-name">${escapeHtml(bgm.filename.replace(/\.(mp3|wav|ogg)$/i, ''))}</div>
            <div class="bgm-duration">${formatTime(bgm.duration)}</div>
            <button class="btn-small bgm-play-btn ${bgmAudio && bgmAudio._bgmIdx === i ? 'playing' : ''}"
                    onclick="event.stopPropagation(); toggleBgmPreview(${i})"
                    title="미리듣기">
                ${bgmAudio && bgmAudio._bgmIdx === i ? '■' : '▶'}
            </button>
        </div>
    `).join('');
}

function selectBgm(index) {
    selectedBgm = bgmList[index].filename;
    bgmDuration = bgmList[index].duration;
    renderBgmList();

    const startSection = document.getElementById('bgm-start-section');
    startSection.classList.remove('hidden');
    const slider = document.getElementById('bgm-start');
    slider.max = bgmDuration;
    slider.value = 0;
    document.getElementById('bgm-start-val').textContent = '0:00';
    document.getElementById('bgm-start-sec').value = '0.0';
    document.getElementById('bgm-start-sec').max = bgmDuration;
}

function toggleBgmPreview(index) {
    if (bgmAudio) {
        bgmAudio.pause();
        const wasPlaying = bgmAudio._bgmIdx === index;
        bgmAudio = null;
        renderBgmList();
        if (wasPlaying) return;
    }

    const bgm = bgmList[index];
    bgmAudio = new Audio(bgm.url);
    bgmAudio._bgmIdx = index;

    if (selectedBgm === bgm.filename) {
        bgmAudio.currentTime = parseFloat(document.getElementById('bgm-start-sec').value) || 0;
    }

    bgmAudio.play();
    renderBgmList();

    bgmAudio._timeout = setTimeout(() => {
        if (bgmAudio) {
            bgmAudio.pause();
            bgmAudio = null;
            renderBgmList();
        }
    }, 15000);

    bgmAudio.addEventListener('ended', () => {
        bgmAudio = null;
        renderBgmList();
    });
}

function formatTime(sec) {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
}

// 페이지 로드 시 초기화
updateTimeline(0);
loadBgmList();
