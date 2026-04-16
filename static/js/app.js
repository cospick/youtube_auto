/**
 * 메인 페이지 - 멀티스텝 대본 생성
 * Step 1: 주제 설정 → Step 2: 제목 선택 → Step 3: 나레이션 확인
 * → Step 4: 이미지 스타일 → Step 5: 음성 설정 → Step 6: BGM 설정
 * → Step 7: 최종 확인 → Job 생성

function showFriendlyError(msg) {
    const is503 = msg.includes('503') || msg.includes('UNAVAILABLE');
    const is429 = msg.includes('429') || msg.includes('RESOURCE_EXHAUSTED');
    let userMsg;
    if (is503) {
        userMsg = 'Google AI 서버가 현재 많이 바쁜 상태입니다.\n자동으로 3회 재시도했지만 실패했습니다.\n\n1~2분 후에 다시 시도해주세요.';
    } else if (is429) {
        userMsg = 'API 요청 횟수 제한에 도달했습니다.\n1분 후에 다시 시도해주세요.';
    } else {
        userMsg = '요청 처리에 실패했습니다.\n다시 시도해주세요.\n\n[상세 정보]\n' + msg;
    }
    alert(userMsg);
}
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
let titleLine1 = '';
let titleLine2 = '';
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
    const isCosmetics = category === 'cosmetics';

    // 영상 목적 드롭다운은 화장품일 때만 표시 (카테고리 바로 다음 위치)
    const ctWrapper = document.getElementById('content-type-wrapper');
    if (ctWrapper) ctWrapper.style.display = isCosmetics ? '' : 'none';

    // 화장품 전용 필드 컨테이너 (페인포인트 · 성분 · 제품 그리드)
    const cosmeticsFields = document.getElementById('cosmetics-fields');
    cosmeticsFields.style.display = isCosmetics ? 'block' : 'none';

    const topicHelp = document.getElementById('topic-help');
    if (topicHelp) {
        topicHelp.innerHTML = isCosmetics
            ? '내 화장품이 해결하는 피부 고민을 적어주세요.<br>예: 홍조 피부 진정 방법 / 건조 피부 보습 루틴 / 모공 축소 관리법'
            : '어떤 내용의 영상을 만들지 한 줄로 적어주세요.<br>예: 여름철 자외선 차단 꿀팁 / 초보 운동 루틴 / 다이어트 식단 추천';
    }

    toggleContentType();
}

// ── 영상 목적(정보성/홍보성) 토글 — 화장품 카테고리에서만 의미 있음 ──
function toggleContentType() {
    const category = document.getElementById('category').value;
    if (category !== 'cosmetics') return;
    const isPromo = document.getElementById('content-type').value === 'promo';
    // 홍보성 전용 필드
    ['painpoint-field', 'ingredient-field', 'product-templates'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.style.display = isPromo ? '' : 'none';
    });
    // 정보성 전용 필드
    const infoKw = document.getElementById('info-keyword-field');
    if (infoKw) infoKw.style.display = isPromo ? 'none' : '';
}

// ──────────────────────────────────
// 제품 이미지 템플릿 (CTA 라인용)
// ──────────────────────────────────
let userProducts = [];
window._selectedProductId = null;

async function loadUserProducts() {
    try {
        const resp = await authFetch('/api/products');
        if (!resp.ok) return;
        userProducts = await resp.json();
        renderProductGrid();
    } catch (e) {
        console.error('제품 목록 로드 실패:', e);
    }
}

function renderProductGrid() {
    const grid = document.getElementById('product-grid');
    if (!grid) return;
    const uploadCard = `
        <div class="style-card product-upload-card" onclick="triggerProductUpload()">
            <div class="style-card-preview">
                <span class="style-card-icon">+</span>
            </div>
            <div class="style-card-meta">
                <span class="style-card-name">제품 추가</span>
            </div>
        </div>
    `;
    const productCards = userProducts.map(p => {
        const selected = window._selectedProductId === p.id ? 'selected' : '';
        return `
            <div class="style-card ${selected}" data-product-id="${p.id}" onclick="selectProduct('${p.id}')">
                <button class="product-delete-btn" onclick="deleteProduct(event, '${p.id}')" title="삭제">×</button>
                <div class="style-card-preview">
                    <img class="product-thumb" src="/api/products/${p.id}/image" alt="${escapeHtml(p.name)}">
                </div>
                <div class="style-card-meta">
                    <span class="style-card-name">${escapeHtml(p.name)}</span>
                </div>
            </div>
        `;
    }).join('');
    grid.innerHTML = productCards + uploadCard;
}

function selectProduct(id) {
    window._selectedProductId = id;
    renderProductGrid();
}

function triggerProductUpload() {
    if (userProducts.length >= 20) {
        alert('제품은 최대 20개까지 등록 가능합니다.\n기존 제품을 삭제 후 다시 시도해주세요.');
        return;
    }
    const input = document.getElementById('product-file-input');
    input.value = '';  // 같은 파일 재선택 허용
    input.onchange = handleProductFileSelect;
    input.click();
}

async function handleProductFileSelect(e) {
    const file = e.target.files[0];
    if (!file) return;

    const name = prompt('제품명을 입력하세요 (최대 50자)', file.name.replace(/\.[^.]+$/, ''));
    if (!name || !name.trim()) return;

    const formData = new FormData();
    formData.append('file', file);
    formData.append('name', name.trim());

    try {
        const resp = await authFetch('/api/products', {
            method: 'POST',
            body: formData,
        });
        if (!resp.ok) {
            const err = await resp.json();
            alert(err.detail || '업로드 실패');
            return;
        }
        const newProduct = await resp.json();
        window._selectedProductId = newProduct.id;  // 업로드 직후 자동 선택
        await loadUserProducts();
    } catch (err) {
        alert('업로드 실패: ' + err.message);
    }
}

async function deleteProduct(event, id) {
    event.stopPropagation();
    const product = userProducts.find(p => p.id === id);
    if (!product) return;
    if (!confirm(`"${product.name}" 제품을 삭제하시겠어요?`)) return;

    try {
        const resp = await authFetch(`/api/products/${id}`, { method: 'DELETE' });
        if (!resp.ok) {
            const err = await resp.json();
            alert(err.detail || '삭제 실패');
            return;
        }
        if (window._selectedProductId === id) {
            window._selectedProductId = null;
        }
        await loadUserProducts();
    } catch (err) {
        alert('삭제 실패: ' + err.message);
    }
}

function autoSplitTitle(text) {
    const words = text.split(' ').filter(w => w);
    if (words.length <= 1) return [text, ''];

    let bestSplit = 1, bestDiff = Infinity;
    for (let i = 1; i < words.length; i++) {
        const l1 = words.slice(0, i).join(' ');
        const l2 = words.slice(i).join(' ');
        const diff = Math.abs(l1.length - l2.length);
        if (diff < bestDiff) {
            bestDiff = diff;
            bestSplit = i;
        }
    }
    return [words.slice(0, bestSplit).join(' '), words.slice(bestSplit).join(' ')];
}

function onTitleLineEdited() {
    titleLine1 = document.getElementById('title-line1').value;
    titleLine2 = document.getElementById('title-line2').value;
    selectedTitle = titleLine2 ? titleLine1 + ' ' + titleLine2 : titleLine1;
    updateTitlePreview();
}

function updateTitlePreview() {
    const el1 = document.getElementById('preview-line1');
    const el2 = document.getElementById('preview-line2');
    const frame = document.getElementById('title-preview-frame');

    el1.textContent = titleLine1;
    el2.textContent = titleLine2;

    requestAnimationFrame(() => {
        const frameW = frame.offsetWidth;
        const overflow = el1.scrollWidth > frameW || el2.scrollWidth > frameW;
        frame.classList.toggle('overflow', overflow);
    });
}

function getCategoryPayload() {
    const category = document.getElementById('category').value;
    const payload = { category };
    if (category === 'cosmetics') {
        const contentType = document.getElementById('content-type').value;
        payload.content_type = contentType;
        if (contentType === 'promo') {
            const painPoint = document.getElementById('pain-point').value.trim();
            const ingredient = document.getElementById('ingredient').value.trim();
            if (painPoint) payload.pain_point = painPoint;
            if (ingredient) payload.ingredient = ingredient;
        } else if (contentType === 'info') {
            const keyword = document.getElementById('info-keyword').value.trim();
            if (keyword) payload.keyword = keyword;
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

    advanceToStep(1);
    showLoading('제목 생성 중...');

    try {
        const payload = { topic, ...getCategoryPayload() };
        // 제목 생성 시점의 키워드를 저장해 나레이션 단계에서 불일치 감지
        window._keywordAtTitleGen = payload.keyword || '';
        const resp = await authFetch('/api/generate/titles', {
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
        showFriendlyError(e.message);
        hideLoading();
        goToStep(0);
    }
}

function displayTitles(data) {
    hideLoading();
    hideStepGuide('step-titles');
    document.getElementById('btn-next-title').disabled = true;
    document.getElementById('title-split-editor').classList.add('hidden');

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

    const [line1, line2] = autoSplitTitle(selectedTitle);
    titleLine1 = line1;
    titleLine2 = line2;
    document.getElementById('title-line1').value = line1;
    document.getElementById('title-line2').value = line2;
    updateTitlePreview();
    document.getElementById('title-split-editor').classList.remove('hidden');
    document.getElementById('btn-next-title').disabled = false;
}

function confirmTitle() {
    generateNarration();
}

// ──────────────────────────────────
// Step 3: 나레이션 생성
// ──────────────────────────────────
async function generateNarration() {
    if (!selectedTitle) return;

    // 핵심 키워드가 제목 생성 이후 바뀌었는지 확인 (정보성 전용)
    const currentKeyword = (document.getElementById('info-keyword')?.value || '').trim();
    const titleGenKeyword = window._keywordAtTitleGen || '';
    const _ct = document.getElementById('content-type')?.value;
    const _cat = document.getElementById('category')?.value;
    if (_cat === 'cosmetics' && _ct === 'info' && currentKeyword !== titleGenKeyword) {
        const ok = confirm(
            `⚠️ 핵심 키워드가 바뀌었습니다\n` +
            `(제목 생성 시: "${titleGenKeyword || '(비어있음)'}" → 현재: "${currentKeyword || '(비어있음)'}")\n\n` +
            `제목은 기존 키워드 방향으로 만들어졌는데, 나레이션은 새 키워드 방향으로 생성됩니다.\n` +
            `앞뒤가 맞지 않을 수 있어요.\n\n` +
            `그래도 진행하시겠어요?\n(취소하고 제목부터 다시 생성하는 것을 권장합니다)`
        );
        if (!ok) return;
        // 사용자가 "진행"을 선택했으므로 스냅샷 갱신 — 재생성 시 같은 경고 반복 방지
        window._keywordAtTitleGen = currentKeyword;
    }

    advanceToStep(2);
    showLoading('나레이션 생성 중...');

    try {
        const topic = document.getElementById('topic').value.trim();
        const payload = {
            topic,
            selected_title: selectedTitle,
            num_lines: 6,
            ...getCategoryPayload(),
        };

        const resp = await authFetch('/api/generate/narration', {
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
        showFriendlyError(e.message);
        hideLoading();
    }
}

function displayNarration(data) {
    hideLoading();
    hideStepGuide('step-narration');

    document.getElementById('selected-title-display').textContent = selectedTitle;

    const roleLabels = {
        hook: 'Hook', problem: '문제', insight: '핵심',
        solution1: '해결 1', solution2: '해결 2', cta: 'CTA',
    };

    const container = document.getElementById('narration-lines');
    container.innerHTML = data.lines.map((line, i) => {
        const charCount = line.text.replace(/[?,!.~…]/g, '').length;
        // 22자 이상: 회색 주의, 28자 초과: 빨간 경고
        const overClass = charCount > 28 ? 'over' : '';
        return `
        <div class="narration-line">
            <div class="line-header">
                <span class="line-num">${i + 1}</span>
                <span class="narration-role">${roleLabels[line.role] || line.role}</span>
                <span class="char-count ${overClass}">${charCount}/28</span>
            </div>
            <input type="text" class="line-text" value="${escapeHtml(line.text)}"
                   data-index="${i}" oninput="updateCharCount(this)">
        </div>
    `}).join('');
}

function updateCharCount(input) {
    const count = input.value.replace(/[?,!.~…]/g, '').length;
    const counter = input.parentElement.querySelector('.char-count');
    counter.textContent = `${count}/28`;
    counter.classList.toggle('over', count > 28);
}

function regenerateNarration() {
    generateNarration();
}

// ──────────────────────────────────
// Step 4: 나레이션 확정 → 이미지 스타일 표시
// ──────────────────────────────────
async function approveNarration() {
    // 편집된 나레이션 텍스트 수집 & 검증
    const textInputs = document.querySelectorAll('#narration-lines .line-text');
    const narrationLines = Array.from(textInputs).map(input => input.value.trim());

    if (narrationLines.length === 0) {
        alert('먼저 나레이션을 생성해주세요');
        return;
    }

    if (narrationLines.some(line => !line)) {
        alert('빈 나레이션 줄이 있습니다');
        return;
    }

    window._approvedNarrationLines = narrationLines;

    advanceToStep(3); // 음성 단계로 먼저 이동
    showLoading('이미지 프롬프트 생성 중...');

    try {
        const resp = await authFetch('/api/generate/image-prompts', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                narration_lines: narrationLines,
                style: 'realistic',
                topic: document.getElementById('topic').value.trim(),
                ...getCategoryPayload(),
            }),
        });
        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || '이미지 프롬프트 생성 실패');
        }
        scriptData = await resp.json();
    } catch (e) {
        hideLoading();
        showFriendlyError(e.message);
        goToStep(2); // 실패 시 나레이션으로 복귀
        return;
    }
    hideLoading();
    updateVoiceOptions();
}

// ──────────────────────────────────
// Step 4: 음성 설정
// ──────────────────────────────────

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
        const resp = await authFetch(`/api/tts/emotions?voice_id=${encodeURIComponent(voiceId)}`);
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
        const resp = await authFetch(url);
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
// Step 5: BGM 설정
// ──────────────────────────────────
function confirmTtsSettings() {
    advanceToStep(4);
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
// Step 6: 최종 확인
// ──────────────────────────────────
function confirmBgmSettings() {
    advanceToStep(5);
}

function buildConfirmSummary() {
    const summaryEl = document.getElementById('confirm-summary');
    const createBtn = document.querySelector('#step-confirm .btn-primary');

    // 필수 단계 검사
    const missing = [];
    if (!document.getElementById('topic').value.trim()) missing.push({ idx: 0, name: '주제 입력' });
    if (!selectedTitle) missing.push({ idx: 1, name: '제목 선택' });
    if (!scriptData) missing.push({ idx: 2, name: '나레이션 확정' });

    if (missing.length > 0) {
        const items = missing.map(m =>
            `<li><a href="#" onclick="goToStep(${m.idx}); return false;">${escapeHtml(m.name)}</a></li>`
        ).join('');
        summaryEl.innerHTML = `
            <p class="text-dim" style="text-align:center; padding:20px; font-size:15px; line-height:1.7;">
                아직 영상 생성 준비가 완료되지 않았습니다.<br>아래 단계를 먼저 진행해주세요:
            </p>
            <ul class="missing-steps">${items}</ul>
        `;
        if (createBtn) createBtn.disabled = true;
        return;
    }
    if (createBtn) createBtn.disabled = false;

    const engine = document.getElementById('tts-engine').value;
    const voiceLabel = document.getElementById('tts-voice').selectedOptions[0]?.text || '';
    const emotion = document.getElementById('tts-emotion').value;
    const speed = document.getElementById('tts-speed').value;
    const bgm = selectedBgm ? selectedBgm.replace(/\.(mp3|wav|ogg)$/i, '') : '없음';
    const bgmVol = document.getElementById('bgm-volume').value;

    summaryEl.innerHTML = `
        <div class="summary-grid">
            <div class="summary-item"><span class="summary-label">제목</span><span>${escapeHtml(selectedTitle || '')}</span></div>
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

    // 홍보성 영상만 제품 이미지 필수. 정보성은 product_image_id를 null로 강제해
    // 이전에 선택한 제품이 CTA에 새는 것을 방지한다.
    const category = document.getElementById('category').value;
    const contentType = category === 'cosmetics'
        ? document.getElementById('content-type').value
        : null;
    if (category === 'cosmetics' && contentType === 'promo' && !window._selectedProductId) {
        alert('홍보성 영상은 제품 이미지를 먼저 등록하고 선택해주세요.\n(주제 설정 단계에서 등록)');
        return;
    }
    const productImageId = (contentType === 'info')
        ? null
        : (window._selectedProductId || null);

    const payload = {
        topic: document.getElementById('topic').value,
        style: 'realistic',
        video_mode: "kenburns",
        tts_engine: document.getElementById('tts-engine').value,
        tts_speed: parseFloat(document.getElementById('tts-speed').value),
        voice_id: document.getElementById('tts-voice').value,
        emotion: document.getElementById('tts-engine').value === 'typecast' ? document.getElementById('tts-emotion').value : null,
        title: selectedTitle,
        title_line1: titleLine1,
        title_line2: titleLine2,
        lines: scriptData.lines,
        bgm_volume: parseInt(document.getElementById('bgm-volume').value) / 100,
        bgm_filename: selectedBgm || null,
        bgm_start_sec: parseFloat(document.getElementById('bgm-start-sec').value) || 0,
        product_image_id: productImageId,
    };

    showLoading('작업 등록 중...');

    try {
        const resp = await authFetch('/api/jobs/', {
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
        showFriendlyError(e.message);
        hideLoading();
    }
}

// ──────────────────────────────────
// 스텝 관리 (타임라인 + 접기/펼치기)
// ──────────────────────────────────
let maxReachedStep = 0;

function showStepGuide(stepId, message) {
    const body = document.querySelector(`#${stepId} .step-body`);
    if (!body) return;
    let guide = body.querySelector('.step-guide-msg');
    if (!guide) {
        guide = document.createElement('div');
        guide.className = 'step-guide-msg';
        const h2 = body.querySelector('h2');
        if (h2) h2.after(guide);
        else body.prepend(guide);
    }
    guide.innerHTML = message;
    guide.classList.remove('hidden');
}

function hideStepGuide(stepId) {
    const guide = document.querySelector(`#${stepId} .step-guide-msg`);
    if (guide) guide.classList.add('hidden');
}

function goToStep(stepIndex) {
    currentStepIndex = stepIndex;

    STEPS.forEach((step, i) => {
        const section = document.getElementById(step.id);
        if (i === stepIndex) {
            section.classList.remove('hidden', 'collapsed');
        } else {
            section.classList.add('hidden');
        }
    });

    // 제목 단계: 데이터 없으면 안내
    if (stepIndex === 1) {
        if (!titleOptions) showStepGuide('step-titles', '주제를 입력하고 "제목 생성하기"를 눌러주세요.');
        else hideStepGuide('step-titles');
    }

    // 나레이션 단계: 데이터 없으면 안내
    if (stepIndex === 2) {
        if (!narrationData) showStepGuide('step-narration', '제목 단계에서 "다음: 나레이션 생성"을 눌러주세요.');
        else hideStepGuide('step-narration');
    }

    // 음성 단계: 옵션이 비어있으면 자동 로드
    if (stepIndex === 3) {
        const voiceSelect = document.getElementById('tts-voice');
        if (voiceSelect && voiceSelect.options.length === 0) {
            updateVoiceOptions();
        }
    }

    // 확인 단계: 요약 자동 생성
    if (stepIndex === 5) {
        buildConfirmSummary();
    }

    updateTimeline();

    setTimeout(() => {
        const current = document.getElementById(STEPS[stepIndex].id);
        current.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
}

// 진행도(maxReachedStep)는 단조 증가만 가능. 자유 이동으로 이전 단계 재진입 시에도 진행도는 보존된다.
function advanceToStep(stepIndex) {
    maxReachedStep = Math.max(maxReachedStep, stepIndex);
    goToStep(stepIndex);
}

function clickTimelineStep(stepIndex) {
    goToStep(stepIndex);
}

function updateTimeline() {
    const items = document.querySelectorAll('.timeline-item');
    items.forEach((item, i) => {
        item.classList.remove('completed', 'active');
        if (i < maxReachedStep) item.classList.add('completed');
        if (i === currentStepIndex) item.classList.add('active');
    });

    const track = document.querySelector('.timeline-track');
    const progress = maxReachedStep === 0 ? 0 : (maxReachedStep / (STEPS.length - 1)) * 100;
    track.style.setProperty('--timeline-progress', progress + '%');
}

// ──────────────────────────────────
// 스타일 카드 선택
// ──────────────────────────────────


// ──────────────────────────────────
// 유틸리티
// ──────────────────────────────────
function hideAllSteps() {
    STEPS.forEach(step => {
        const el = document.getElementById(step.id);
        if (el) { el.classList.add('hidden'); el.classList.remove('collapsed'); }
    });
}

function showLoading(text) {
    const stepEl = document.getElementById(STEPS[currentStepIndex].id);
    let overlay = stepEl.querySelector('.step-loading-overlay');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.className = 'step-loading-overlay';
        overlay.innerHTML = '<div class="spinner"></div><p class="loading-text"></p>';
        stepEl.appendChild(overlay);
    }
    overlay.querySelector('.loading-text').textContent = text;
    overlay.classList.remove('hidden');
}

function hideLoading() {
    document.querySelectorAll('.step-loading-overlay').forEach(el => el.classList.add('hidden'));
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
        const resp = await authFetch('/api/assets/bgm');
        if (!resp.ok) return;
        bgmList = await resp.json();
        renderBgmList();
    } catch (e) {
        console.error('BGM 목록 로드 실패:', e);
        document.getElementById('bgm-list').innerHTML =
            '<p class="text-dim">BGM 목록을 불러올 수 없습니다</p>';
    }
}

let _bgmPlaying = false;

function renderBgmList() {
    const container = document.getElementById('bgm-list');
    if (bgmList.length === 0) {
        container.innerHTML = '<p class="text-dim">BGM을 업로드하세요</p>';
        return;
    }
    const isSelected = (bgm) => selectedBgm === bgm.filename;
    container.innerHTML = bgmList.map((bgm, i) => `
        <div class="bgm-card ${isSelected(bgm) ? 'selected' : ''}"
             onclick="selectBgm(${i})">
            <div class="bgm-name">${escapeHtml(bgm.filename.replace(/\.(mp3|wav|ogg)$/i, ''))}</div>
            <div class="bgm-duration">${formatTime(bgm.duration)}</div>
            <button class="btn-small bgm-delete-btn" onclick="event.stopPropagation(); deleteBgm('${bgm.id || bgm.filename}', ${i})">삭제</button>
        </div>
        ${isSelected(bgm) ? renderBgmPlayer(i) : ''}
    `).join('');
}

function handleBgmUploadClick() {
    if (bgmList.length >= 3) {
        alert('BGM은 최대 3개까지 업로드 가능합니다.\n기존 BGM을 삭제 후 다시 시도해주세요.');
        return;
    }
    document.getElementById('bgm-upload-input').click();
}

function renderBgmPlayer(index) {
    const bgm = bgmList[index];
    const cur = bgmAudio ? bgmAudio.currentTime || 0 : 0;
    const dur = bgm.duration || 0;
    return `
        <div class="bgm-player" id="bgm-player">
            <button class="bgm-player-btn" onclick="event.stopPropagation(); toggleBgmPlayPause()">
                ${_bgmPlaying ? '⏸' : '▶'}
            </button>
            <span class="bgm-player-time" id="bgm-player-time">${formatTime(cur)}</span>
            <input type="range" class="bgm-player-bar" id="bgm-player-bar"
                   min="0" max="${Math.floor(dur)}" value="${Math.floor(cur)}" step="1"
                   onclick="event.stopPropagation()"
                   oninput="seekBgm(this.value)">
            <span class="bgm-player-duration">${formatTime(dur)}</span>
        </div>
    `;
}

// ── BGM 업로드 ──
document.getElementById('bgm-upload-input')?.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const btn = document.getElementById('bgm-upload-btn');
    btn.textContent = '업로드 중...';
    btn.disabled = true;

    try {
        const formData = new FormData();
        formData.append('file', file);
        const resp = await authFetch('/api/assets/bgm', { method: 'POST', body: formData });
        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || '업로드 실패');
        }
        await loadBgmList();
    } catch (err) {
        alert(err.message);
    } finally {
        btn.textContent = '+ BGM 업로드';
        btn.disabled = false;
        e.target.value = '';
    }
});

async function deleteBgm(bgmId, index) {
    if (!confirm('이 BGM을 삭제하시겠습니까?')) return;
    try {
        const resp = await authFetch(`/api/assets/bgm/${bgmId}`, { method: 'DELETE' });
        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || '삭제 실패');
        }
        if (selectedBgm === bgmList[index]?.filename) {
            selectedBgm = null;
            bgmDuration = 0;
            if (bgmAudio) { bgmAudio.pause(); bgmAudio = null; _bgmPlaying = false; }
        }
        await loadBgmList();
    } catch (err) {
        alert(err.message);
    }
}

function selectBgm(index) {
    // 기존 재생 중지
    if (bgmAudio) {
        bgmAudio.pause();
        bgmAudio = null;
        _bgmPlaying = false;
    }

    selectedBgm = bgmList[index].filename;
    bgmDuration = bgmList[index].duration;

    // Audio 객체 생성 (preload 안 함, 재생바만 표시)
    const bgm = bgmList[index];
    bgmAudio = new Audio();
    bgmAudio.preload = 'none';
    bgmAudio.src = bgm.url;
    bgmAudio._bgmIdx = index;

    bgmAudio.addEventListener('timeupdate', () => {
        const bar = document.getElementById('bgm-player-bar');
        const timeEl = document.getElementById('bgm-player-time');
        if (bar && bgmAudio) {
            bar.value = Math.floor(bgmAudio.currentTime);
            timeEl.textContent = formatTime(bgmAudio.currentTime);
        }
    });

    bgmAudio.addEventListener('ended', () => {
        _bgmPlaying = false;
        const btn = document.querySelector('.bgm-player-btn');
        if (btn) btn.textContent = '▶';
    });

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

function toggleBgmPlayPause() {
    if (!bgmAudio) return;
    if (_bgmPlaying) {
        bgmAudio.pause();
        _bgmPlaying = false;
    } else {
        bgmAudio.play();
        _bgmPlaying = true;
    }
    const btn = document.querySelector('.bgm-player-btn');
    if (btn) btn.textContent = _bgmPlaying ? '⏸' : '▶';
}

function seekBgm(val) {
    if (bgmAudio) {
        bgmAudio.currentTime = parseFloat(val);
        const timeEl = document.getElementById('bgm-player-time');
        if (timeEl) timeEl.textContent = formatTime(val);
    }
}

function formatTime(sec) {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
}

// 페이지 로드 시 초기화
updateTimeline(0);
loadBgmList();
loadUserProducts();
toggleCategoryFields();  // 카테고리 + 영상 목적 UI 초기 상태 세팅
