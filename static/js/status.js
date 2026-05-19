/**
 * 진행 상태 페이지 - SSE로 실시간 업데이트
 * phase=images: 이미지 생성 (카드 그리드 + skeleton → 이미지 fade-in)
 * phase=render: 영상 제작 (프로그레스 바)
 */

const params = new URLSearchParams(window.location.search);
const jobId = params.get('job');
const phase = params.get('phase') || 'render';

let cardsBuilt = false;
let revealedImages = new Set();
let currentSource = null;
// 카드 B 5단계 스테퍼용. index.html의 STEPS_USER_ASSETS와 동일하게 유지해야 함.
const CARD_B_STEPS = [
    { id: 'step-user-script', label: '제목·대본' },
    { id: 'step-user-lines',  label: '자산' },
    { id: 'step-tts',         label: '음성' },
    { id: 'step-bgm',         label: 'BGM' },
    { id: 'step-render',      label: '영상 제작' },
];
let stepperCardB = false;  // /draft-state 호출 성공 시 true
// 페이지 진입 즉시 카드 B 여부 판별 promise를 만들고 보존.
// SSE의 completed 응답이 더 빨리 도착하는 race로 attachDiscardHandler가 등록 안 되던
// 문제를 막기 위해 showCompleted에서 이 promise를 await한다.
let stepperInitPromise = null;

/* ── 초기 UI ── */

function initUI() {
    const title = document.getElementById('page-title');
    if (phase === 'images') {
        title.textContent = '이미지 생성 중';
    } else if (phase === 'clips') {
        title.textContent = 'AI 영상 클립 생성 중';
    } else {
        title.textContent = '영상 제작 중';
        document.getElementById('video-loading-section').classList.remove('hidden');
    }
}

/* ── SSE 연결 ── */

function connectSSE() {
    if (!jobId) return;

    // 기존 연결 정리 (재시도 시 안전한 재진입)
    if (currentSource) {
        currentSource.close();
        currentSource = null;
    }

    currentSource = new EventSource(`/api/jobs/${jobId}/stream`);

    currentSource.onmessage = function(event) {
        const data = JSON.parse(event.data);

        if (data.error && !data.status) {
            currentSource.close();
            currentSource = null;
            return;
        }

        if (phase === 'images') {
            handleImagePhase(data);
        } else if (phase === 'clips') {
            handleClipPhase(data);
        } else {
            handleRenderPhase(data);
        }

        // 이미지 생성 완료 → 미리보기 페이지로
        if (data.status === 'preview_ready' && phase === 'images') {
            currentSource.close();
            currentSource = null;
            setTimeout(() => {
                window.location.href = `/static/preview.html?job=${jobId}`;
            }, 1500);
            return;
        }

        // AI 클립 생성 완료 → 클립 미리보기 페이지로
        if (data.status === 'clips_ready' && phase === 'clips') {
            currentSource.close();
            currentSource = null;
            setTimeout(() => {
                window.location.href = `/static/clip_preview.html?job=${jobId}`;
            }, 1500);
            return;
        }

        // 영상 완성
        if (data.status === 'completed') {
            currentSource.close();
            currentSource = null;
            showCompleted(data.video_url);
            return;
        }

        // 실패
        if (data.status === 'failed') {
            currentSource.close();
            currentSource = null;
            showError(data.error || '알 수 없는 에러');
            return;
        }
    };

    currentSource.onerror = function() {
        currentSource.close();
        currentSource = null;
        startPolling();
    };
}

/* ── 이미지 생성 단계 ── */

function handleImagePhase(data) {
    // 첫 메시지에서 skeleton 카드 생성
    if (!cardsBuilt && data.lines && data.lines.length > 0) {
        buildSkeletonCards(data.lines);
        cardsBuilt = true;
    }

    // 완성된 이미지 표시
    if (data.completed_images) {
        data.completed_images.forEach(function(idx) {
            if (!revealedImages.has(idx)) {
                revealImage(idx);
                revealedImages.add(idx);
            }
        });

        // 진행률 텍스트 (예: "3 / 6")
        if (data.lines) {
            var total = data.lines.length;
            var done = data.completed_images.length;
            document.getElementById('progress-percent').textContent = done + ' / ' + total;
        }
    }

    document.getElementById('progress-step').textContent = data.current_step || '';
}

function buildSkeletonCards(lines) {
    var grid = document.getElementById('image-progress-grid');
    grid.classList.remove('hidden');

    // 카드가 나타나면 프로그레스 바 숨김
    document.getElementById('progress-bar-section').classList.add('hidden');

    grid.innerHTML = lines.map(function(line, i) {
        return '<div class="preview-card" id="status-card-' + i + '">' +
            '<div class="preview-image-wrap">' +
                '<div class="skeleton-shimmer"></div>' +
            '</div>' +
            '<div class="preview-info">' +
                '<span class="line-num">' + (i + 1) + '</span>' +
                '<p class="preview-text">' + escapeHtml(line.text) + '</p>' +
                '<span class="line-motion">' + escapeHtml(line.motion) + '</span>' +
            '</div>' +
        '</div>';
    }).join('');
}

function revealImage(idx) {
    var card = document.getElementById('status-card-' + idx);
    if (!card) return;
    var wrap = card.querySelector('.preview-image-wrap');
    wrap.innerHTML = '<img src="/api/jobs/' + jobId + '/images/' + idx + '?t=' + Date.now() + '" ' +
                     'alt="이미지 ' + (idx + 1) + '" ' +
                     'class="preview-image image-fade-in">';
}

/* ── AI 클립 생성 단계 ── */

function handleClipPhase(data) {
    // 첫 메시지에서 skeleton 카드 생성
    if (!cardsBuilt && data.lines && data.lines.length > 0) {
        buildSkeletonCards(data.lines);
        cardsBuilt = true;
    }

    // 완성된 클립 표시
    if (data.completed_clips) {
        data.completed_clips.forEach(function(idx) {
            if (!revealedImages.has(idx)) {
                revealClip(idx);
                revealedImages.add(idx);
            }
        });

        if (data.lines) {
            var total = data.lines.length;
            var done = data.completed_clips.length;
            document.getElementById('progress-percent').textContent = done + ' / ' + total;
        }
    }

    document.getElementById('progress-step').textContent = data.current_step || '';
}

function revealClip(idx) {
    var card = document.getElementById('status-card-' + idx);
    if (!card) return;
    var wrap = card.querySelector('.preview-image-wrap');
    wrap.innerHTML = '<video src="/api/jobs/' + jobId + '/clips/' + idx + '?t=' + Date.now() + '" ' +
                     'class="preview-image image-fade-in" autoplay loop muted playsinline></video>';
}

/* ── 영상 제작 단계 ── */

function handleRenderPhase(data) {
    // 서버 progress 0.4~1.0 → UI 0~100% 정규화
    var raw = data.progress || 0;
    var percent = Math.max(0, Math.min(100, Math.round(((raw - 0.4) / 0.6) * 100)));

    document.getElementById('progress-fill').style.width = percent + '%';
    document.getElementById('progress-percent').textContent = percent + '%';
    document.getElementById('progress-step').textContent = data.current_step || '';

    // 로딩 텍스트도 현재 단계 반영
    var loadingText = document.querySelector('.video-loading-text');
    if (loadingText) {
        loadingText.textContent = data.current_step || '영상 제작 중...';
    }
}

/* ── 폴링 폴백 ── */

function startPolling() {
    var interval = setInterval(async function() {
        try {
            var resp = await authFetch('/api/jobs/' + jobId);
            var data = await resp.json();

            if (phase === 'images') {
                document.getElementById('progress-step').textContent = data.current_step || '';

                if (data.status === 'preview_ready') {
                    clearInterval(interval);
                    window.location.href = '/static/preview.html?job=' + jobId;
                } else if (data.status === 'failed') {
                    clearInterval(interval);
                    showError(data.error || '알 수 없는 에러');
                }
            } else if (phase === 'clips') {
                document.getElementById('progress-step').textContent = data.current_step || '';

                if (data.status === 'clips_ready') {
                    clearInterval(interval);
                    window.location.href = '/static/clip_preview.html?job=' + jobId;
                } else if (data.status === 'failed') {
                    clearInterval(interval);
                    showError(data.error || '알 수 없는 에러');
                }
            } else {
                handleRenderPhase(data);

                if (data.status === 'completed') {
                    clearInterval(interval);
                    showCompleted(data.video_url);
                } else if (data.status === 'failed') {
                    clearInterval(interval);
                    showError(data.error || '알 수 없는 에러');
                }
            }
        } catch (e) {
            // 네트워크 에러 시 계속 재시도
        }
    }, 2000);
}

/* ── 공통 ── */

async function showCompleted(videoUrl) {
    // stepperCardB 결정될 때까지 대기 — SSE가 initCardBStepper보다 빨리 도착하는
    // race 방지 (카드 B에서 attachDiscardHandler/attachFinalizeHandler 등록 누락 차단).
    if (stepperInitPromise) {
        try { await stepperInitPromise; } catch (e) {}
    }

    // 비디오 로딩 플레이스홀더 숨김
    var loadingSection = document.getElementById('video-loading-section');
    if (loadingSection) loadingSection.classList.add('hidden');

    // 프로그레스 바 100%
    document.getElementById('progress-fill').style.width = '100%';
    document.getElementById('progress-percent').textContent = '100%';

    // 완료 섹션 표시
    document.getElementById('completed-section').classList.remove('hidden');
    if (videoUrl) {
        document.getElementById('download-link').href = videoUrl;
        var video = document.getElementById('video-preview');
        video.src = videoUrl;
        video.classList.add('video-fade-in');
    }

    // 카드 B 전용: 다운로드/새 영상 핸들러 + 안내문 + 스테퍼 step-render 완료 표시
    if (stepperCardB) {
        attachFinalizeHandler();
        attachDiscardHandler();
        const hint = document.getElementById('finalize-hint');
        if (hint) hint.classList.remove('hidden');
        renderStepperOnStatus(true);
    }
}

function attachFinalizeHandler() {
    const link = document.getElementById('download-link');
    if (!link || link.dataset.finalizeAttached) return;
    link.dataset.finalizeAttached = '1';
    link.addEventListener('click', function() {
        // 다운로드는 브라우저 기본 동작 그대로 진행, finalize는 best-effort.
        try {
            fetch('/api/jobs/' + jobId + '/finalize', {
                method: 'POST',
                credentials: 'include',
                keepalive: true,
            }).catch(function() {});
        } catch (e) {}
    });
}

function attachDiscardHandler() {
    // '새 영상 만들기' <a href="/"> 를 버튼-like로 가로채 discard 호출 후 이동
    const btnGroup = document.querySelector('#completed-section .btn-group');
    if (!btnGroup) return;
    const links = btnGroup.querySelectorAll('a');
    // download-link가 아닌 다른 a (새 영상 만들기) 찾기
    links.forEach(function(a) {
        if (a.id === 'download-link') return;
        if (a.dataset.discardAttached) return;
        a.dataset.discardAttached = '1';
        a.addEventListener('click', async function(e) {
            e.preventDefault();
            const targetHref = a.getAttribute('href') || '/';
            a.classList.add('disabled');
            try {
                await fetch('/api/jobs/' + jobId + '/discard', {
                    method: 'POST',
                    credentials: 'include',
                });
            } catch (err) {
                // 서버 실패해도 사용자는 새 영상 만들기로 이동시킴 — 30일 크론으로 결국 정리됨
            }
            window.location.href = targetHref;
        });
    });
}

function renderStepperOnStatus(jobCompleted) {
    const track = document.getElementById('timeline-track');
    const tl = document.getElementById('workflow-timeline');
    if (!track || !tl) return;
    tl.classList.remove('hidden');
    const currentIdx = jobCompleted ? 4 : 4;  // 영상 제작 단계
    track.innerHTML = CARD_B_STEPS.map(function(step, i) {
        const completed = i < currentIdx ? ' completed' : '';
        const active = i === currentIdx ? ' active' : '';
        return '<li class="timeline-item' + completed + active + '" data-step="' + i + '" onclick="statusStepperClick(' + i + ')">' +
            '<span class="timeline-dot"></span>' +
            '<span class="timeline-label">' + step.label + '</span>' +
        '</li>';
    }).join('');
}

function statusStepperClick(idx) {
    // 1~4 단계: index.html로 돌아가며 reopen 흐름 트리거. 5단계: 현재 페이지라 no-op.
    if (idx === 4) return;
    // 진입 시점에 stepperCardB가 true이고 영상이 완료된 상태에서만 reopen 가능.
    if (!stepperCardB) return;
    window.location.href = '/static/index.html?job_id=' + jobId + '&restore=1';
}

async function initCardBStepper() {
    if (!jobId) return;
    try {
        const resp = await authFetch('/api/jobs/' + jobId + '/draft-state');
        if (!resp.ok) return;  // 404면 카드 A — 스테퍼 미노출
        const data = await resp.json();
        if (data && data.generation_mode === 'user_assets') {
            stepperCardB = true;
            const jobCompleted = data.status === 'completed' && !data.intermediates_purged;
            renderStepperOnStatus(jobCompleted);
        }
    } catch (e) {
        // 조용히 실패 — 스테퍼 미노출
    }
}

async function retryImages() {
    var jobId = new URLSearchParams(window.location.search).get('job');
    if (!jobId) return;

    var btn = document.getElementById('retry-images-btn');
    if (btn) { btn.disabled = true; btn.textContent = '재시도 중...'; }

    try {
        var resp = await authFetch('/api/jobs/' + jobId + '/retry-images', { method: 'POST' });
        if (!resp.ok) {
            var err = await resp.json();
            alert(err.detail || '재시도 실패');
            if (btn) { btn.disabled = false; btn.textContent = '이미지 생성 재시도'; }
            return;
        }
        // 상태 초기화 후 SSE 재연결
        cardsBuilt = false;
        revealedImages = new Set();
        document.getElementById('error-section').classList.add('hidden');
        document.getElementById('image-progress-grid').innerHTML = '';
        document.getElementById('image-progress-grid').classList.add('hidden');
        document.getElementById('progress-bar-section').classList.remove('hidden');
        document.getElementById('progress-step').textContent = '이미지 생성 재시도 중...';
        document.getElementById('progress-percent').textContent = '0%';
        document.getElementById('progress-fill').style.width = '0%';
        connectSSE();
    } catch (e) {
        alert('재시도 요청 실패: ' + e.message);
        if (btn) { btn.disabled = false; btn.textContent = '이미지 생성 재시도'; }
    }
}

function showError(message) {
    // 로딩 플레이스홀더 숨김
    var loadingSection = document.getElementById('video-loading-section');
    if (loadingSection) loadingSection.classList.add('hidden');

    var is503 = message.includes('503') || message.includes('UNAVAILABLE');
    var is429 = message.includes('429') || message.includes('RESOURCE_EXHAUSTED');
    var displayMsg;
    if (is503) {
        displayMsg = 'Google AI 서버가 현재 많이 바쁜 상태입니다.\n아래 버튼을 눌러 재시도해주세요.';
    } else if (is429) {
        displayMsg = 'API 요청 횟수 제한에 도달했습니다.\n1분 후에 재시도해주세요.';
    } else {
        displayMsg = message;
    }

    document.getElementById('error-section').classList.remove('hidden');
    document.getElementById('error-message').textContent = displayMsg;
}

function escapeHtml(str) {
    var div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

initUI();
stepperInitPromise = initCardBStepper();
connectSSE();
