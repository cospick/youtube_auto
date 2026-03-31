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

function showCompleted(videoUrl) {
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
connectSSE();
