// --- State (Updated: 2026-02-09 15:05) ---
let authToken = localStorage.getItem('authToken');
let currentUser = null;
let currentImageFile = null;
let updateInterval = null;

const screens = {
    login: document.getElementById('login-screen'),
    pending: document.getElementById('pending-screen'),
    landing: document.getElementById('landing-screen'),
    search: document.getElementById('search-screen'),
    update: document.getElementById('update-screen'),
    userMgmt: document.getElementById('user-mgmt-screen')
};

// --- Init ---
document.addEventListener('DOMContentLoaded', () => {
    console.log("MFi Initializing...");
    initAuth();
    setupEventListeners();
});

function initAuth() {
    const urlParams = new URLSearchParams(window.location.search);
    const tokenFromUrl = urlParams.get('token');

    if (tokenFromUrl) {
        authToken = tokenFromUrl;
        localStorage.setItem('authToken', authToken);
        window.history.replaceState({}, document.title, "/");
    }

    if (authToken) checkAuth(); else showScreen('login');
}

async function checkAuth() {
    try {
        const res = await fetch('/me', {
            headers: { 'Authorization': `Bearer ${authToken}` }
        });

        if (res.ok) {
            currentUser = await res.json();
            localStorage.setItem('currentUser', JSON.stringify(currentUser));
            showScreen('landing');
            updateUI();
        } else if (res.status === 403) {
            showScreen('pending');
        } else {
            logout();
        }
    } catch (e) {
        console.error("Auth check failed:", e);
        logout();
    }
}

function showScreen(id) {
    Object.values(screens).forEach(s => s && s.classList.add('hidden'));
    if (screens[id]) screens[id].classList.remove('hidden');
    if (id === 'update') startPolling(); else stopPolling();
    if (id === 'userMgmt') loadUserList();
}

function updateUI() {
    if (!currentUser) return;
    const nameDisp = document.getElementById('user-display-name');
    if (nameDisp) nameDisp.innerText = currentUser.username;

    document.querySelectorAll('.admin-only').forEach(el => {
        if (currentUser.role === 'admin') el.classList.remove('hidden');
        else el.classList.add('hidden');
    });
}

function logout() {
    localStorage.clear(); authToken = null; currentUser = null;
    stopPolling(); showScreen('login');
}

// --- Events ---
function setupEventListeners() {
    document.getElementById('login-form').addEventListener('submit', handleLogin);

    const googleBtn = document.getElementById('google-login-btn');
    if (googleBtn) googleBtn.addEventListener('click', () => {
        window.location.href = "/auth/google/login";
    });

    ['logout-btn', 'search-logout-btn', 'update-logout-btn', 'pending-logout-btn', 'mgmt-logout-btn'].forEach(id => {
        const btn = document.getElementById(id);
        if (btn) btn.addEventListener('click', logout);
    });

    document.getElementById('go-to-search').addEventListener('click', () => showScreen('search'));
    document.getElementById('go-to-update').addEventListener('click', () => showScreen('update'));
    document.getElementById('go-to-user-mgmt').addEventListener('click', () => showScreen('userMgmt'));

    ['back-to-landing-from-search', 'back-to-landing-from-update', 'back-to-landing-from-mgmt'].forEach(id => {
        const btn = document.getElementById(id);
        if (btn) btn.addEventListener('click', () => showScreen('landing'));
    });

    const dz = document.getElementById('drop-zone');
    const fi = document.getElementById('file-input');
    if (dz) dz.addEventListener('click', () => fi.click());
    if (dz) {
        dz.addEventListener('dragover', (e) => { e.preventDefault(); dz.style.borderColor = "#6366f1"; });
        dz.addEventListener('dragleave', () => { dz.style.borderColor = ""; });
        dz.addEventListener('drop', (e) => {
            e.preventDefault(); dz.style.borderColor = "";
            handleFile(e.dataTransfer.files[0]);
        });
    }
    if (fi) fi.addEventListener('change', (e) => handleFile(e.target.files[0]));

    window.addEventListener('paste', (e) => {
        if (screens.search && screens.search.classList.contains('hidden')) return;
        const items = (e.clipboardData || e.originalEvent.clipboardData).items;
        for (let item of items) if (item.type.includes('image')) handleFile(item.getAsFile());
    });

    document.getElementById('search-btn').addEventListener('click', runSearch);
    document.getElementById('run-update-btn').addEventListener('click', runUpdate);
    document.getElementById('stop-update-btn').addEventListener('click', runStopUpdate);

    // ★ 모달 이벤트 초기화 (is-visible 기반)
    const modal = document.getElementById('image-modal');
    const modalClose = document.getElementById('modal-close-btn');
    if (modalClose) {
        modalClose.onclick = () => {
            console.log("Closing modal (btn)");
            modal.classList.remove('is-visible');
        };
    }
    if (modal) {
        modal.onclick = (e) => {
            if (e.target === modal) {
                console.log("Closing modal (backdrop)");
                modal.classList.remove('is-visible');
            }
        };
    }
}

function handleFile(file) {
    if (!file || !file.type.startsWith('image/')) return;
    currentImageFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        const prev = document.getElementById('preview-img');
        if (prev) { prev.src = e.target.result; prev.classList.remove('hidden'); }
        const content = document.querySelector('.drop-zone-content');
        if (content) content.classList.add('hidden');
        const searchBtn = document.getElementById('search-btn');
        if (searchBtn) searchBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

// ★ Global Accessibility
function openImageModal(url, title) {
    console.log("openImageModal triggering for:", url);
    const modal = document.getElementById('image-modal');
    const modalImg = document.getElementById('modal-img');
    const modalCaption = document.getElementById('modal-caption');

    if (modal && modalImg) {
        modalImg.src = url;
        if (modalCaption) modalCaption.innerText = title || "문항 이미지 확대";
        modal.classList.add('is-visible');
        console.log("is-visible added. Display state:", getComputedStyle(modal).display);
    } else {
        console.error("Critical: Modal elements missing!", { modal, modalImg });
    }
}

async function handleLogin(e) {
    e.preventDefault();
    const u = document.getElementById('username').value;
    const p = document.getElementById('password').value;
    const err = document.getElementById('login-error');
    try {
        const fd = new FormData(); fd.append('username', u); fd.append('password', p);
        const res = await fetch('/token', { method: 'POST', body: fd });
        if (res.ok) {
            const data = await res.json();
            authToken = data.access_token;
            localStorage.setItem('authToken', authToken);
            currentUser = { username: u, role: u === 'admin' ? 'admin' : 'user' };
            localStorage.setItem('currentUser', JSON.stringify(currentUser));
            checkAuth();
        } else {
            err.innerText = "로그인 실패: 아이디/비밀번호 확인";
        }
    } catch (e) { err.innerText = "서버 연결 오류"; }
}

async function runSearch() {
    const ov = document.getElementById('loading-overlay');
    if (ov) ov.classList.remove('hidden');
    const fd = new FormData(); fd.append('file', currentImageFile);
    try {
        const res = await fetch('/search', {
            method: 'POST',
            headers: { 'Authorization': `Bearer ${authToken}` },
            body: fd
        });
        if (res.ok) renderResults(await res.json());
        else if (res.status === 403) alert("계정 승인이 대기 중입니다.");
        else alert("검색 중 오류가 발생했습니다.");
    } catch (e) { alert("검색 실패"); }
    finally { if (ov) ov.classList.add('hidden'); }
}

function renderResults(results) {
    const grid = document.getElementById('results-grid');
    const count = document.getElementById('result-count');
    grid.innerHTML = '';
    if (count) count.innerText = results.length;

    results.forEach(r => {
        const card = document.createElement('div');
        card.className = 'glass-card result-card';

        // 인라인 onclick 대신 addEventListener 사용을 위해 요소 생성 후 이벤트 부착
        const container = document.createElement('div');
        container.className = 'result-img-container';
        container.innerHTML = `
            <img src="${r.image_url}">
            <div class="img-hover-overlay"><span>🔍 크게 보기</span></div>
        `;
        // 클로저를 이용해 안전하게 이벤트 전달
        container.onclick = () => openImageModal(r.image_url, r.source_title);

        const info = document.createElement('div');
        info.className = 'result-info';
        info.innerHTML = `
            <span class="result-source">${r.source_title}</span>
            <div class="result-footer">
                <span>ID: ${r.problem_id.substring(0, 6)}</span>
                <span class="similarity-badge">${(r.similarity * 100).toFixed(1)}% 매칭</span>
            </div>
        `;

        card.appendChild(container);
        card.appendChild(info);
        grid.appendChild(card);
    });
    document.getElementById('results-section').classList.remove('hidden');
}

// --- Admin Update Polling ---
function startPolling() { fetchStats(); updateInterval = setInterval(fetchStats, 3000); }
function stopPolling() { clearInterval(updateInterval); }

async function fetchStats() {
    try {
        const res = await fetch('/admin/stats', { headers: { 'Authorization': `Bearer ${authToken}` } });
        if (!res.ok) return;
        const d = await res.json();

        const elTotal = document.getElementById('total-embeddings');
        const elPending = document.getElementById('pending-count');
        const elLast = document.getElementById('last-updated');
        if (elTotal) elTotal.innerText = d.total_embeddings.toLocaleString();
        if (elPending) elPending.innerText = d.pending_count.toLocaleString();
        if (elLast) elLast.innerText = d.last_updated;

        const btn = document.getElementById('run-update-btn');
        const stopBtn = document.getElementById('stop-update-btn');
        if (!stopBtn) console.warn("DEBUG: stop-update-btn not found in DOM!");

        if (btn) {
            if (d.update_in_progress) {
                btn.innerText = "업데이트 진행 중..."; btn.disabled = true;
                if (stopBtn) stopBtn.classList.remove('hidden');
            } else {
                btn.innerText = "지금 업데이트 시작하기"; btn.disabled = false;
                if (stopBtn) stopBtn.classList.add('hidden');
            }
        }

        const progSection = document.getElementById('update-progress-container');
        if (d.update_in_progress) {
            if (progSection) { progSection.classList.remove('hidden'); progSection.style.display = 'block'; }
            const elPRemaining = document.getElementById('p-remaining');
            const elPSpeed = document.getElementById('p-speed');
            const elPEta = document.getElementById('p-eta');
            if (elPRemaining) elPRemaining.innerText = d.pending_count.toLocaleString();
            if (elPSpeed) elPSpeed.innerText = d.processed_this_session > 0 ? `${d.items_per_min} it/m` : "준비 중...";

            let etaText = "계산 중...";
            if (d.processed_this_session > 0) {
                const totalMins = d.estimated_minutes_left;
                if (totalMins >= 60) {
                    const h = Math.floor(totalMins / 60);
                    const m = Math.round(totalMins % 60);
                    etaText = `${h}h ${m}m`;
                } else {
                    etaText = `${totalMins}m`;
                }
            }
            if (elPEta) elPEta.innerText = etaText;

            const total = Math.max(1, d.pending_count + d.processed_this_session);
            const percent = (d.processed_this_session / total * 100).toFixed(1);
            const elPercent = document.getElementById('update-percentage');
            const elBar = document.getElementById('update-progress-bar');
            const elStat = document.getElementById('update-status-text');
            if (elPercent) elPercent.innerText = `${percent}%`;
            if (elBar) elBar.style.width = `${percent}%`;
            if (elStat) elStat.innerText = d.processed_this_session > 0
                ? `수집 중... (${d.processed_this_session}개 완료)` : `엔진 준비 및 DB 스캔 중...`;
        } else if (progSection) {
            progSection.classList.add('hidden'); progSection.style.display = 'none';
        }
    } catch (e) { console.error("Stats Error:", e); }
}

async function runUpdate() {
    const btn = document.getElementById('run-update-btn');
    if (!btn) return;
    btn.disabled = true;
    btn.innerText = "업데이트 진행 중...";
    try {
        const res = await fetch('/admin/update-embeddings', {
            method: 'POST',
            headers: { 'Authorization': `Bearer ${authToken}` }
        });
        if (res.ok) {
            console.log("Update task triggered successfully.");
            await fetchStats();
        } else {
            alert("이미 진행 중이거나 권한이 없습니다.");
            btn.disabled = false; btn.innerText = "지금 업데이트 시작하기";
        }
    } catch (e) {
        console.error("RunUpdate Failure:", e);
        btn.disabled = false; btn.innerText = "지금 업데이트 시작하기";
    }
}

async function runStopUpdate() {
    const btn = document.getElementById('stop-update-btn');
    if (!btn) return;
    btn.disabled = true;
    btn.innerText = "중단 요청 중...";
    try {
        const res = await fetch('/admin/stop-update', {
            method: 'POST',
            headers: { 'Authorization': `Bearer ${authToken}` }
        });
        if (res.ok) {
            console.log("Stop request sent.");
            await fetchStats();
        } else {
            alert("중단 요청에 실패했습니다.");
            btn.disabled = false; btn.innerText = "업데이트 중단하기";
        }
    } catch (e) {
        console.error("StopUpdate Failure:", e);
        btn.disabled = false; btn.innerText = "업데이트 중단하기";
    }
}

async function loadUserList() {
    try {
        const res = await fetch('/admin/users', {
            headers: { 'Authorization': `Bearer ${authToken}` }
        });
        if (res.ok) {
            const users = await res.json();
            renderUserList(users);
        }
    } catch (e) { console.error("Load users error:", e); }
}

function renderUserList(users) {
    const body = document.getElementById('user-list-body');
    if (!body) return;
    body.innerHTML = '';
    users.forEach(u => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td><img src="${u.picture_url || 'https://www.w3schools.com/howto/img_avatar.png'}" class="user-avatar"></td>
            <td>
                <div class="user-id">${u.username}</div>
                <div class="user-email">${u.email || ''}</div>
            </td>
            <td><span class="status-badge ${u.is_approved ? 'approved' : 'pending'}">${u.is_approved ? '승인됨' : '대기 중'}</span></td>
            <td>
                <div class="action-btns">
                    ${!u.is_approved ? `<button onclick="approveUser('${u.id}')" class="btn-sm btn-approve">승인</button>` : ''}
                    <button onclick="confirmDeleteUser('${u.id}')" class="btn-sm btn-delete">삭제</button>
                </div>
            </td>
        `;
        body.appendChild(tr);
    });
}

async function approveUser(id) {
    try {
        const res = await fetch(`/admin/users/${id}/approve`, {
            method: 'POST',
            headers: { 'Authorization': `Bearer ${authToken}` }
        });
        if (res.ok) loadUserList();
    } catch (e) { alert("승인 실패"); }
}

function confirmDeleteUser(id) {
    if (confirm("정말 이 사용자를 삭제하시겠습니까? (탈퇴 처리)")) {
        deleteUser(id);
    }
}

async function deleteUser(id) {
    try {
        const res = await fetch(`/admin/users/${id}`, {
            method: 'DELETE',
            headers: { 'Authorization': `Bearer ${authToken}` }
        });
        if (res.ok) loadUserList();
    } catch (e) { alert("삭제 실패"); }
}

// ★ Expose functions to global scope
window.approveUser = approveUser;
window.confirmDeleteUser = confirmDeleteUser;
window.openImageModal = openImageModal;
