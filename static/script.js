// --- State ---
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
    console.log("MCAT_QANDA Initializing... (Production-ready Version)");
    initAuth();
    setupEventListeners();
});

// Google OAuth 콜백 시 URL에 포함된 토큰 처리
function initAuth() {
    const urlParams = new URLSearchParams(window.location.search);
    const tokenFromUrl = urlParams.get('token');

    if (tokenFromUrl) {
        authToken = tokenFromUrl;
        localStorage.setItem('authToken', authToken);
        // 토큰 노출 방지를 위해 URL 정리
        window.history.replaceState({}, document.title, "/");
    }

    if (authToken) checkAuth(); else showScreen('login');
}

async function checkAuth() {
    try {
        // 서버에서 최신 유저 정보(is_approved 포함) 가져오기
        const res = await fetch('/admin/stats', { // stats API가 auth 체크를 겸함
            headers: { 'Authorization': `Bearer ${authToken}` }
        });

        if (res.ok) {
            // 편의상 stats API의 응답과는 별개로 세션 정보를 위해 /me 엔드포인트가 있으면 좋으나 
            // 현재는 stats가 200이면 승인된 상태로 간주 (403이면 미승인)
            // 임시로 localStorage의 정보를 사용하거나 별도 API 호출 필요
            // 여기서는 단순화를 위해 localStorage 사용 후 403 에러 시 처리
            const userJson = localStorage.getItem('currentUser');
            if (userJson) currentUser = JSON.parse(userJson);
            showScreen('landing');
            updateUI();
        } else if (res.status === 403) {
            // 승인 대기 상태
            showScreen('pending');
        } else {
            logout();
        }
    } catch (e) {
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

    // 구글 로그인 리다이렉트
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
        const items = e.clipboardData.items;
        for (let item of items) if (item.type.includes('image')) handleFile(item.getAsFile());
    });

    document.getElementById('search-btn').addEventListener('click', runSearch);
    document.getElementById('run-update-btn').addEventListener('click', runUpdate);
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
            checkAuth(); // 승인 여부 체크를 포함한 진입
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
        card.innerHTML = `
            <div class="result-img-container"><img src="${r.image_url}"></div>
            <div class="result-info">
                <span class="result-source">${r.source_title}</span>
                <div class="result-footer">
                    <span>ID: ${r.problem_id.substring(0, 6)}</span>
                    <span class="similarity-badge">${(r.similarity * 100).toFixed(1)}% 매칭</span>
                </div>
            </div>
        `;
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

        document.getElementById('total-embeddings').innerText = d.total_embeddings.toLocaleString();
        document.getElementById('pending-count').innerText = d.pending_count.toLocaleString();
        document.getElementById('last-updated').innerText = d.last_updated;

        const btn = document.getElementById('run-update-btn');
        if (btn) {
            if (d.update_in_progress) {
                btn.innerText = "업데이트 진행 중..."; btn.disabled = true;
            } else {
                btn.innerText = "지금 업데이트 시작하기"; btn.disabled = false;
            }
        }

        const progSection = document.getElementById('update-progress-container');
        if (d.update_in_progress) {
            if (progSection) { progSection.classList.remove('hidden'); progSection.style.display = 'block'; }
            document.getElementById('p-remaining').innerText = d.pending_count.toLocaleString();
            document.getElementById('p-speed').innerText = d.processed_this_session > 0 ? `${d.items_per_min} it/m` : "준비 중...";

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
            document.getElementById('p-eta').innerText = etaText;

            const total = Math.max(1, d.pending_count + d.processed_this_session);
            const percent = (d.processed_this_session / total * 100).toFixed(1);
            document.getElementById('update-percentage').innerText = `${percent}%`;
            document.getElementById('update-progress-bar').style.width = `${percent}%`;
            document.getElementById('update-status-text').innerText = d.processed_this_session > 0
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

// --- User Management Logic ---
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

// 외부에서 호출 가능하도록 노출
window.approveUser = approveUser;
window.confirmDeleteUser = confirmDeleteUser;
