/* ============================================
   Dashboard JS - Socket.IO + Utils (dùng chung mọi trang)
   ============================================ */

let socket = null;

function initSocket() {
    if (typeof io === 'undefined') return;
    socket = io();
    socket.on('connect', function() { updateConnectionStatus(true); });
    socket.on('disconnect', function() { updateConnectionStatus(false); });
    socket.on('connect_error', function() { updateConnectionStatus(false); });
}

function updateConnectionStatus(connected) {
    const el = document.getElementById('connection-status');
    if (!el) return;
    el.classList.toggle('online', connected);
    el.classList.toggle('offline', !connected);
    el.innerHTML = connected
        ? '<i class="bi bi-circle-fill"></i> Máy chủ đang kết nối'
        : '<i class="bi bi-circle-fill"></i> Mất kết nối máy chủ';
}

// HTML Escape (chống XSS)
function escapeHtml(text) {
    if (text === null || text === undefined) return '';
    const div = document.createElement('div');
    div.appendChild(document.createTextNode(String(text)));
    return div.innerHTML;
}

function renderTopbarDate() {
    const el = document.getElementById('topbar-date');
    if (!el) return;
    el.textContent = new Date().toLocaleDateString('vi-VN', {
        weekday: 'long', day: '2-digit', month: '2-digit', year: 'numeric'
    });
}

document.addEventListener('DOMContentLoaded', function() {
    initSocket();
    renderTopbarDate();
});
