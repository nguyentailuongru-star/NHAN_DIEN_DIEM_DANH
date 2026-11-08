/* ============================================
   Dashboard JS - Socket.IO + Utils
   ============================================ */

// Socket.IO Connection
let socket = null;

function initSocket() {
    socket = io();

    socket.on('connect', function() {
        console.log('Connected to server');
        updateConnectionStatus(true);
    });

    socket.on('disconnect', function() {
        console.log('Disconnected from server');
        updateConnectionStatus(false);
    });

    socket.on('connect_error', function(err) {
        console.error('Connection error:', err);
        updateConnectionStatus(false);
    });
}

function updateConnectionStatus(connected) {
    const el = document.getElementById('connection-status');
    if (el) {
        el.innerHTML = connected
            ? '<i class="bi bi-circle-fill text-success"></i> Online'
            : '<i class="bi bi-circle-fill text-danger"></i> Offline';
    }
}

// HTML Escape (chống XSS)
function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.appendChild(document.createTextNode(text));
    return div.innerHTML;
}

// Init khi page load
document.addEventListener('DOMContentLoaded', function() {
    initSocket();
});
