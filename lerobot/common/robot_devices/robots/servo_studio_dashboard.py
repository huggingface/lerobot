#!/usr/bin/env python3
"""
Embedded Single-Page HTML5/CSS3 Dashboard Generator for WebUI Calibration Studio.
Provides clean UI components, responsive layout, and zero external web dependencies.
"""

def get_dashboard_html() -> str:
    """Generates the single-page HTML5/CSS3 dashboard string for the WebUI studio.

    Returns:
        str: Complete HTML string with inline CSS and native JavaScript.
    """
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SO-ARM100 / LeRobot WebUI Calibration Studio</title>
    <style>
        :root {
            --bg-dark: #0f172a;
            --card-bg: #1e293b;
            --accent-blue: #38bdf8;
            --accent-green: #22c55e;
            --accent-red: #ef4444;
            --accent-amber: #f59e0b;
            --text-main: #f8fafc;
            --text-sub: #94a3b8;
            --border-color: #334155;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
        body { background-color: var(--bg-dark); color: var(--text-main); padding: 20px; line-height: 1.5; }
        
        .header { display: flex; justify-content: space-between; align-items: center; padding-bottom: 20px; border-bottom: 1px solid var(--border-color); margin-bottom: 20px; }
        .header h1 { font-size: 1.5rem; font-weight: 700; color: var(--accent-blue); }
        .badge { background: #0284c7; color: white; padding: 4px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 600; }
        
        .grid { display: grid; grid-template-columns: 1fr 2fr; gap: 20px; }
        @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
        
        .card { background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 10px; padding: 20px; }
        .card h2 { font-size: 1.1rem; color: var(--text-sub); margin-bottom: 15px; border-bottom: 1px solid var(--border-color); padding-bottom: 8px; }
        
        .servo-select { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin-bottom: 20px; }
        .btn-servo { background: #334155; color: var(--text-main); border: 1px solid var(--border-color); padding: 12px; border-radius: 6px; cursor: pointer; text-align: left; font-weight: 600; transition: all 0.2s; }
        .btn-servo:hover { background: #475569; }
        .btn-servo.active { background: #0284c7; border-color: var(--accent-blue); }

        .telemetry-box { background: #090d16; border-radius: 8px; padding: 15px; margin-bottom: 20px; font-family: monospace; }
        .telemetry-row { display: flex; justify-content: space-between; margin-bottom: 6px; }
        .telemetry-value { color: var(--accent-blue); font-weight: bold; }

        .action-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 15px; }
        .btn { padding: 10px; border-radius: 6px; border: none; cursor: pointer; font-weight: 600; color: white; transition: opacity 0.2s; }
        .btn:hover { opacity: 0.9; }
        .btn-green { background: var(--accent-green); }
        .btn-amber { background: var(--accent-amber); }
        .btn-blue { background: #0284c7; }
        .btn-red { background: var(--accent-red); }
        
        .status-banner { margin-top: 15px; padding: 10px; border-radius: 6px; font-size: 0.9rem; font-weight: 500; display: none; }
        .status-success { background: rgba(34, 197, 94, 0.2); border: 1px solid var(--accent-green); color: var(--accent-green); }
        .status-error { background: rgba(239, 68, 68, 0.2); border: 1px solid var(--accent-red); color: var(--accent-red); }
    </style>
</head>
<body>
    <div class="header">
        <h1>SO-ARM100 WebUI Calibration Studio</h1>
        <span class="badge">LeRobot Integration</span>
    </div>

    <div class="grid">
        <div class="card">
            <h2>1. Select Joint</h2>
            <div class="servo-select" id="servoList"></div>
            
            <h2>Global Actions</h2>
            <div class="action-grid" style="grid-template-columns: 1fr;">
                <button class="btn btn-green" onclick="saveCalibration()">💾 Save Calibration to Disk</button>
                <button class="btn btn-red" onclick="toggleTorque()" id="torqueBtn">⚡ Enable Torque (30% Cap)</button>
            </div>
        </div>

        <div class="card">
            <h2>2. Interactive 3-Point Calibration</h2>
            <div class="telemetry-box">
                <div class="telemetry-row"><span>Active Joint:</span><span class="telemetry-value" id="activeJointName">None</span></div>
                <div class="telemetry-row"><span>Live Position:</span><span class="telemetry-value" id="livePos">---- ticks</span></div>
                <div class="telemetry-row"><span>Captured Min:</span><span class="telemetry-value" id="capMin">----</span></div>
                <div class="telemetry-row"><span>Captured Home:</span><span class="telemetry-value" id="capHome">----</span></div>
                <div class="telemetry-row"><span>Captured Max:</span><span class="telemetry-value" id="capMax">----</span></div>
            </div>

            <h2>3-Point Position Capture</h2>
            <div class="action-grid">
                <button class="btn btn-blue" onclick="capture('min')">📍 Capture Min</button>
                <button class="btn btn-amber" onclick="capture('home')">🏠 Capture Home</button>
                <button class="btn btn-blue" onclick="capture('max')">📍 Capture Max</button>
            </div>

            <h2 style="margin-top: 20px;">Safe Bounded Test Drive</h2>
            <div class="action-grid">
                <button class="btn btn-amber" onclick="moveTarget(2048)">Move to 2048 (Zero)</button>
                <button class="btn btn-green" onclick="moveToHome()">Move to Home</button>
            </div>

            <div id="statusBanner" class="status-banner"></div>
        </div>
    </div>

    <script>
        let currentServo = 1;
        const servos = {
            1: 'Shoulder Pan', 2: 'Shoulder Lift', 3: 'Elbow Flex',
            4: 'Wrist Flex', 5: 'Wrist Roll', 6: 'Gripper'
        };

        function renderServos() {
            const container = document.getElementById('servoList');
            container.innerHTML = '';
            Object.keys(servos).forEach(id => {
                const btn = document.createElement('button');
                btn.className = `btn-servo ${id == currentServo ? 'active' : ''}`;
                btn.innerText = `[${id}] ${servos[id]}`;
                btn.onclick = () => selectServo(parseInt(id));
                container.appendChild(btn);
            });
        }

        function selectServo(id) {
            currentServo = id;
            document.getElementById('activeJointName').innerText = `[${id}] ${servos[id]}`;
            renderServos();
            fetchStatus();
        }

        async function fetchStatus() {
            try {
                const res = await fetch(`/api/status?servo_id=${currentServo}`);
                const data = await res.json();
                if(data.success) {
                    document.getElementById('livePos').innerText = `${data.live_position} ticks`;
                    document.getElementById('capMin').innerText = data.calib.min !== null ? data.calib.min : '----';
                    document.getElementById('capHome').innerText = data.calib.home !== null ? data.calib.home : '----';
                    document.getElementById('capMax').innerText = data.calib.max !== null ? data.calib.max : '----';
                }
            } catch(e) {}
        }

        async function capture(point) {
            const res = await fetch(`/api/capture_${point}?servo_id=${currentServo}`, { method: 'POST' });
            const data = await res.json();
            showStatus(data.message, data.success);
            fetchStatus();
        }

        async function moveTarget(pos) {
            const res = await fetch(`/api/move_servo`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ servo_id: currentServo, target_ticks: pos })
            });
            const data = await res.json();
            showStatus(data.message, data.success);
        }

        async function moveToHome() {
            const res = await fetch(`/api/move_home?servo_id=${currentServo}`, { method: 'POST' });
            const data = await res.json();
            showStatus(data.message, data.success);
        }

        async function saveCalibration() {
            const res = await fetch(`/api/save_calibration`, { method: 'POST' });
            const data = await res.json();
            showStatus(data.message, data.success);
        }

        async function toggleTorque() {
            const res = await fetch(`/api/torque`, { method: 'POST' });
            const data = await res.json();
            showStatus(data.message, data.success);
        }

        function showStatus(msg, isSuccess) {
            const banner = document.getElementById('statusBanner');
            banner.className = `status-banner ${isSuccess ? 'status-success' : 'status-error'}`;
            banner.innerText = msg;
            banner.style.display = 'block';
            setTimeout(() => banner.style.display = 'none', 4000);
        }

        renderServos();
        selectServo(1);
        setInterval(fetchStatus, 500);
    </script>
</body>
</html>"""
