import threading
import time
import cv2
from flask import Flask, Response, jsonify

app = Flask(__name__)

web_state = {
    "decision": "SAFE TO REVERSE",
    "threat_level": "SAFE",
    "risk_score": 0,
    "smoothed_risk": 0.0,
    "object_count": 0,
    "approaching_count": 0,
    "vehicle_count_total": 0,
    "near_miss_count": 0,
    "danger_event_count": 0,
    "highest_risk_score": 0,
    "highest_risk_object": None,
    "occupancy_percent": 0.0,
    "traffic_state": "LIGHT",
    "frame_jpeg": None,
}
_state_lock = threading.Lock()


def update_state(**kwargs):
    with _state_lock:
        web_state.update(kwargs)


def _generate_frames():
    while True:
        with _state_lock:
            jpeg = web_state.get("frame_jpeg")
        if jpeg is None:
            time.sleep(0.05)
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')


@app.route('/video')
def video():
    return Response(
        _generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/snapshot')
def snapshot():
    with _state_lock:
        jpeg = web_state.get("frame_jpeg")
    if jpeg is None:
        return Response(status=204)
    return Response(jpeg, mimetype='image/jpeg')


@app.route('/api/status')
def api_status():
    with _state_lock:
        state = {k: v for k, v in web_state.items() if k != "frame_jpeg"}
    return jsonify(state)


@app.route('/')
def index():
    with _state_lock:
        decision = web_state.get("decision", "SAFE TO REVERSE")
        threat_level = web_state.get("threat_level", "SAFE")
        smoothed_risk = web_state.get("smoothed_risk", 0.0)
        object_count = web_state.get("object_count", 0)
        approaching_count = web_state.get("approaching_count", 0)
        traffic_state = web_state.get("traffic_state", "LIGHT")
        occupancy_percent = web_state.get("occupancy_percent", 0.0)
        vehicle_count_total = web_state.get("vehicle_count_total", 0)
        near_miss_count = web_state.get("near_miss_count", 0)
        danger_event_count = web_state.get("danger_event_count", 0)

    if decision == "SAFE TO REVERSE":
        bg = "#00aa44"
        icon = "✓"
    elif decision == "CAUTION":
        bg = "#cc7700"
        icon = "⚠"
    else:
        bg = "#cc0000"
        icon = "✕"

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<title>Driveway Safety</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #111;
    color: #eee;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    padding: 16px;
    max-width: 900px;
    margin: 0 auto;
  }}
  h1 {{
    font-size: 1.1em;
    font-weight: 600;
    color: #888;
    margin-bottom: 12px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
  }}
  .status {{
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 12px;
    background: {bg};
    color: white;
    border-radius: 14px;
    padding: 22px 16px;
    margin-bottom: 16px;
    font-size: 1.8em;
    font-weight: 800;
    letter-spacing: 0.02em;
    text-align: center;
  }}
  .status .icon {{
    font-size: 1.2em;
    line-height: 1;
  }}
  .video-wrap {{
    width: 100%;
    border-radius: 10px;
    overflow: hidden;
    margin-bottom: 16px;
    background: #000;
    border: 1px solid #333;
  }}
  .video-wrap img {{
    width: 100%;
    display: block;
  }}
  .grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-bottom: 8px;
  }}
  .card {{
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 10px;
    padding: 12px 14px;
  }}
  .card-label {{
    font-size: 0.7em;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 4px;
  }}
  .card-value {{
    font-size: 1.4em;
    font-weight: 700;
    color: #eee;
  }}
  .card-value.warn {{ color: #f59e0b; }}
  .card-value.danger {{ color: #ef4444; }}
  .card-value.safe {{ color: #22c55e; }}
  .footer {{
    text-align: center;
    font-size: 0.7em;
    color: #444;
    margin-top: 16px;
    padding-bottom: 16px;
  }}
</style>
</head>
<body>
<h1>Driveway Safety System</h1>
<div class="status">
  <span class="icon">{icon}</span>
  <span>{decision}</span>
</div>
<div class="video-wrap">
  <img id="feed" src="/snapshot"/>
</div>
<div class="grid">
  <div class="card">
    <div class="card-label">Threat Level</div>
    <div id="v-threat" class="card-value {'danger' if threat_level == 'DANGER' else 'safe' if threat_level == 'SAFE' else 'warn'}">{threat_level}</div>
  </div>
  <div class="card">
    <div class="card-label">Risk Score</div>
    <div id="v-risk" class="card-value {'danger' if smoothed_risk >= 8 else 'warn' if smoothed_risk >= 4 else ''}">{smoothed_risk:.1f}</div>
  </div>
  <div class="card">
    <div class="card-label">Objects in Zone</div>
    <div id="v-objects" class="card-value {'warn' if object_count > 0 else ''}">{object_count}</div>
  </div>
  <div class="card">
    <div class="card-label">Approaching</div>
    <div id="v-approaching" class="card-value {'danger' if approaching_count > 0 else ''}">{approaching_count}</div>
  </div>
  <div class="card">
    <div class="card-label">Traffic</div>
    <div id="v-traffic" class="card-value {'danger' if traffic_state == 'HEAVY' else 'warn' if traffic_state == 'MODERATE' else 'safe'}">{traffic_state}</div>
  </div>
  <div class="card">
    <div class="card-label">Occupancy</div>
    <div id="v-occupancy" class="card-value">{occupancy_percent:.1f}%</div>
  </div>
  <div class="card">
    <div class="card-label">Vehicles Today</div>
    <div id="v-vehicles" class="card-value">{vehicle_count_total}</div>
  </div>
  <div class="card">
    <div class="card-label">Danger Events</div>
    <div id="v-events" class="card-value {'warn' if danger_event_count > 0 else ''}">{danger_event_count}</div>
  </div>
</div>
<div class="footer">Live &nbsp;·&nbsp; RidgeAI</div>
<script>
  // Refresh snapshot image every 500ms (works on Safari, Chrome, all mobile browsers)
  function refreshFeed() {{
    const img = document.getElementById('feed');
    img.src = '/snapshot?t=' + Date.now();
  }}
  setInterval(refreshFeed, 500);

  // Refresh stats every 2 seconds via API without reloading the page
  function refreshStats() {{
    fetch('/api/status')
      .then(r => r.json())
      .then(d => {{
        const decision = d.decision || 'SAFE TO REVERSE';
        const bg = decision === 'SAFE TO REVERSE' ? '#00aa44' : decision === 'CAUTION' ? '#cc7700' : '#cc0000';
        const icon = decision === 'SAFE TO REVERSE' ? '✓' : decision === 'CAUTION' ? '⚠' : '✕';
        const statusEl = document.querySelector('.status');
        statusEl.style.background = bg;
        statusEl.innerHTML = '<span class="icon">' + icon + '</span><span>' + decision + '</span>';
        document.getElementById('v-threat').textContent = d.threat_level || 'SAFE';
        document.getElementById('v-risk').textContent = (d.smoothed_risk || 0).toFixed(1);
        document.getElementById('v-objects').textContent = d.object_count || 0;
        document.getElementById('v-approaching').textContent = d.approaching_count || 0;
        document.getElementById('v-traffic').textContent = d.traffic_state || 'LIGHT';
        document.getElementById('v-occupancy').textContent = (d.occupancy_percent || 0).toFixed(1) + '%';
        document.getElementById('v-vehicles').textContent = d.vehicle_count_total || 0;
        document.getElementById('v-events').textContent = d.danger_event_count || 0;
      }})
      .catch(() => {{}});
  }}
  setInterval(refreshStats, 2000);
</script>
</body>
</html>"""


def start_server(host="0.0.0.0", port=5000):
    t = threading.Thread(
        target=lambda: app.run(host=host, port=port, debug=False, use_reloader=False),
        daemon=True
    )
    t.start()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
