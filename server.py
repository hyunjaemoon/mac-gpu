"""
M4 GPU Demo - Server using HTMX + SSE
"""

import os

from flask import Flask, request, Response, send_from_directory
import threading
import time
from model_runner import HuggingFaceModelRunner

app = Flask(__name__)
ROOT = os.path.dirname(os.path.abspath(__file__))

runner = None
model_loaded = False
status_log = ["Ready. Click 'Load Model' to begin."]


def generate_gpu_stats():
    """Generator that yields GPU stats as SSE events."""
    while True:
        if runner is None or not model_loaded:
            html = """
            <div class="gpu-status">
                <span class="dot"></span>
                <span>Load model to start monitoring</span>
            </div>
            <div class="gpu-dashboard">
                <div class="metric"><div class="label">Samples</div><div class="value">—</div></div>
                <div class="metric"><div class="label">Duration</div><div class="value">—</div></div>
                <div class="metric"><div class="label">Avg Power</div><div class="value">—</div></div>
                <div class="metric"><div class="label">Min Power</div><div class="value">—</div></div>
                <div class="metric"><div class="label">Max Power</div><div class="value">—</div></div>
            </div>
            """
        else:
            stats = runner.gpu_monitor.get_statistics()
            monitoring = runner.gpu_monitor.monitoring
            
            samples = stats.get('total_samples', 0)
            duration = stats.get('monitoring_duration_seconds', 0)
            avg = stats.get('gpu_power_avg_watts')
            mn = stats.get('gpu_power_min_watts')
            mx = stats.get('gpu_power_max_watts')
            
            status_class = "gpu-status active" if monitoring else "gpu-status"
            dot_class = "dot live" if monitoring else "dot"
            status_text = f"Monitoring... {samples} samples" if monitoring else "Idle"
            
            html = f"""
            <div class="{status_class}">
                <span class="{dot_class}"></span>
                <span>{status_text}</span>
            </div>
            <div class="gpu-dashboard">
                <div class="metric"><div class="label">Samples</div><div class="value">{samples}</div></div>
                <div class="metric"><div class="label">Duration</div><div class="value">{f'{duration:.1f}s' if duration > 0 else '0s'}</div></div>
                <div class="metric"><div class="label">Avg Power</div><div class="value">{f'{avg:.2f}W' if avg else 'N/A'}</div></div>
                <div class="metric"><div class="label">Min Power</div><div class="value">{f'{mn:.2f}W' if mn else 'N/A'}</div></div>
                <div class="metric"><div class="label">Max Power</div><div class="value">{f'{mx:.2f}W' if mx else 'N/A'}</div></div>
            </div>
            """
        
        # SSE format: data lines followed by blank line
        data = html.replace('\n', ' ').strip()
        yield f"data: {data}\n\n"
        time.sleep(1)


@app.route('/')
def index():
    return send_from_directory(ROOT, 'index.html')


@app.route('/gpu-stream')
def gpu_stream():
    """SSE endpoint for GPU stats."""
    return Response(
        generate_gpu_stats(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        }
    )


@app.route('/load', methods=['POST'])
def load_model():
    global runner, model_loaded, status_log
    
    model_name = request.form.get('model_name', 'cross-encoder/stsb-roberta-large')
    
    try:
        runner = HuggingFaceModelRunner(model_name=model_name)
        runner.start_monitoring()
        runner.load_model()
        model_loaded = True
        msg = f"✓ Model loaded on {runner.device}"
    except Exception as e:
        msg = f"✗ Error: {e}"
    
    status_log.append(msg)
    return "<br>".join(status_log[-10:])


@app.route('/verify', methods=['GET'])
def verify_device():
    global runner, status_log
    
    if runner is None:
        msg = "✗ Load model first"
    else:
        info = runner.verify_device()
        msg = f"Device: {info['device']} | GPU: {info['is_gpu']} | Test: {info['test_passed']}"
    
    status_log.append(msg)
    return "<br>".join(status_log[-10:])


@app.route('/evaluate', methods=['POST'])
def evaluate():
    global runner, model_loaded, status_log
    
    if not model_loaded or runner is None:
        status_log.append("✗ Load model first")
        return '<div class="results"><p>✗ Load model first</p></div>'
    
    sentence1 = request.form.get('sentence1', '').strip()
    sentence2 = request.form.get('sentence2', '').strip()
    
    if not sentence1 or not sentence2:
        return '<div class="results"><p>✗ Enter both sentences</p></div>'
    
    try:
        result = runner.evaluate(sentence1, sentence2)
        score = result['score']
        interp = result['interpretation']
        
        score_class = "high" if score > 0.5 else "low"
        interp_class = "good" if result['is_similar'] else "bad"
        
        status_log.append(f"✓ Similarity Score: {score:.4f}")
        
        return f"""
        <div class="results">
            <p><strong>Sentence 1:</strong> {sentence1}</p>
            <p><strong>Sentence 2:</strong> {sentence2}</p>
            <div class="score {score_class}">Similarity Score: {score:.4f}</div>
            <div class="interp {interp_class}">{interp}</div>
        </div>
        """
    except Exception as e:
        status_log.append(f"✗ Error: {e}")
        return f'<div class="results"><p>✗ Error: {e}</p></div>'


def main():
    import webbrowser

    PORT = 8000  # Avoid 5000 – macOS AirPlay Receiver uses it and can cause 403
    print("=" * 50)
    print("M4 GPU Demo")
    print("=" * 50)
    print(f"Server: http://127.0.0.1:{PORT}")
    print("Press Ctrl+C to stop")
    if getattr(os, "geteuid", lambda: -1)() != 0:
        print("Tip: run 'sudo ./run_ui.sh' for power metrics (Avg/Min/Max W)")
    print("=" * 50)
    
    def open_browser():
        time.sleep(1.0)
        webbrowser.open(f'http://127.0.0.1:{PORT}')
    
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Use threaded=True for SSE support; port 8000 avoids AirPlay on 5000 (403)
    app.run(host='127.0.0.1', port=PORT, threaded=True)


if __name__ == '__main__':
    main()
