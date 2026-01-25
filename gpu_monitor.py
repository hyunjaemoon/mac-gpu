"""GPU Power Monitoring for Apple Silicon (M4)."""

import subprocess
import time
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, List


@dataclass
class GPUMetrics:
    timestamp: datetime
    gpu_power_watts: Optional[float] = None


class GPUPowerMonitor:
    """Monitor GPU power usage on Apple Silicon."""
    
    def __init__(self):
        self.monitoring = False
        self.metrics_history: List[GPUMetrics] = []
        self.monitor_thread: Optional[threading.Thread] = None
        
    def get_gpu_metrics(self) -> GPUMetrics:
        """Get current GPU metrics. Uses powermetrics; works when run as root (sudo ./run_ui.sh)."""
        metrics = GPUMetrics(timestamp=datetime.now())
        
        cmd = ['powermetrics', '-i', '1000', '-n', '1', '--samplers', 'gpu_power']
        # When running as root (sudo ./run_ui.sh), powermetrics works directly.
        # Otherwise power readings are N/A; samples/duration still work.
        
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0 and result.stdout:
                for line in result.stdout.split('\n'):
                    if 'GPU Power' in line or 'GPU power' in line:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if 'mW' in part or 'W' in part:
                                try:
                                    power = float(parts[i-1].replace(',', ''))
                                    if 'mW' in part:
                                        power /= 1000.0
                                    metrics.gpu_power_watts = power
                                    break
                                except (ValueError, IndexError):
                                    pass
        except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError):
            pass
        
        return metrics
    
    def start_monitoring(self, interval: float = 1.0) -> None:
        """Start monitoring GPU power. Run with sudo ./run_ui.sh for power metrics."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.metrics_history = []
        
        def monitor_loop():
            while self.monitoring:
                self.metrics_history.append(self.get_gpu_metrics())
                time.sleep(interval)
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> None:
        """Stop monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
    
    def get_statistics(self) -> Dict:
        """Get statistics from collected metrics."""
        if not self.metrics_history:
            return {"total_samples": 0}
        
        gpu_powers = [m.gpu_power_watts for m in self.metrics_history if m.gpu_power_watts is not None]
        
        stats = {
            "total_samples": len(self.metrics_history),
            "monitoring_duration_seconds": (
                (self.metrics_history[-1].timestamp - self.metrics_history[0].timestamp).total_seconds()
                if len(self.metrics_history) > 1 else 0
            )
        }
        
        if gpu_powers:
            stats.update({
                "gpu_power_avg_watts": sum(gpu_powers) / len(gpu_powers),
                "gpu_power_min_watts": min(gpu_powers),
                "gpu_power_max_watts": max(gpu_powers),
            })
        
        return stats
