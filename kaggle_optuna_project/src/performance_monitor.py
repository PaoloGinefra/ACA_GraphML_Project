"""
Performance monitoring utilities for tracking system resources and model performance.
"""

import time
import psutil
import torch
import gc
from typing import Dict, Any, Optional
import threading
from dataclasses import dataclass
from collections import defaultdict, deque


@dataclass
class ResourceSnapshot:
    """Snapshot of system resources at a point in time."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    gpu_memory_used_gb: Optional[float] = None
    gpu_utilization: Optional[float] = None


class PerformanceMonitor:
    """
    Real-time performance monitoring for system resources and model metrics.
    
    This class provides comprehensive monitoring capabilities for:
    - CPU and memory usage
    - GPU utilization and memory
    - Training metrics over time
    - Performance bottleneck detection
    """
    
    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self.is_monitoring = False
        self.monitor_thread = None
        
        # Data storage
        self.resource_history = deque(maxlen=1000)
        self.metrics_history = defaultdict(list)
        
        # Peak tracking
        self.peak_cpu = 0.0
        self.peak_memory = 0.0
        self.peak_gpu_memory = 0.0
        
    def start_monitoring(self):
        """Start background monitoring of system resources."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop background monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
    
    def _monitor_loop(self):
        """Main monitoring loop running in background thread."""
        while self.is_monitoring:
            try:
                snapshot = self._take_snapshot()
                self.resource_history.append(snapshot)
                
                # Update peaks
                self.peak_cpu = max(self.peak_cpu, snapshot.cpu_percent)
                self.peak_memory = max(self.peak_memory, snapshot.memory_used_gb)
                if snapshot.gpu_memory_used_gb:
                    self.peak_gpu_memory = max(self.peak_gpu_memory, snapshot.gpu_memory_used_gb)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(self.monitoring_interval)
    
    def _take_snapshot(self) -> ResourceSnapshot:
        """Take a snapshot of current system resources."""
        # CPU and system memory
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        memory_used_gb = memory.used / (1024 ** 3)
        
        # GPU metrics
        gpu_memory_used_gb = None
        gpu_utilization = None
        
        if torch.cuda.is_available():
            try:
                gpu_memory_used_gb = torch.cuda.memory_allocated() / (1024 ** 3)
                # GPU utilization requires nvidia-ml-py3 which might not be available
                # gpu_utilization = self._get_gpu_utilization()
            except Exception:
                pass
        
        return ResourceSnapshot(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_gb=memory_used_gb,
            gpu_memory_used_gb=gpu_memory_used_gb,
            gpu_utilization=gpu_utilization
        )
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """Log a training metric."""
        timestamp = time.time()
        self.metrics_history[name].append({
            'value': value,
            'timestamp': timestamp,
            'step': step
        })
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """Get summary of resource usage."""
        if not self.resource_history:
            return {}
        
        recent_snapshots = list(self.resource_history)[-10:]  # Last 10 snapshots
        
        current = recent_snapshots[-1] if recent_snapshots else None
        avg_cpu = sum(s.cpu_percent for s in recent_snapshots) / len(recent_snapshots)
        avg_memory = sum(s.memory_used_gb for s in recent_snapshots) / len(recent_snapshots)
        
        summary = {
            'current_cpu_percent': current.cpu_percent if current else 0,
            'current_memory_gb': current.memory_used_gb if current else 0,
            'current_gpu_memory_gb': current.gpu_memory_used_gb if current else 0,
            'avg_cpu_percent': avg_cpu,
            'avg_memory_gb': avg_memory,
            'peak_cpu_percent': self.peak_cpu,
            'peak_memory_gb': self.peak_memory,
            'peak_gpu_memory_gb': self.peak_gpu_memory,
            'total_snapshots': len(self.resource_history)
        }
        
        return summary
    
    def get_memory_usage_gb(self) -> float:
        """Get current total memory usage in GB."""
        memory_gb = psutil.virtual_memory().used / (1024 ** 3)
        
        if torch.cuda.is_available():
            try:
                gpu_memory_gb = torch.cuda.memory_allocated() / (1024 ** 3)
                memory_gb += gpu_memory_gb
            except Exception:
                pass
        
        return memory_gb
    
    def cleanup_memory(self):
        """Force garbage collection and clear GPU cache."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def __enter__(self):
        """Context manager entry."""
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_monitoring()


class TimingContext:
    """Context manager for timing operations."""
    
    def __init__(self, name: str, monitor: Optional[PerformanceMonitor] = None):
        self.name = name
        self.monitor = monitor
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        
        if self.monitor:
            self.monitor.log_metric(f"timing_{self.name}", duration)
        
        print(f"{self.name} completed in {duration:.2f} seconds")
    
    @property
    def duration(self) -> Optional[float]:
        """Get duration in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


def profile_memory_usage(func):
    """Decorator to profile memory usage of a function."""
    def wrapper(*args, **kwargs):
        import tracemalloc
        
        tracemalloc.start()
        result = func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        print(f"Function {func.__name__}:")
        print(f"  Current memory: {current / 1024 / 1024:.2f} MB")
        print(f"  Peak memory: {peak / 1024 / 1024:.2f} MB")
        
        return result
    
    return wrapper
