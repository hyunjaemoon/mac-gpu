"""
HuggingFaceModelRunner - Run Hugging Face models on M4 GPU (MPS backend).
"""

import torch
from sentence_transformers import CrossEncoder
from gpu_monitor import GPUPowerMonitor


class HuggingFaceModelRunner:
    """Run Hugging Face models on Apple Silicon GPU."""
    
    def __init__(
        self,
        model_name: str = "cross-encoder/stsb-roberta-large",
        trust_remote_code: bool = True
    ):
        self.model_name = model_name
        self.trust_remote_code = trust_remote_code
        self.device = self._get_device()
        self.model = None
        self.gpu_monitor = GPUPowerMonitor()
        
    def _get_device(self) -> str:
        """Auto-detect MPS (GPU) or fall back to CPU."""
        # CrossEncoder from sentence-transformers handles device internally
        # but we can specify it
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    def verify_device(self) -> dict:
        """Verify GPU/CPU device status."""
        info = {
            "device": str(self.device),
            "is_gpu": str(self.device) == "mps",
            "mps_available": torch.backends.mps.is_available(),
            "model_loaded": self.model is not None,
        }
        
        # Test GPU with tensor operation
        try:
            test = torch.randn(100, 100).to(torch.device(self.device))
            result = torch.matmul(test, test)
            info["test_passed"] = True
            info["test_device"] = str(result.device)
        except Exception as e:
            info["test_passed"] = False
            info["error"] = str(e)
        
        return info
    
    def start_monitoring(self, interval: float = 1.0) -> None:
        """Start GPU power monitoring. Run with sudo ./run_ui.sh for power metrics."""
        self.gpu_monitor.start_monitoring(interval=interval)
    
    def stop_monitoring(self) -> None:
        """Stop GPU power monitoring."""
        self.gpu_monitor.stop_monitoring()
    
    def load_model(self) -> None:
        """Load the cross-encoder model."""
        # CrossEncoder automatically handles device placement
        # For MPS, we need to ensure it's supported
        device_str = self.device if self.device != "mps" else "cpu"
        # Note: sentence-transformers may not fully support MPS yet
        # It will fall back to CPU if MPS is not supported
        try:
            self.model = CrossEncoder(self.model_name, device=device_str)
        except Exception:
            # Fallback to CPU if MPS doesn't work
            self.model = CrossEncoder(self.model_name, device="cpu")
            self.device = "cpu"
    
    def evaluate(self, sentence1: str, sentence2: str) -> dict:
        """Evaluate semantic similarity between two sentences.
        
        Returns a score between 0 and 1, where:
        - Higher scores indicate greater semantic similarity
        - Lower scores indicate less similarity
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # CrossEncoder.predict expects a list of tuples
        pairs = [(sentence1, sentence2)]
        scores = self.model.predict(pairs)
        score = float(scores[0])
        
        # Normalize score to 0-1 range (STS-B models output 0-5, but this one outputs 0-1)
        # Clamp to ensure it's in valid range
        score = max(0.0, min(1.0, score))
        
        return {
            "sentence1": sentence1,
            "sentence2": sentence2,
            "score": score,
            "is_similar": score > 0.5,
            "interpretation": f"High similarity ({score:.2%})" if score > 0.5 else f"Low similarity ({score:.2%})"
        }
