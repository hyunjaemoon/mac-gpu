"""
M4 GPU Demo - Command line interface.

Example usage of HuggingFaceModelRunner for hallucination evaluation.
"""

from model_runner import HuggingFaceModelRunner


def main():
    # Initialize and load model
    runner = HuggingFaceModelRunner()
    
    print(f"Device: {runner.device}")
    print("Loading model...")
    runner.load_model()
    print("Model loaded!\n")
    
    # Verify GPU
    info = runner.verify_device()
    print(f"GPU Available: {info['is_gpu']}")
    print(f"Test Passed: {info['test_passed']}\n")
    
    # Example evaluations
    examples = [
        ("The M4 chip is Apple's latest processor with GPU.", "The M4 has a powerful GPU."),
        ("The capital of France is Berlin.", "The capital of France is Paris."),
        ("I am in California.", "I am in United States."),
    ]
    
    print("Running pairwise evaluations:")
    print("-" * 50)
    
    for sentence1, sentence2 in examples:
        result = runner.evaluate(sentence1, sentence2)
        print(f"Sentence 1: {sentence1}")
        print(f"Sentence 2: {sentence2}")
        print(f"Similarity Score: {result['score']:.4f} - {result['interpretation']}")
        print("-" * 50)


if __name__ == "__main__":
    main()
