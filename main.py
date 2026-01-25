import torch
from transformers import AutoModelForSequenceClassification


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        print("MPS (Metal) backend available. Using GPU.")
        return torch.device("mps")
    print("MPS not available. Using CPU.")
    return torch.device("cpu")


def main() -> None:
    device = get_device()

    model_name = "vectara/hallucination_evaluation_model"
    print(f"Loading model: {model_name}")
    
    # Load model with trust_remote_code=True for custom code
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    model.to(device)

    # Example: Evaluate factual consistency
    # Premise: The source/evidence text
    premise = "The M4 chip is Apple's latest processor, featuring advanced GPU capabilities and Neural Engine."
    
    # Hypothesis: The generated text to evaluate
    hypothesis = "The M4 chip includes a powerful GPU optimized for machine learning tasks."

    # Use the model's predict() method with pairs of (premise, hypothesis)
    pairs = [(premise, hypothesis)]
    
    # The predict() method returns scores between 0 and 1
    # 0 = not evidenced at all, 1 = fully supported
    scores = model.predict(pairs)
    score = scores[0].item()

    print(f"\nPremise: {premise}")
    print(f"Hypothesis: {hypothesis}")
    print(f"\nFactual Consistency Score: {score:.4f}")
    print(f"Interpretation: {'Factually consistent' if score > 0.5 else 'Potential hallucination detected'}")


if __name__ == "__main__":
    main()
