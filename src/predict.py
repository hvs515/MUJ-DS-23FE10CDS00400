"""
Prediction module for political bias detection.
Includes CLI interface for making predictions on new articles.
"""

import json
import argparse
import joblib
from pathlib import Path

import numpy as np
import pandas as pd


class BiasPredictor:
    """Makes predictions on new articles."""
    
    def __init__(self, model_dir='models'):
        self.model_dir = Path(model_dir)
        self.feature_extractor = None
        self.model = None
        self.label_encoder = None
        self.idx_to_label = None
        self._load_models()
    
    def _load_models(self):
        """Load trained models and feature extractor."""
        print("Loading trained models...")
        
        # Load feature extractor
        path = self.model_dir / "feature_extractor.pkl"
        self.feature_extractor = joblib.load(path)
        
        # Load label encoder
        path = self.model_dir / "label_encoder.pkl"
        self.label_encoder = joblib.load(path)
        self.idx_to_label = {v: k for k, v in self.label_encoder.items()}
        
        # Load ensemble model
        path = self.model_dir / "voting_ensemble.pkl"
        self.model = joblib.load(path)
        
        print("Models loaded successfully!")
    
    def predict(self, text, confidence_threshold=0.5):
        """
        Predict political bias for a given article text.
        
        Args:
            text: article text string
            confidence_threshold: if max probability below this, return UNCERTAIN
            
        Returns:
            dict with prediction results
        """
        # Check article length
        word_count = len(text.split())
        warning = None
        
        if word_count < 100:
            warning = f"Article is short ({word_count} words). Prediction may be less reliable."
        
        # Extract features
        features, _ = self.feature_extractor.extract_all_features([text], fit_tfidf=False)
        
        # Get probabilities
        probabilities = self.model.predict_proba(features)[0]
        
        # Get prediction
        predicted_idx = np.argmax(probabilities)
        predicted_label = self.idx_to_label[predicted_idx]
        confidence = probabilities[predicted_idx]
        
        # Check confidence threshold
        uncertain = confidence < confidence_threshold
        if uncertain:
            predicted_label = "UNCERTAIN"
        
        # Build probability breakdown
        breakdown = {}
        for idx, label in self.idx_to_label.items():
            breakdown[label] = float(probabilities[idx])
        
        result = {
            "prediction": predicted_label,
            "confidence": float(confidence),
            "uncertain": bool(uncertain),
            "breakdown": breakdown,
            "warning": warning
        }
        
        return result
    
    def predict_from_file(self, file_path, confidence_threshold=0.5):
        """
        Predict bias for article from a file.
        
        Args:
            file_path: path to text file
            confidence_threshold: confidence threshold
            
        Returns:
            dict with prediction results
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        return self.predict(text, confidence_threshold)


def main():
    """CLI interface."""
    parser = argparse.ArgumentParser(
        description='Political Bias Detector - Classifies news articles as Far-Left, Left, Center, Right, or Far-Right'
    )
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--text', type=str, help='Article text to classify')
    input_group.add_argument('--file', type=str, help='Path to article text file')
    
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Confidence threshold for prediction (default: 0.5)')
    
    parser.add_argument('--model-dir', type=str, default='models',
                       help='Path to models directory (default: models)')
    
    parser.add_argument('--llm', action='store_true',
                       help='Also classify with Gemini (needs GEM_API in .env)')
    
    args = parser.parse_args()
    
    # Initialize predictor
    print("Initializing predictor...")
    predictor = BiasPredictor(model_dir=args.model_dir)
    print()
    
    # Make prediction
    if args.text:
        print(f"Article text: {args.text[:100]}...")
        print()
        result = predictor.predict(args.text, confidence_threshold=args.threshold)
    else:
        print(f"Reading from file: {args.file}")
        print()
        result = predictor.predict_from_file(args.file, confidence_threshold=args.threshold)
    
    # Optional LLM second opinion
    if args.llm:
        from llm_predict import predict_with_llm
        if args.file:
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            text = args.text
        result["llm"] = predict_with_llm(text)
    
    # Pretty print result
    print("="*60)
    print("PREDICTION RESULT")
    print("="*60)
    print(json.dumps(result, indent=2))
    print("="*60)
    
    return result


if __name__ == "__main__":
    main()
