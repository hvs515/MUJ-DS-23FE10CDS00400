# Political Bias Detector
In a world where Bias is probably the most important meteric when making any decision ,Political Bias shapes how you think about the future of your country .
This project is a  comprehensive classical machine learning system to classify news articles into 5 political bias categories: **Far-Left**, **Left**, **Center**, **Right**, and **Far-Right**.

## Project Overview

This project uses classical machine learning (no transformers) with sophisticated feature engineering to detect political bias in text. It combines:

- **TF-IDF features** for vocabulary analysis
- **Political lexicon features** for biased language detection  
- **Sentiment analysis** using VADER
- **Stylistic features** (punctuation, pronoun usage, hedging)
- **Named entity recognition** to identify politically-aligned entities
- **Readability metrics** for text complexity analysis

The system trains 5 complementary models and uses a soft voting ensemble for final predictions.

## Features

### 1. Feature Engineering (6 types)

#### TF-IDF Features
- Unigrams and bigrams
- 5,000 maximum features with sublinear TF scaling
- Min document frequency: 2, Max: 0.85

#### Political Lexicon Features
- **Left lexicon**: progressive, systemic, marginalized, equity, climate crisis, etc.
- **Right lexicon**: illegal alien, radical left, deep state, free market, etc.
- **Loaded words**: regime, radical, extreme, dangerous, crisis, invasion, etc.
- Outputs: counts, ratios, net lean score

#### Sentiment Features
- VADER sentiment polarity (positive, negative, neutral, compound)
- Sentence-level sentiment variance
- Sentiment range across article

#### Stylistic Features
- Average sentence and word length
- Hedge word ratio (allegedly, reportedly, suggests)
- Certainty word ratio (definitely, clearly, obviously)
- First/third person pronoun ratios
- Punctuation ratios (exclamation, question, quotes)
- Certainty-to-hedge ratio

#### Named Entity Features  
- Person, Organization, Location entity counts (normalized)
- Left/Right-leaning entity hits (Biden, Trump, CNN, Fox News, etc.)
- Entity lean score

#### Readability Features
- Flesch Reading Ease
- Flesch-Kincaid Grade Level
- Gunning Fog Index
- SMOG Index
- Automated Readability Index

### 2. Classification Models

1. **Logistic Regression** - Fast, interpretable baseline
2. **LinearSVC + Calibration** - SVM with probability calibration
3. **Multinomial Naive Bayes** - Probabilistic text classifier
4. **XGBoost** - Gradient boosted tree ensemble
5. **Soft Voting Ensemble** - Weighted combination (weights: 3, 4, 1, 3)

## Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

1. Clone/navigate to the project directory:
```bash
cd political-bias-detector
```

2. Create and activate virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/Scripts/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download required spacy model:
```bash
python -m spacy download en_core_web_sm
```

## Usage

### Training

Train all 5 models on the dataset:

```bash
python src/train.py
```

This will:
- Generate synthetic dataset (if no real dataset available) with 500 articles
- Extract comprehensive features from all articles
- Train all 5 models
- Save trained models to `models/` directory

**Output:**
- `models/logistic_regression.pkl`
- `models/linear_svc.pkl`
- `models/multinomial_nb.pkl`
- `models/xgboost.pkl`
- `models/voting_ensemble.pkl`
- `models/feature_extractor.pkl`
- `models/label_encoder.pkl`

### Evaluation

Run stratified 5-fold cross-validation on all models:

```bash
python src/evaluate.py
```

This will:
- Perform 5-fold cross-validation on each model
- Generate confusion matrix for the ensemble (saved as `models/confusion_matrix.png`)
- Print top 20 predictive TF-IDF features per class
- Compare all models in a summary table
- Print detailed classification reports

### Prediction

Classify individual articles:

#### Using command line with text:
```bash
python src/predict.py --text "Your article text here"
```

#### Using command line with file:
```bash
python src/predict.py --file path/to/article.txt
```

#### Optional arguments:
```bash
python src/predict.py --text "..." --threshold 0.6 --model-dir models
```

**Example Output:**
```json
{
  "prediction": "RIGHT",
  "confidence": 0.81,
  "uncertain": false,
  "breakdown": {
    "FAR_LEFT": 0.02,
    "LEFT": 0.07,
    "CENTER": 0.10,
    "RIGHT": 0.81,
    "FAR_RIGHT": 0.00
  },
  "warning": null
}
```

#### LLM second opinion (Google Gemini):
Compare the ML ensemble with an LLM classification. Create a `.env` file in the project root containing your Gemini API key:
```
GEM_API=your-gemini-api-key
```
Then add `--llm` to any prediction, or run the LLM on its own:
```bash
python src/predict.py --llm --text "Your article text here"
python src/llm_predict.py --file path/to/article.txt
```
The result gains an `"llm"` field with Gemini's `prediction`, `confidence` and `explanation`. If a Gemini model is overloaded, the next model in `MODELS` (see `src/llm_predict.py`) is tried.

#### Programmatic usage:
```python
from src.predict import BiasPredictor

predictor = BiasPredictor()
result = predictor.predict("article text here", confidence_threshold=0.5)
print(result)
```

### Testing

Run unit tests to verify feature extraction and prediction:

```bash
pytest tests/test_features.py -v
```

Tests cover:
- TF-IDF feature extraction
- Lexicon feature detection
- Sentiment feature ranges
- Stylistic feature extraction
- Named entity features
- Readability metrics
- Prediction output structure
- Confidence scoring
- Probability calibration

## Project Structure

```
political-bias-detector/
├── data/
│   └── raw/
│       └── articles.csv           # Dataset (auto-generated if missing)
├── src/
│   ├── preprocess.py              # Dataset generation/loading
│   ├── features.py                # Feature extraction (6 types)
│   ├── train.py                   # Model training (5 models)
│   ├── evaluate.py                # Evaluation & cross-validation
│   └── predict.py                 # Prediction & CLI interface
├── models/
│   ├── logistic_regression.pkl
│   ├── linear_svc.pkl
│   ├── multinomial_nb.pkl
│   ├── xgboost.pkl
│   ├── voting_ensemble.pkl
│   ├── feature_extractor.pkl
│   ├── label_encoder.pkl
│   └── confusion_matrix.png
├── notebooks/
│   └── exploration.ipynb          # Exploratory analysis
├── tests/
│   └── test_features.py           # Unit tests
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Feature Engineering Details

### Feature Combination

All features are combined using scipy's sparse hstack:
1. **TF-IDF features** (5,000 sparse features) - kept sparse for efficiency
2. **Hand-crafted features** (35 dense features) - scaled with MaxAbsScaler
3. **Total: ~5,035 features per article**

### Feature Scaling

- TF-IDF features: Built-in normalization via TfidfVectorizer
- Hand-crafted features: MaxAbsScaler for scaling to [-1, 1] range
- Preserves sparsity of TF-IDF features

## Model Architecture

### Individual Models

| Model | Type | Key Parameters | Use Case |
|-------|------|-----------------|----------|
| Logistic Regression | Linear | C=1.0, balanced | Fast baseline, interpretable |
| LinearSVC | SVM | class_weight='balanced' | Margin-based classification |
| Multinomial NB | Probabilistic | alpha=0.1 | TF-IDF specialized |
| XGBoost | Tree Ensemble | 200 trees, lr=0.1 | Complex feature interactions |
| Ensemble | Meta | Soft voting, weights: 3,4,1,3 | Combines all strengths |

### Ensemble Strategy

Uses soft voting with weighted predictions:
- Logistic Regression: weight 3
- LinearSVC: weight 4 (most confident)
- MultinomialNB: weight 1 (specialized view)
- XGBoost: weight 3

Soft voting averages probability distributions, better calibrated than hard voting.

## Performance Metrics

### Evaluation Methodology
- **Stratified 5-fold cross-validation** maintains class balance in each fold
- **Accuracy**: Overall correctness
- **F1-macro**: Average F1 score across all classes (handles imbalance)
- **Per-class F1**: Detailed performance on each bias category

### Confidence Threshold
- Default threshold: 0.5
- If max predicted probability < threshold, prediction marked as UNCERTAIN
- Improves precision by avoiding low-confidence predictions

### Short Article Warning
- Articles < 100 words trigger warning
- Indicates prediction may be less reliable due to limited text

## Example Usage Walkthrough

```python
# 1. Train models
python src/train.py

# 2. Evaluate on test data
python src/evaluate.py

# 3. Run tests
pytest tests/ -v

# 4. Predict on new article
python src/predict.py --text "The government must secure our borders \
immediately and stop the invasion of illegal aliens destroying our great nation."
```

**Expected output for right-leaning text:**
```json
{
  "prediction": "RIGHT",
  "confidence": 0.75,
  "uncertain": false,
  "breakdown": {
    "FAR_LEFT": 0.01,
    "LEFT": 0.05,
    "CENTER": 0.08,
    "RIGHT": 0.75,
    "FAR_RIGHT": 0.11
  },
  "warning": null
}
```

## Dataset

### Synthetic Dataset (Default)
- 500 articles total (100 per class)
- Auto-generated with specific vocabulary patterns:
  - FAR_LEFT: systemic racism, climate crisis, equity, reproductive rights
  - LEFT: progressive, regulation, healthcare, workers rights
  - CENTER: balanced language, policy-focused, factual reporting
  - RIGHT: free market, border security, big government, law and order
  - FAR_RIGHT: invasion, regime, destroy, patriot, extreme language

### Real Dataset (Optional)
Replace `data/raw/articles.csv` with real data containing:
- `text`: Article content
- `bias`: Label (FAR_LEFT, LEFT, CENTER, RIGHT, FAR_RIGHT)

## Troubleshooting

### Model files not found error
```
FileNotFoundError: [models/voting_ensemble.pkl] not found
```
**Solution:** Run `python src/train.py` to train models first

### Spacy model error
```
OSError: [E050] Can't find model "en_core_web_sm"
```
**Solution:** Run `python -m spacy download en_core_web_sm`

### Out of memory with large datasets
- Reduce max_features in TfidfVectorizer (in features.py)
- Process articles in batches
- Consider using a server with more RAM

### Predictions too uncertain
- Lower confidence_threshold (e.g., `--threshold 0.3`)
- Train on more diverse data
- Check article length (< 100 words is flagged)

## Advanced Usage

### Custom confidence threshold:
```bash
python src/predict.py --text "..." --threshold 0.6
```

### Make predictions programmatically:
```python
from src.predict import BiasPredictor

predictor = BiasPredictor(model_dir='models')
result = predictor.predict("Your article text", confidence_threshold=0.5)
print(result['prediction'])
print(result['breakdown'])
```

### Remove uncertain predictions:
```python
result = predictor.predict(text)
if not result['uncertain']:
    print(f"Confident prediction: {result['prediction']}")
```

### Access individual model predictions:
```python
from train import ModelTrainer

trainer = ModelTrainer()
trainer.load_models()

# Get each model's prediction
for name, model in trainer.models.items():
    pred = model.predict(features)
    print(f"{name}: {pred}")
```

## Citation

If you use this project, please cite:

```
@software{political_bias_detector,
  title={Political Bias Detector},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/political-bias-detector}
}
```

## License

MIT License - feel free to use for research and commercial projects

## Contributing

Contributions welcome! Areas for improvement:
- Real dataset integration (using AllSides or similar)
- More sophisticated feature engineering
- Deep learning experiments (though keeping classical ML as baseline)
- Multi-language support
- Real-time prediction API (FastAPI/Flask)

---

**Created:** 2024  
**Python:** 3.8+  
**ML Framework:** scikit-learn, XGBoost (Classical ML only)
