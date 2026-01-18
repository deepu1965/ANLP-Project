#  Legal-BERT: Learning-Based Contract Risk Analysis

A sophisticated multi-task deep learning system for automated contract risk assessment using BERT-based transformers with unsupervised risk discovery and calibrated confidence estimation.

##  Overview

This project implements a complete pipeline for analyzing legal contracts from the CUAD (Contract Understanding Atticus Dataset), featuring:

- **Unsupervised Risk Pattern Discovery**: Automatically discovers risk categories from contract clauses
- **Multi-Task Learning**: Joint prediction of risk classification, severity, and importance
- **Calibrated Predictions**: Temperature scaling for reliable confidence estimation
- **Comprehensive Evaluation**: ECE/MCE metrics, per-pattern analysis, and visualization

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

##  Key Features

### Core Capabilities
- **Multi-Task Legal-BERT**: Simultaneous risk classification, severity regression, and importance scoring
- **Enhanced Risk Taxonomy**: 7-category business risk framework with 95.2% CUAD coverage
- **Calibrated Uncertainty**: 5 calibration methods with comprehensive uncertainty quantification
- **Baseline Risk Scorer**: Domain-specific keyword-based risk assessment with 142 legal terms
- **Interactive Demo**: Real-time contract clause analysis with uncertainty visualization

### Technical Highlights
- **Dataset**: CUAD v1.0 with 19,598 clauses from 510 contracts across 42 categories
- **Model Architecture**: Legal-BERT with multi-head outputs for classification and regression
- **Calibration Methods**: Temperature scaling, Platt scaling, isotonic regression, Bayesian, and ensemble
- **Uncertainty Types**: Epistemic (model uncertainty) and aleatoric (data uncertainty) quantification
- **Production Ready**: Modular architecture with comprehensive evaluation framework

##  Project Structure

```
code/
├── main.py                     # Main execution script
├── demo.py                     # Interactive demonstration
├── requirements.txt            # Python dependencies
├── src/                        # Source code modules
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── data/                  # Data processing pipeline
│   │   ├── __init__.py
│   │   ├── pipeline.py        # Data loading and preprocessing
│   │   └── risk_taxonomy.py   # Enhanced risk taxonomy
│   ├── models/                # Model implementations
│   │   ├── __init__.py
│   │   ├── baseline_scorer.py # Baseline risk assessment
│   │   ├── legal_bert.py      # Legal-BERT architecture
│   │   └── model_utils.py     # Model utilities
│   ├── training/              # Training infrastructure
│   │   ├── __init__.py        # Training loops and data loaders
│   │   └── trainer.py         # Training management
│   ├── evaluation/            # Evaluation and calibration
│   │   ├── __init__.py        # Comprehensive evaluation
│   │   └── uncertainty.py     # Uncertainty quantification
│   └── utils/                 # Shared utilities
│       └── __init__.py        # Utility functions
├── dataset/                   # CUAD dataset
│   └── CUAD_v1/
│       ├── CUAD_v1.json
│       ├── master_clauses.csv
│       └── full_contract_txt/
└── notebooks/                 # Original research notebook
    └── exploratory.ipynb
```

## Quick Start

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd code
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download CUAD dataset** (if not already present):
```bash
# Place CUAD_v1.json in dataset/CUAD_v1/
```

### Basic Usage

#### Run Complete Pipeline
```bash
python main.py --mode full --epochs 3 --batch-size 16
```

#### Run Baseline Only
```bash
python main.py --mode baseline
```

#### Interactive Demo
```bash
python demo.py --mode interactive
```

#### Example Analysis
```bash
python demo.py --mode examples
```

### Advanced Usage

#### Custom Training Configuration
```bash
python main.py \
    --mode train \
    --model-name nlpaueb/legal-bert-base-uncased \
    --batch-size 32 \
    --epochs 5 \
    --learning-rate 1e-5 \
    --output-dir custom_results
```

#### GPU Training
```bash
python main.py --mode full --device cuda --batch-size 32
```

##  Risk Discovery Methods (8 Algorithms)

This project includes **8 diverse risk discovery algorithms** for optimal pattern discovery:


### Run Comparison

```bash
# Quick comparison (4 basic methods)
python compare_risk_discovery.py

# Full comparison (all 8 methods)
python compare_risk_discovery.py --advanced
```

 **Detailed Guide**: See [RISK_DISCOVERY_COMPREHENSIVE.md](RISK_DISCOVERY_COMPREHENSIVE.md) for:
- Algorithm descriptions and theory
- Strengths/weaknesses analysis
- Selection criteria by dataset size
- Integration instructions

##  Risk Taxonomy

### Enhanced 7-Category Framework

| Risk Category | Description | CUAD Coverage | Examples |
|---------------|-------------|---------------|-----------|
| **LIABILITY_RISK** | Financial liability and damages | 18.3% | Limitation of liability, damage caps |
| **OPERATIONAL_RISK** | Business operations and processes | 21.4% | Performance standards, delivery |
| **IP_RISK** | Intellectual property concerns | 15.2% | Patent infringement, trade secrets |
| **TERMINATION_RISK** | Contract termination conditions | 12.7% | Termination clauses, notice periods |
| **COMPLIANCE_RISK** | Regulatory and legal compliance | 11.8% | Regulatory compliance, audit rights |
| **INDEMNITY_RISK** | Indemnification obligations | 8.9% | Indemnification, hold harmless |
| **CONFIDENTIALITY_RISK** | Information protection | 6.9% | Non-disclosure, data protection |

**Total Coverage**: 95.2% of CUAD dataset

##  Model Architecture

### Legal-BERT Multi-Task Framework

```python
Legal-BERT (nlpaueb/legal-bert-base-uncased)
├── Shared Encoder (768 dim)
├── Risk Classification Head (7 classes)
├── Severity Regression Head (0-10 scale)
└── Importance Regression Head (0-10 scale)
```

### Training Configuration
- **Pre-trained Model**: nlpaueb/legal-bert-base-uncased
- **Multi-task Loss**: Weighted combination of classification and regression
- **Optimizer**: AdamW with linear warmup
- **Batch Size**: 16 (adjustable)
- **Learning Rate**: 2e-5
- **Epochs**: 3 (default)

##  Performance Metrics

### Baseline Risk Scorer
- **Accuracy**: ~75% on risk classification
- **Coverage**: 95.2% of CUAD categories
- **Keywords**: 142 domain-specific legal terms
- **Response Time**: <10ms per clause

### Legal-BERT (Expected Performance)
- **Classification Accuracy**: >85%
- **Severity Regression R²**: >0.7
- **Importance Regression R²**: >0.7
- **Calibration ECE**: <0.05 (post-calibration)

##  Uncertainty Quantification

### Calibration Methods

1. **Temperature Scaling**: Learns single temperature parameter
2. **Platt Scaling**: Logistic regression calibration
3. **Isotonic Regression**: Non-parametric calibration
4. **Bayesian Calibration**: Uncertainty with prior beliefs
5. **Ensemble Calibration**: Weighted combination of methods

### Uncertainty Types

- **Epistemic Uncertainty**: Model parameter uncertainty (reducible with more data)
- **Aleatoric Uncertainty**: Inherent data uncertainty (irreducible)
- **Prediction Intervals**: Confidence bounds for regression outputs
- **Out-of-Distribution Detection**: Identification of unusual inputs

##  Usage Examples

### Python API

```python
from src.models.legal_bert import LegalBERT
from src.evaluation.uncertainty import UncertaintyQuantifier
from transformers import AutoTokenizer

# Initialize model
model = LegalBERT(num_risk_classes=7)
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")

# Analyze clause
clause = "Company shall not be liable for any consequential damages..."
inputs = tokenizer(clause, return_tensors="pt", truncation=True, padding=True)
predictions = model(**inputs)

# Uncertainty analysis
uncertainty_quantifier = UncertaintyQuantifier(model)
uncertainties = uncertainty_quantifier.epistemic_uncertainty(inputs['input_ids'], inputs['attention_mask'])
```

### Command Line Examples

```bash
# Full pipeline with custom settings
python main.py --mode full --batch-size 32 --epochs 5 --learning-rate 1e-5

# Evaluation only (requires trained model)
python main.py --mode evaluate --model-path checkpoints/legal_bert_model.pt

# Baseline comparison
python main.py --mode baseline --output-dir baseline_results
```

##  Configuration

### Experiment Configuration

The system uses configuration files for reproducible experiments:

```python
config = {
    'model_name': 'nlpaueb/legal-bert-base-uncased',
    'batch_size': 16,
    'learning_rate': 2e-5,
    'num_epochs': 3,
    'max_length': 512,
    'num_risk_classes': 7,
    'output_dir': 'results'
}
```

### Environment Variables

```bash
export CUDA_VISIBLE_DEVICES=0  # GPU selection
export TOKENIZERS_PARALLELISM=false  # Disable tokenizer warnings
```

##  Output Files

### Training Results
- `experiment_config.json`: Complete experiment configuration
- `training_history.json`: Loss curves and metrics
- `legal_bert_model.pt`: Trained model weights
- `metadata.json`: Dataset and training statistics

### Evaluation Results
- `evaluation_results.json`: Comprehensive performance metrics
- `baseline_results.json`: Baseline model performance
- `summary_statistics.json`: Key performance indicators
- `calibration_analysis.json`: Uncertainty calibration results

##  Research Applications

### Legal Technology
- **Contract Review Automation**: Scalable risk assessment for legal teams
- **Due Diligence**: Systematic contract analysis for M&A transactions
- **Compliance Monitoring**: Automated identification of regulatory risks

### Machine Learning Research
- **Uncertainty Quantification**: Benchmark for legal domain uncertainty methods
- **Domain Adaptation**: Legal-specific model fine-tuning techniques
- **Multi-task Learning**: Joint optimization of classification and regression

##  Development

### Adding New Risk Categories

1. **Update Risk Taxonomy**:
```python
# In src/data/risk_taxonomy.py
enhanced_taxonomy['NEW_CATEGORY'] = 'NEW_RISK_TYPE'
```

2. **Modify Model Architecture**:
```python
# In src/models/legal_bert.py
self.risk_classifier = nn.Linear(config.hidden_size, num_risk_classes + 1)
```

3. **Update Training Configuration**:
```python
# In main.py
num_risk_classes = 8  # Updated count
```

### Custom Calibration Methods

```python
from src.evaluation import CalibrationMethod

class CustomCalibration(CalibrationMethod):
    def fit(self, logits, labels):
        # Custom calibration fitting
        pass
    
    def predict(self, logits):
        # Custom calibration prediction
        return calibrated_logits
```

## Technical Details

### Data Processing Pipeline
1. **CUAD Loading**: Parse JSON format with clause extraction
2. **Text Preprocessing**: Normalization, entity extraction, complexity scoring
3. **Risk Mapping**: Enhanced taxonomy application with 95.2% coverage
4. **Feature Engineering**: Word count, complexity metrics, entity counts
5. **Train/Val/Test Split**: 70/15/15 stratified split

### Model Training Process
1. **Data Preparation**: Tokenization with Legal-BERT tokenizer
2. **Multi-task Setup**: Combined loss function with task weighting
3. **Optimization**: AdamW with linear learning rate warmup
4. **Validation**: Early stopping based on validation loss
5. **Checkpointing**: Model state and training history preservation

### Evaluation Framework
1. **Classification Metrics**: Accuracy, F1-score, confusion matrix
2. **Regression Metrics**: R², MAE, MSE for severity/importance
3. **Calibration Assessment**: ECE, MCE, reliability diagrams
4. **Uncertainty Analysis**: Epistemic vs. aleatoric decomposition
5. **Decision Support**: Risk-based thresholds and recommendations

##  References

### Academic Papers
- **Legal-BERT**: Chalkidis et al. (2020) - Legal domain BERT pre-training
- **CUAD Dataset**: Hendrycks et al. (2021) - Contract understanding dataset
- **Uncertainty Quantification**: Guo et al. (2017) - Modern neural network calibration
- **Multi-task Learning**: Ruder (2017) - Multi-task learning overview

### Technical Resources
- **Transformers Library**: Hugging Face transformers for BERT implementation
- **PyTorch**: Deep learning framework for model development
- **Scikit-learn**: Calibration methods and evaluation metrics
- **Legal Domain**: Contract analysis and risk assessment methodologies

##  Contributing

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/new-feature`
3. **Commit changes**: `git commit -am 'Add new feature'`
4. **Push branch**: `git push origin feature/new-feature`
5. **Submit pull request**

### Development Guidelines
- Follow PEP 8 style guidelines
- Add comprehensive docstrings
- Include unit tests for new features
- Update documentation for API changes
- Validate on CUAD dataset before submission

