# AXL-ICT: Adversarially-Enhanced Negative Sampling Techniques for Dense Passage Retrieval in a Multilingual Setting
## 📖 Overview

AXL-ICT is a novel framework for multilingual dense passage retrieval that extends ICT-P through three key innovations:

- **Generative Adversarial Negatives** - Synthetic hard negatives via Qwen
- **Adversarial Filtering** - Dynamic retention of challenging examples

**Target Performance**: improved over ICT-P baseline across 11 languages

---

## 🎯 Key Features

✅ **Teacher-Free** - No expensive cross-encoder models required  
✅ **Efficient** - Single GPU training in ~48 hours  
✅ **Multilingual** - Supports 11 typologically diverse languages  
✅ **Scalable** - Avoids repeated re-indexing like ANCE  

### Supported Languages

Arabic • Bengali • English • Finnish • Indonesian • Japanese • Korean • Russian • Swahili • Telugu • Thai

---

## 🚀 Quick Start

### Installation
Create environment
python -m venv venv
source venv/bin/activate # Windows: venv\Scripts\activate

Install dependencies
pip install -r requirements.txt

### Requirements

torch>=2.0.0
transformers>=4.30.0
faiss-gpu>=1.7.4
datasets>=2.14.0
numpy>=1.24.0
scikit-learn>=1.3.0
wandb>=0.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0

---

## 💾 Datasets

### MS MARCO (Pretraining)
- 8.8M passages, ~500K queries
- [Download](https://microsoft.github.io/msmarco/)
python data/download_msmarco.py --output data/msmarco/

### Mr.TyDi (Fine-tuning & Evaluation)
- 11 languages, ~15K queries per language
- [Download](https://github.com/castorini/mr-tydi)
python data/download_mrtydi.py --output data/mrtydi/


---

## 🏃 Training & Evaluation

### 1. Pretrain on MS MARCO
### 2. Fine-tune on Mr.TyDi
### 3. Evaluate
### 4. Visualize Clusters

## 🔬 Methodology

### Three-Component Framework

#### 1️⃣ Generative Adversarial Negatives
- Uses Qwen 2.5-3B Instruct for synthetic negative generation
- Techniques: entity substitution, numerical perturbation, logical negation

#### 2️⃣ Adversarial Filtering
- Dynamic threshold based on model confidence
- Retains top 25% most challenging negatives
- Creates generator-retriever feedback loop

### Training Pipeline

MS MARCO Pretraining (10K steps)
↓
Mr.TyDi Fine-tuning (50K steps)
↓
├── Generate adversarial negatives
└── Filter by similarity threshold
