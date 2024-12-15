# Quora Question Pair Similarity Detection

This repository contains the implementation of a deep learning project to detect semantic similarity between question pairs using the Quora Question Pairs dataset. The project explores multiple state-of-the-art transformer-based architectures, including bi-encoders (Siamese networks with BERT/SBERT) and cross-encoder BERT models. The results demonstrate the effectiveness of transformer models in capturing nuanced semantic relationships for question similarity tasks.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Approaches and Models](#approaches-and-models)
4. [Results](#results)
5. [Project Highlights](#project-highlights)
6. [Installation](#installation)
7. [Usage](#usage)
8. [File Structure](#file-structure)
9. [References](#references)

---

## Introduction

Community-driven Q&A platforms like Quora often face challenges related to duplicate questions. This project addresses the problem by building models to identify semantically equivalent question pairs efficiently. Leveraging the Quora Question Pairs dataset, multiple architectures were implemented, compared, and optimized, resulting in an **F1-macro score of 93.28%** using a cross-encoder BERT model.

---

## Dataset

The dataset used in this project is the **Quora Question Pairs** dataset, which contains:

- **404,290** question pairs  
- **Features**:  
  - `id`: Unique identifier for the pair  
  - `qid1`, `qid2`: IDs of the two questions  
  - `question1`, `question2`: The text of the two questions  
  - `is_duplicate`: Binary label indicating duplicate status  

### Key Characteristics:
- **Imbalance**:  
  - 36.92% duplicate pairs  
  - 63.08% non-duplicate pairs  
- **Data Preprocessing**:  
  - Normalization  
  - Cleaning  
  - Stemming  
  - Tokenization  

---

## Approaches and Models

### 1. **Bi-Encoder (Siamese SBERT)**:
- **Dynamic Masking**: 15% probability for auxiliary task.  
- **Embedding Alignment**: Contrastive loss.  
- **Pooling Strategies**: Mean, Max, and CLS pooling.  
- **Performance**: F1-macro score of **89.68%** with dynamic masking.  

### 2. **Embedding Alignment (Contrastive Loss)**:
- Applied to both **SBERT** and **BERT** models.  
- **Performance**:  
  - SBERT: F1-macro score of **80.48%**  
  - BERT: F1-macro score of **79.48%**  

### 3. **Cross-Encoder BERT**:
- Combined question pairs during tokenization using `[CLS]` and `[SEP]`.  
- Employed **mixed-precision training** for computational efficiency.  
- **Performance**: F1-macro score of **93.28%**.  

---

## Results

| **Algorithm**                 | **F1-Macro Score** |
|-------------------------------|--------------------|
| SBERT + Dynamic Masking       | 89.68%            |
| SBERT + Contrastive Loss      | 80.48%            |
| BERT + Contrastive Loss       | 79.48%            |
| Cross-Encoder BERT            | 93.28%            |

---

## Project Highlights

- **Dynamic Masking**: Improved model robustness by simulating noise during training.  
- **Contrastive Loss**: Enhanced embedding alignment for better semantic representation.  
- **Cross-Encoder**: Achieved the highest performance by capturing pairwise interactions.  
- **Mixed Precision Training**: Reduced computational overhead without sacrificing accuracy.  

---

## Installation

### Prerequisites
- Python 3.8 or higher  
- PyTorch  
- Hugging Face Transformers  
- CUDA-enabled GPU (recommended)  


