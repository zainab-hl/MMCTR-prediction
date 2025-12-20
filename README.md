## References

This project is built upon [1st Place Solution of WWW 2025 EReL@MIR Workshop
Multimodal CTR Prediction Challenge](https://arxiv.org/pdf/2505.03543) with all components  reimplemented from scratch, and small yet influential adjustments, including changes to embedding dimensions and modifications in how certain inputs are handled and integrated into the model.

## Overall Architecture

The pipeline follows a **four-stage design**:

### 1. Item Representation

Each item embedding is built by concatenating:
- Item ID embedding (learnable)
- Multiple tag embeddings (learnable)
- Frozen multimodal embedding (precomputed, non-trainable) (first used the ones provided, later extracted then used mine)

This yields a rich multimodal item representation.

---

### 2. Sequential Feature Learning

User interaction history is processed using a Transformer encoder.
Each historical item is concatenated with the target item embedding.

The model extracts:
- Short-term interest (last *k* interactions)
- Long-term interest (masked max-pooling)

**Output:** a fixed-size sequential feature vector.

---

### 3. Feature Interaction (DCNv2)

Target item embedding, side features, and sequential features are concatenated.
A Deep & Cross Network v2 (DCNv2) models:
- Explicit feature crosses
- High-order nonlinear interactions
- later, instead of passing the whole item, we passed only its Id_embedding.
---

### 4. CTR Prediction

A lightweight MLP with sigmoid activation predicts the click probability.
The model is trained using binary cross-entropy loss.

---

## Data Handling

- Parquet datasets from the WWW 2025 MM-CTR benchmark
- Custom Dataset for sequence padding and truncation
- Collator for efficient batching
- Frozen multimodal embeddings are loaded once and reused across the model

---

## Training Details

- **Optimizer:** Adam  
- **Loss:** Binary Cross-Entropy  

**Metrics:**
- ROC-AUC
- LogLoss
- Accuracy  

**Training strategies:**
- Gradient clipping
- L2 regularization
- Early stopping
- Learning rate scheduling

---

## Notes on Modifications

- Complete reimplementation of all model components
- Modified embedding dimensionality
- Revised input preprocessing and integration
- Modular and extensible architecture


## Results & Discussion

The proposed model achieves a ROC-AUC score of 0.8674 on the test set.
While this performance does not fully match the original winning solution, it demonstrates that the reimplemented architecture is able to effectively capture multimodal, sequential, and feature interaction signals.

Notably, this performance was reached after only 9 training epochs, indicating fast convergence and stable optimization.
Due to **time constraints and limited computational resources**, further hyperparameter tuning and extended experimentation (e.g., deeper architectures, alternative embedding sizes, or longer training schedules) were not feasible.

We believe that additional tuning and scaling would likely lead to further performance improvements.

