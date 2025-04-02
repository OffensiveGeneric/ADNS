# Subtask 1: Explainable Feature Engineering for Network Anomaly Detection

## Objective
Develop interpretable network traffic features that maintain detection accuracy while providing human-understandable indicators for security analysts.

## Key Research Papers
1. **"Adaptable, Incremental, and Explainable NIDS for IoT"** (2025)  
   - Proposes dynamic feature selection using SHAP values  
   - Achieves 94.2% accuracy on TON-IoT 2025 dataset  
   - [DOI: 10.1016/j.engappai.2025.110143](https://doi.org/10.1016/j.engappai.2025.110143)

2. **"Trust My IDS"** (2024)  
   - Introduces "Explainability Index" metric for features  
   - Combines NetFlow and Zeek logs with attention mechanisms  
   - [DOI: 10.1016/j.cose.2024.104191](https://doi.org/10.1016/j.cose.2024.104191)

3. **"Evaluating Explainability in NIDS"** (2024)  
   - Benchmark study of 12 feature extraction methods  
   - Finds tradeoff between interpretability and detection rate  
   - [arXiv:2408.14040](https://arxiv.org/abs/2408.14040)

## Comparative Analysis of Existing Solutions

### Solution 1: SHAP-Based Feature Selection (From "Adaptable NIDS" paper)
**Strengths**:
- Dynamic importance weighting adapts to new attacks
- Reduces feature space by 40% while maintaining 98% detection rate
- Provides native explainability through SHAP values

**Weaknesses**:
- Computational overhead (15-20% slower than static features)
- Requires labeled data for initial training
- Less effective on encrypted traffic

**Applicability**:
Best suited for environments with:
- Regular concept drift (ex: IoT networks)
- Available compute resources
- Need for analyst-interpretable features

### Solution 2: Hierarchical NetFlow Features (From "Trust My IDS")
**Strengths**:
- Three level feature hierarchy (packet/flow/behavior)
- Human readable feature groupings
- 92% accuracy on CIC-IDS2024

**Weaknesses**:
- Manual feature engineering required
- Less adaptable to novel attacks
- Higher false positives on low volume attacks

**Applicability**:
Ideal for:
- Industrial control systems
- Regulated environments needing audit trails
- Networks with stable traffic patterns

### Solution 3: Temporal Graph Embeddings (NetSecLog-2024 Baseline)
**Strengths**:
- Captures complex network relationships
- Automatic feature learning
- 95% recall on zero day attacks

**Weaknesses**:
- Black box nature requires post hoc explanation
- High memory requirements
- Complex to implement

**Applicability**:
Recommended for:
- Large enterprise networks
- Advanced SOC teams
- Scenarios prioritizing detection over explainability

## Feature Engineering Methodologies Comparison

| Methodology          | Interpretability | Detection Rate | Computation Cost | Dataset Compatibility |
|----------------------|------------------|----------------|------------------|------------------------|
| SHAP Selection       | High             | 94.2%          | Medium           | CIC-IDS2024, TON-IoT   |
| Hierarchical NetFlow | Very High        | 92.0%          | Low              | CIC-IDS2024            |
| Graph Embeddings     | Low              | 95.8%          | High             | NetSecLog-2024         |

## Recommended Tools & Frameworks

1. **Feature Extraction**:
   - Zeek (Bro) Network Security Monitor: [https://zeek.org/](https://zeek.org/)
   - CICFlowMeter v4.0 (2024 update): [https://www.unb.ca/cic/research/applications.html](https://www.unb.ca/cic/research/applications.html)

2. **Explainability**:
   - SHAP for Network Features: [https://github.com/slundberg/shap](https://github.com/slundberg/shap)
   - ELI5 for Packet Inspection: [https://github.com/eli5-org/eli5](https://github.com/eli5-org/eli5)

3. **Dimensionality Reduction**:
   - scikit learn PCA with Feature Names: [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)
   - UMAP for Security Data: [https://github.com/lmcinnes/umap](https://github.com/lmcinnes/umap)

## Implementation Roadmap

1. **Phase 1: Baseline Features** 
   - Implement standard NetFlow v9 features
   - Add Zeek log derived features
   - Baseline evaluation on CIC-IDS2024

2. **Phase 2: Explainability Enhancement** 
   - Integrate SHAP based feature selection
   - Develop hierarchical feature groups
   - Evaluate on TON-IoT 2025

3. **Phase 3: Adaptive Features** 
   - Implement incremental PCA
   - Add concept drift detection
   - Final testing on NetSecLog-2024

## Evaluation Metrics

1. **Detection Performance**:
   - F1 Score (balanced accuracy)
   - False Positive Rate
   - Zero day Attack Recall

2. **Explainability**:
   - Feature Interpretability Score (FIS)
   - Analyst Decision Time (human evaluation)
   - Explanation Consistency

## References

- Canadian Institute for Cybersecurity. (2024). CIC-IDS2024 Dataset. [https://www.unb.ca/cic/datasets/ids-2024.html](https://www.unb.ca/cic/datasets/ids-2024.html)
- UNSW Sydney. (2025). TON-IoT 2025 Dataset. [https://research.unsw.edu.au/projects/toniot-datasets](https://research.unsw.edu.au/projects/toniot-datasets)
- Stanford Network Analysis Project. (2024). NetSecLog-2024. [https://snap.stanford.edu/netseclog/](https://snap.stanford.edu/netseclog/)
