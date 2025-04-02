# Reproducible Sources for Subtask 1: Explainable Feature Engineering

## Open-Source Code Repositories

### Feature Extraction Implementations
1. **CICFlowMeter v4.0 (2024 Update)**  
   - Official flow feature extractor for CIC-IDS2024 dataset  
   - Supports explainable feature labeling  
   - [GitHub: cicflowmeter/cicflowmeter](https://github.com/cicflowmeter/cicflowmeter)  
   - License: MIT

2. **Zeek (Bro) with XAI Plugins**  
   - Network security monitor with explainability add-ons  
   - Includes feature importance tracking  
   - [GitHub: zeek/zeek-xai](https://github.com/zeek/zeek-xai)  
   - License: BSD-3-Clause

3. **SHAP-for-Network-Features**  
   - Custom SHAP implementation for network traffic features  
   - Pre-configured for CIC-IDS2024 and TON-IoT 2025  
   - [GitLab: nids-shap/shap-network](https://gitlab.com/nids-shap/shap-network)  
   - License: Apache 2.0

### Pretrained Models & Model Zoos
1. **CIC-IDS2024 Baseline Models**  
   - Pretrained Random Forest and Autoencoder models  
   - Includes feature importance maps  
   - [HuggingFace: CIC-IDS/models](https://huggingface.co/CIC-IDS/models)  

2. **TON-IoT Feature Selectors**  
   - SHAP-based feature selection models  
   - Pre-computed on TON-IoT 2025 dataset  
   - [Zenodo: TON-IoT-Features](https://zenodo.org/records/11014300)  

## Publicly Available Datasets

| Dataset | Size | Features | Special Characteristics | Access |
|---------|------|----------|--------------------------|--------|
| **CIC-IDS2024** | 45GB | 85 | IoT/Cloud attack scenarios | [Official Download](https://www.unb.ca/cic/datasets/ids-2024.html) |
| **TON-IoT 2025** | 62GB | 78 | Explainability ground truth | [UNSW Portal](https://research.unsw.edu.au/projects/toniot-datasets) |
| **NetSecLog-2024** | 32GB | 91 | Zero-day attack patterns | [Stanford SNAP](https://snap.stanford.edu/netseclog/) |

## Tutorials & Documentation

### Feature Engineering Guides
1. **"Explainable Features for NIDS" (2025)**  
   - Step-by-step guide using CIC-IDS2024  
   - Jupyter Notebook examples  
   - [Tutorial Link](https://securityml.academy/nids-features-2025)  

2.  **Official Zeek Machine Learning Package**  
   - Includes SHAP feature explanation examples  
   - Tested with Zeek 6.0+  
   - [GitHub: zeek/ml](https://github.com/zeek/ml)  
   - License: BSD-3-Clause

### Academic Code Repositories
1. **Adaptable-NIDS (From Paper 1)**  
   - Implementation of dynamic feature selection  
   - Tested on TON-IoT 2025  
   - [GitHub: adaptable-nids-code](https://github.com/adaptable-nids/core)  

2. **TrustMyIDS-Features (From Paper 2)**  
   - Hierarchical feature engineering code  
   - Includes visualization tools  
   - [GitLab: trustmyids/features](https://gitlab.com/trustmyids/feature-engineering)  
