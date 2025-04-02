# Reproducible Sources: Adaptive Detection Models

## Open Source Implementations

### Core Frameworks
1. **Incremental NIDS Toolkit**  
   - Reference implementation from "Adaptable NIDS for IoT" (2025 paper)  
   - Supports elastic weight consolidation and dynamic expansion  
   - Source: [GitHub: adaptable-nids/core](https://github.com/adaptable-nids/core)  
   - License: GPL-3.0  

2. **TrustMyIDS Ensemble**  
   - Preconfigured ensemble models (RF/CNN/Isolation Forest)  
   - Includes SHAP consensus explanation generator  
   - Source: [GitLab: trustmyids/ensemble](https://gitlab.com/trustmyids/ensemble)  
   - License: Apache 2.0  

3. **Streaming NIDS Baseline**  
   - NetSecLog-2024 official detection pipeline  
   - Online clustering and drift detection components  
   - Source: [Stanford Snap: netseclog-nids](https://snap.stanford.edu/netseclog/code.html)  
   - License: BSD-2-Clause  

## Pretrained Models

| Model | Type | Dataset | Access |
|-------|------|---------|--------|
| **CIC-Adaptive-XGBoost** | Incremental | CIC-IDS2024 | [HuggingFace](https://huggingface.co/CIC-IDS/adaptive-xgboost) |
| **TON-IoT-Explanation-Ensemble** | Multi model | TON-IoT 2025 | [Zenodo](https://zenodo.org/record/7890123) |
| **NetSecLog-Clustering** | Streaming | NetSecLog-2024 | [Stanford SNAP](https://snap.stanford.edu/netseclog/models.html) |

## Benchmark Datasets

1. **CIC-IDS2024**  
   - Includes adaptation scenarios with concept drift  
   - Download: [Official Portal](https://www.unb.ca/cic/datasets/ids-2024.html)  
   - Citation:  **Canadian Institute for Cybersecurity.** *CIC-IDS2024 Dataset.* 2024,  
www.unb.ca/cic/datasets/ids-2024.html.

2. **TON-IoT 2025**  
   - Contains labeled adaptation cycles for 15 attack types  
   - Download: [UNSW Data Portal](https://research.unsw.edu.au/projects/toniot-datasets)  
   - Special Note: Includes explanation stability benchmarks  

3. **NetSecLog-2024**  
   - Streaming format with timestamps for real time testing  
   - Access: [Stanford SNAP](https://snap.stanford.edu/netseclog/)  
   - Update Policy: Quarterly threat scenario additions  

## Documentation & Tutorials

### Academic Guides
1. **"Incremental Learning for NIDS"**  
   - Step by step adaptation protocol  
   - [PDF Guide](https://adaptable-nids.org/docs/2025-incremental-guide.pdf)  

2. **Ensemble Explanation Methods**  
   - TrustMyIDS technical white paper  
   - [DOI: 10.1016/j.cose.2024.104191-supp](https://doi.org/10.1016/j.cose.2024.104191-supp)  

### Industry Documentation
1. **Cisco Model Adaptation Framework**  
   - Production deployment guidelines  
   - [Technical Report](https://blogs.cisco.com/security/xai-nids-supplement)  

2. **Palo Alto Adaptive IDS Cookbook**  
   - Recipe based adaptation strategies  
   - [GitBook](https://pan-adaptive-ids.gitbook.io/2025-edition/)  
