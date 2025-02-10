# ADNS
Tyler Kleint - Research 
Vadzim Ruzha - System Savvy
Mazen Zarrouk - Project Manager


###Anomaly Detection in Network Security

##Overview:
Build a machine learning system that detects unusual or suspicious activity in network traffic.
Can help identify cyber threats like DoS attacks, data breaches, or malware infections.

##Tech Stack:
Python, Scikit-learn, TensorFlow/PyTorch for ML models.
Pandas, NumPy for data preprocessing.
Wireshark / Zeek (Bro) for collecting network traffic data.
NSL-KDD, CICIDS2017, UNSW-NB15 as common datasets for intrusion detection.

##Approach:
Collect and preprocess network traffic data (packet features like IP addresses, ports, timestamps).
Extract relevant features and reduce dimensionality with PCA or Autoencoders.

##Train an ML model:
Unsupervised: Autoencoders, Isolation Forest, One-Class SVM for detecting novel attacks.
Supervised: Random Forest, XGBoost, CNN/RNN models for classifying known attack types.
Evaluate the model’s accuracy, precision, recall.
(Optional) Deploy as a real-time IDS (Intrusion Detection System).

##Challenges:
Handling imbalanced datasets (since normal traffic is much more frequent than attacks).
Reducing false positives.
Optimizing for real-time detection.

##Competitors:
[Cisco Stealthwatch](https://www.cisco.com/c/en/us/products/collateral/security/stealthwatch/secure-network-analytics-aag.html)
[IBM QRadar Network Insights](https://www.ibm.com/docs/en/qsip/7.4?topic=insights-qradar-network-overview)
[NetFlow Network Anomaly Detection](https://www.manageengine.com/products/netflow/network-anomaly-detection.html)


![Blank diagram](https://github.com/user-attachments/assets/c2712032-cc1a-4121-ada8-1fbf973dab76)

