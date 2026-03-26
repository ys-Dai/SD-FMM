# Detecting Market Manipulation with Dual-branch Self-supervised Learning: A Unified Framework Integrating Frequency-informed Anomaly Synthesis and Domain-Specific Features

This repository contains the official implementation of **SD-FMM**, a Self-supervised Detection framework tailored to Financial Market Manipulation.


---

## Overview

SD-FMM makes four primary contributions:

1. We propose a novel self-supervised detection framework for financial market manipulation named SD-FMM, where a Dual-branch Contrastive Detection Neural Network (DCDNN) can characterize manipulation from different perspectives, ensuring that the unique patterns of these anomalies are deeply acquired. Additionally, an innovative contrastive pretext task within DCDNN significantly boosts the network’s sensitivity to the anomaly boundaries and its responsiveness across the entire anomaly interval. Consequently, our approach can detect the occurrence of manipulation more quickly and can localize their duration more accurately.
2. We propose a new synthetic anomaly generation method based on the few-shot learning of domain-knowledge and the dynamic frequency analysis of market manipulation. Given the scarcity of annotated data in practice, our approach enables the generation of unlimited realistic anomaly samples through the natural simulation of real-world manipulation patterns, supporting the self-supervised learning of our detection framework. To the best of our knowledge, this is among the first methods to systematically use dynamic frequency distributions both as perturbation targets and as synthetic outputs for time series anomaly generation.
3. We propose a novel method that leverages domain-specific features to facilitate the detection of market manipulations. These features, derived from various domains, provide a comprehensive expression and thorough exploration of manipulation patterns. Innovatively, we set their fusion results as the base for all subsequent self-supervised learning and detection processes within our framework. Our method refines and clusters unique traces of anomalous transaction behaviors from multiple perspectives, enabling the anomaly signals to be amplified repeatedly. Consequently, the temporal representation during manipulation intervals exhibits substantially increased intensity and distinctiveness, making it easier to identify the presence of various types of manipulation. 
4. We evaluate our approach using a real-world case study, based on a newly collected proprietary dataset of 25 Chinese stock market manipulation cases. Our framework consistently outperforms 12 state-of-the-art baseline methods across a comprehensive assessment of 11 metrics, encompassing both precision and timeliness. Meanwhile, in order to strengthen the credibility of our method’s superiority and verify its robustness, we also introduce a public benchmark of 338 cryptocurrency pump-and-dump schemes for evaluation. Extensive quantitative and qualitative experiments demonstrate that our framework not only achieves superior detection accuracy and generalization capabilities but also exhibits a significantly faster and stronger response to real-world market manipulations, offering novel insights for practical financial surveillance applications.


## Installation

```bash
git clone https://github.com/SD-FMM.git
cd SD-FMM
pip install -r requirements.txt
```

Requires Python ≥ 3.8 and PyTorch ≥ 1.13.

---

## Usage

### Train the model

```bash
python train.py
```



## License

This project is released under the MIT License. See `LICENSE` for details.
