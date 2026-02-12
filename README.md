
🚀DB3 – :Unifying Vision and Language for Robust Fake News Detection Using 
Novel Deep Samples

🫂 Team Information
 Shaik Siraz- 22471A05O2 ( [LinkedIn](https://www.linkedin.com/in/siraz-shaik-25108a28b) )
 Contribution:
 - Complete end-to-end project implementation
 - Dataset selection and preparation
 - Image preprocessing and enhancement pipeline
 - MobileNetV2 fine-tuning and optimization
 - Attention mechanism integration
 - Model training, validation, and evaluation
 - Comparative analysis with multiple CNN architectures
 - Result analysis, documentation, and GitHub setup
 ---
Shaik Malka Jan Shafi- 22471A0509 ( [LinkedIn](https://www.linkedin.com/in/jan-shafi-shaik-malka-432664287) )
Work Done: 
- Literature survey assistance
- Dataset understanding and validation support
- Result verification and documentation support
---
Nuti Nanda Kameswar- 23475A0504 ( [LinkedIn](https://www.linkedin.com/in/nanda-kameswar-801784207) )
Work Done: 
- Model testing assistance
- Presentation preparation
- Project report formatting

---

📌 Abstract
—Fake news identification has gained it’s relevance
over the last few years as a result of the large-scale propagation
of fake information through social media. The paper presents
a new method for detecting fake news that uses both text and
image information together for identification with multimodal
learning that combines both text and image modalities. Using
the Fakeddit dataset, three new models were created and tested:
(1) Retrained MLP Classifier with BERT + MobileNetV2 (91
precision), (2) CLIP + MLP (88.24 precision) and (3) DistilBERT
+ EfficientNet + MLP (89 precision). The three models all achieve
better performance than the baseline 88.83 in the original paper.
This paper proves that combining different architectures beyond
the conventional literature can achieve better classification results
in fake news.The three models all achieve better performance
than the baseline 88.83% from the original paper.
Index Terms-Fake news detection, multimodal deep learning,
transformer models, BERT, MobileNetV2, CLIP, DistilBERT,
EfficientNet, MLP, vision language fusion, binary classification,
lightweight neural networks, deep fusion architectures.

---

## Paper Reference (Inspiration)
👉 **[Paper Title Multimodal Fake News Detection Based on Contrastive Learning and Similarity Fusion
  – Author Names Yan Li
 ](https://ieeexplore.ieee.org/document/10718307)**
This project is inspired by the architectural concepts, attention mechanisms, and preprocessing strategies presented in the Yan Li research paper, while adapting the implementation to a MobileNetV2-based lightweight architecture suitable for academic and practical deployment.

---

✨ Our Improvement Over Existing Paper
- Lightweight MobileNetV2 backbone instead of heavier CNNs
- Reduced computational cost while maintaining high accuracy
- Designed for easy deployment and academic reproducibility

---

📌 About the Project
🔍 What the Project Does
This project presents a Multimodal Fake News Detection System that automatically classifies news posts as:

✅ Real News

❌ Fake News

The system analyzes both:

📝 Text content (headlines/titles)

🖼 Associated images

It uses advanced deep learning architectures to combine visual and textual information for robust classification.
🔄 System Workflow
Text Input + Image Input
↓
Text Preprocessing (Cleaning, Tokenization)
↓
Image Preprocessing (Resizing, Normalization)
↓
Feature Extraction

BERT / DistilBERT (Text Encoder)

MobileNetV2 / EfficientNet / CLIP (Image Encoder)
↓
Multimodal Feature Fusion (Concatenation / Unified Embedding)
↓
MLP Classifier
↓
Prediction (Real / Fake)

---

 📁 Dataset Used
Fakeddit Multimodal Dataset

A large-scale multimodal fake news dataset containing Reddit posts with:

Post title (text)

Associated image

Binary label (Real / Fake)

📊 Dataset Statistics (After Filtering)
Split	Total Samples	Real	Fake
Train	40,000	20,000	20,000
Validation	5,000	2,500	2,500
Test	5,000	2,500	2,500
Total	50,000	25,000	25,000

---
🛠 Technologies & Dependencies
Python 3.x

TensorFlow / PyTorch

HuggingFace Transformers

OpenAI CLIP

NumPy

Pandas

OpenCV

Matplotlib

Scikit-learn

Google Colab (Tesla T4 GPU)

---

🔎 Data Preprocessing
📝 Text Processing
Lowercasing

Special character removal

Tokenization

Encoding using BERT/DistilBERT tokenizer

🖼 Image Processing
Resizing to 224 × 224

Normalization with ImageNet statistics

Removal of corrupted/missing images

🏷 Label Encoding
Real → 0

Fake → 1

🧪 Model Architectures
1️⃣ BERT + MobileNetV2 + MLP (Best Performing Model)
Text Encoder: BERT (768-dim embeddings)

Image Encoder: MobileNetV2 (1280-dim features)

Fusion: Feature Concatenation

Classifier: Multi-Layer Perceptron

Loss: Binary Cross-Entropy

Optimizer: AdamW

2️⃣ CLIP + MLP
Unified 512-dim multimodal embeddings

Direct multimodal alignment

Lightweight architecture

3️⃣ DistilBERT + EfficientNet + MLP
Reduced computational complexity

Suitable for edge deployment

Dropout + ReLU activation

⚙ Training Configuration
Parameter	Value
Batch Size	32
Epochs	10–15
Optimizer	AdamW
Learning Rate	2e-5 (BERT), 1e-4 (Others)
Loss Function	Binary Cross-Entropy
Platform	Google Colab (GPU)
📊 Model Evaluation
📈 Metrics Used
Accuracy

Precision

Recall

F1-Score

Confusion Matrix
---

🏆 Performance Results
🔹 Model Comparison
Model	Accuracy	F1-Score
BERT + MobileNetV2 + MLP	91.03%	0.91
CLIP + MLP	88.23%	0.88
DistilBERT + EfficientNet + MLP	82.00%	0.82
Base Paper (Bagged CNN)	88.83%	0.88
✅ Proposed model outperforms base paper benchmark (88.83%)
✅ Strong generalization across both Real and Fake classes
✅ Efficient multimodal fusion improves accuracy

🔬 Ablation Study
Modality	Accuracy
Text-only (BERT)	87.50%
Image-only (MobileNetV2)	82.00%
Multimodal Fusion	91.03%
---
📌 Multimodal learning clearly improves classification performance.

⚠ Limitations & Future Work
🔻 Limitations
Binary classification only

Late fusion architecture

No explainability module integrated

Tested on single dataset (Fakeddit)
---

🚀 Future Enhancements
Cross-modal attention mechanisms

Explainable AI (Grad-CAM, LIME)

Multilingual fake news detection

Federated learning integration

Real-time social media deployment

3-way or multi-class classification
---

🌍 Deployment Applications
Social media misinformation monitoring

News verification platforms

Browser extensions for fake news alerts

Content moderation tools

AI-based fact-checking systems

👨‍💻 Developed By
Shaik Siraz
Project Lead & Developer
🔗 https://www.linkedin.com/in/siraz-shaik-25108a28b


📧 Email: sksiraz29@gmail.com
🔗 LinkedIn: (https://www.linkedin.com/in/siraz-shaik-25108a28b)


🙏 Acknowledgments
Fakeddit Dataset Contributors

HuggingFace Transformers Library

OpenAI CLIP Framework

Google Colab GPU Resources

Research Community & IEEE References


---
