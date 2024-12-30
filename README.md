# **Emotion Recognition with DistilBERT**

*A Natural Language Processing (NLP) project to classify text into emotions using the pre-trained DistilBERT model.*

---

## **Table of Contents**
1. [Introduction](#introduction)
2. [Features](#features)
3. [Dataset](#dataset)
4. [Model Architecture](#model-architecture)
5. [Training Process](#training-process)
6. [Requirements](#requirements)
7. [Usage](#usage)
8. [Streamlit Interface](#streamlit-interface)
9. [Project Structure](#project-structure)
10. [Results](#results)
11. [About the Creator](#about-the-creator)

---

## **Introduction**
Emotion Recognition is a Natural Language Processing (NLP) task that involves classifying text data into distinct emotion categories like *joy*, *sadness*, *anger*, *fear*, *love*, etc. This project utilizes the **DistilBERT** model, a lightweight version of BERT, fine-tuned for multi-class emotion classification.

This project aims to bridge the gap between textual data and human emotions, providing accurate predictions for real-world applications like chatbots, mental health analysis, and more.

---

## **Features**
- **Pre-trained Model:** Uses Hugging Face’s DistilBERT for efficient training and inference.
- **Multi-Class Emotion Classification:** Predicts emotions from text with high accuracy.
- **Streamlit Interface:** A user-friendly web application to interact with the model.
- **Customizable:** The model can be extended to other text classification tasks.

---

## **Dataset**
- **Source:** The dataset used for training is a publicly available dataset for emotion classification (e.g., from Kaggle or Hugging Face datasets library).
- **Format:** The dataset consists of two columns:
  - `Text`: Input text (e.g., "I feel amazing today!").
  - `Emotion`: Target label (e.g., "joy").
- **Preprocessing:**
  - Removal of special characters, URLs, and unnecessary symbols.
  - Tokenization and padding using DistilBERT's tokenizer.

---

## **Model Architecture**
The fine-tuned **DistilBERT** model includes:
1. A **Transformer Encoder**: Extracts meaningful embeddings from text.
2. A **Classification Head**: A dense layer for emotion prediction.

---

## **Training Process**
1. **Preprocessing:**
   - Tokenization using DistilBERT tokenizer.
   - Padding and truncation of text to ensure uniform sequence length.
2. **Model Fine-Tuning:**
   - Loss Function: Cross-Entropy Loss.
   - Optimizer: AdamW with weight decay.
   - Learning Rate Scheduler: Warm-up followed by decay.
3. **Evaluation:**
   - Validation on a separate dataset to track model performance.

For a detailed explanation, refer to the [Training Process Section](#training-process) in the documentation.

---

## **Requirements**
- **Python** (>= 3.8)
- Libraries:
  - `transformers`
  - `torch`
  - `pandas`
  - `numpy`
  - `streamlit`

Install dependencies using:  
```bash
pip install -r requirements.txt
```

---

## **Usage**
1. Clone the repository:
   ```bash
   git clone https://github.com/<YourUsername>/Emotion-Recognition.git
   cd Emotion-Recognition
   ```

2. Download the pre-trained model:
   Save the model files (`config.json`, `model.safetensors`, `tokenizer.json`, etc.) in the directory `emotion_model`.

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

4. Open the app in your browser at `http://localhost:8501`.

---

## **Streamlit Interface**
The Streamlit interface allows you to:
- Input text into a text box.
- Receive emotion predictions from the fine-tuned DistilBERT model.
- View a description of the project and creator details.

The app includes:
- **Home Section**: Predict emotions in real-time.
- **About Section**: Learn more about the project.
- **Creator Section**: View creator information and social links.

---

## **Project Structure**
```
Emotion-Recognition/
│
├── emotion_model/
│   ├── config.json
│   ├── model.safetensors
│   ├── special_tokens_map.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── vocab.txt
│
├── app.py               # Streamlit application code
├── requirements.txt     # List of dependencies
├── README.md            # Project documentation
└── data/                # (Optional) Folder to store dataset
```

---

## **Results**
- **Training Accuracy:** 0.94  
- **Validation Accuracy:** 0.92  
- **Test Accuracy:** 0.91  

The model achieves high accuracy and generalizes well on unseen data.

---

## **About the Creator**
- **Name:** Sayambar Roy Chowdhury  
- **LinkedIn:** [Sayambar Roy Chowdhury](https://www.linkedin.com/in/sayambar-roy-chowdhury)  
- **GitHub:** [Sayambar2004](https://github.com/Sayambar2004)  

Feel free to connect for feedback, suggestions, or collaborations!
