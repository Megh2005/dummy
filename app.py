import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from PIL import Image

# Streamlit Page Configuration (must be the first Streamlit command)
st.set_page_config(page_title="Emotion Recognition", page_icon="😊", layout="centered")

# Load the model and tokenizer
@st.cache_resource
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained("emotion_model", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("emotion_model")
    return model, tokenizer

model, tokenizer = load_model()

# Define emotion labels (adjust as per your model's training)
emotion_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']  # Example emotions

# Sidebar with About Section
st.sidebar.title("About This Project")
st.sidebar.info(
    """
    This is an **Emotion Recognition** application that uses a fine-tuned DistilBERT model 
    to predict emotions expressed in textual data. 

    - **Model**: DistilBERT
    - **Framework**: Hugging Face Transformers
    - **Purpose**: Demonstrates text classification using modern NLP techniques.

    Upload your text and see the magic happen! 😊
    """
)
st.sidebar.markdown("---")
st.sidebar.title("Creator Info")
st.sidebar.info(
    """
    **Sayambar Roy Chowdhury**  
    - [LinkedIn](https://www.linkedin.com/in/sayambar-roy-chowdhury/)  
    - [GitHub](https://github.com/Sayambar2004)  
    """
)

# Header Section
st.title("Emotion Recognition using DistilBERT")
st.markdown(
    """
    <style>
        .big-font {
            font-size:20px !important;
            color: #4F8BF9;
        }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown('<p class="big-font">Predict emotions expressed in your text with a fine-tuned DistilBERT model!</p>', unsafe_allow_html=True)

# Input Section
st.write("### Input Text")
user_input = st.text_area("Enter your text here:", placeholder="Type something meaningful...")

if st.button("Analyze Emotion"):
    if user_input.strip():
        # Tokenize input
        inputs = tokenizer(user_input, return_tensors="pt")
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][predicted_class].item()

        # Display result
        st.success("Analysis Complete!")
        st.subheader("Predicted Emotion:")
        st.write(f"**{emotion_labels[predicted_class]}** (Confidence: {confidence:.2f})")
    else:
        st.error("Please enter some text to analyze.")

# Footer Section
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center;">
        <strong>Created by Sayambar Roy Chowdhury</strong>  
        <a href="https://www.linkedin.com/in/sayambar-roy-chowdhury-731b0a282/" target="_blank">LinkedIn</a> | 
        <a href="https://github.com/Sayambar2004" target="_blank">GitHub</a>
    </div>
    """,
    unsafe_allow_html=True,
)
