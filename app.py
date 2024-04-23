import streamlit as st
import safetensors
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F


model_path = 'model_save'
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)
model.eval()
model.to('cuda' if torch.cuda.is_available() else 'cpu')


label_dict = {
    'sadness': 0,
    'anger': 1,
    'love': 2,
    'surprise': 3,
    'fear': 4,
    'joy': 5
}

# Reverse the dictionary for mapping numeric IDs back to labels
id_to_label = {v: k for k, v in label_dict.items()}

# Load the model and tokenizer
model_path = 'model_save'
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)
model.eval()
model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

def predict_emotion(input_data):
    if isinstance(input_data, str) and input_data.endswith('.txt'):
        with open(input_data, 'r', encoding='utf-8') as file:
            text = file.read()
    else:
        text = input_data

    # Encode and prepare input data
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = inputs['input_ids'].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    attention_mask = inputs['attention_mask'].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Make prediction
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        probabilities = F.softmax(outputs.logits, dim=1)
        probs = probabilities.squeeze().tolist()

    # Determine the most likely emotion
    predicted_index = torch.argmax(outputs.logits, dim=1).item()
    overall_emotion = id_to_label[predicted_index]
    emotion_distribution = {id_to_label[i]: prob for i, prob in enumerate(probs)}

    return overall_emotion, emotion_distribution

def get_sentiment_prediction(text):
    # Use the model to predict sentiment
    sentiment, prob = predict_emotion(text)
    return sentiment, prob

# Streamlit interface
def main():
    st.title("Sentiment Analysis with Safetensors Model")
    st.write("Enter your text below:")
    user_input = st.text_area("Input Text", "Type here...")
    
    if st.button("Predict"):
        sentiment, prob = get_sentiment_prediction(user_input)
        st.write(f"Sentiment: {sentiment}")
        st.write(f"Sentiment: {prob}")

if __name__ == "__main__":
    main()
