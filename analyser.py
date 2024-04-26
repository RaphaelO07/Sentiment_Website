import streamlit as st

from firebase_admin import firestore
import safetensors
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F
from io import StringIO
import plotly.graph_objects as go

#-------LOADING MODEL--------
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

#------------------------------------------------------------------------

def plot_dict(data_dict):
        traces = [go.Bar(x=list(data_dict.keys()), y=list(data_dict.values()))]
        fig = go.Figure(data=traces)
        fig.update_layout(title='Overall Degree of Emotions',xaxis_title='Emotions',yaxis_title='Value')
        st.plotly_chart(fig)

def app():
    if 'db' not in st.session_state:
        st.session_state.db = ''

    db=firestore.client()
    st.session_state.db=db

    ph = ''
    if st.session_state.username =='':
        ph = 'Login Required'
        st.header(ph)
    else:
        
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

        def get_segmented_emotions(text):
            text = text.replace("\n", "")
            text = text.replace("\r", "")
            text = text.replace("\r\n", "")
            sentences = text.split('.')
            print(sentences)
            emotion_list = {}
            for i in sentences:
                if len(i)> 0:
                    if len(sentences)>1:
                        s  = i + '.'
                    else:
                        s = i
                    senti, prob = get_sentiment_prediction(s)
                    emotion_list[s] = senti
            return emotion_list

        def get_color_for_emotion(emotion):
            if emotion == 'sadness':
                return "#0000FF" # Blue
            elif emotion == 'anger':
                return "#FF0000" # Red
            elif emotion == 'love':
                return ("#FF1493")  # Pink
            elif emotion == 'surprise':
                return ("#FFFF00")  # Yellow
            elif emotion == 'fear':
                return ("#8B4513")  # Brown
            elif emotion == 'joy':
                return ("#32CD32")  # Green
            else:
                return ("#FFFFFF")  # Default white


        def get_tooltip(sentence_emotion_dict):
            html = ""
            for sentence, emotion in sentence_emotion_dict.items():
                color = get_color_for_emotion(emotion)
                tooltip = emotion.capitalize()
                html += f'<p style="background-color:{color}; padding: 5px" title="{tooltip}">{sentence}</p>'
            return html

        # Streamlit interface
        def main():
            st.title("Sentiment Analysis with Safetensors Model")
            st.write("Enter your text below:")
            user_input = st.text_area("Input Text")
            st.write("OR")
            uploaded_files = st.file_uploader("Select Files", type="txt")
            
            if st.button("Predict"):
                if len(user_input) != 0:
                    sentiment, prob = get_sentiment_prediction(user_input)
                    st.subheader(f"Overall Emotion: {sentiment}")

                    plot_dict(prob)
                    st.subheader("Sentence Wise Emotions")
                    sentence_emotion_dict = get_segmented_emotions(user_input)
                    st.markdown(get_tooltip(sentence_emotion_dict), unsafe_allow_html=True)


                elif uploaded_files is not None:
                    stringio = StringIO(uploaded_files.getvalue().decode("utf-8"))
   
                    string_data = stringio.read()
                    sentiment, prob = get_sentiment_prediction(string_data)
                    st.subheader(f"Overall Emotion: {sentiment}")

                    plot_dict(prob)
                    st.subheader("Sentence Wise Emotions")
                    sentence_emotion_dict = get_segmented_emotions(string_data)
                    st.markdown(get_tooltip(sentence_emotion_dict), unsafe_allow_html=True)
                else:
                    st.warning("Text cannot be Empty")


        
        main()
