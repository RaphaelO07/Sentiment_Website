import streamlit as st
import streamlit_survey as ss
from firebase_admin import firestore
import safetensors
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F
from io import StringIO
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np

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


questions = [
            "How was your day? Write down the noticeable events.",
            "How do you see yourself currently? Academically or career-wise.",
            "How do you imagine yourself in the bigger picture of life?",
            "When was the last time you sang for yourself?",
            "When was the last time you cried?",
            "List out a few qualities that you think you have.",
            "List out a few defects that you think you have."
        ]
answers = [""] * len(questions)

#------------------------------------------------------------------------

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

        def form_success(answers):
            def autopct_generator(threshold):
                def inner_autopct(pct):
                    return ('%.1f%%' % pct) if pct > threshold else ''
                return inner_autopct

            st.success("Successfully, Predicting your current mood ...")
            probs = {}
            for i in range(len(answers)):

                if len(answers[i])>0:
                    sentiment, prob = predict_emotion(answers[i])
                    for j in prob:
                        if j in probs:
                            probs[j]+=prob[j]
                        else:
                            probs[j] = prob[j]

            group_threshold = 0.05  # Threshold for grouping into Others
            others = sum(prob for emotion, prob in probs.items() if prob < group_threshold)
            grouped_emotions = {emotion: prob for emotion, prob in probs.items() if prob >= group_threshold}
            if others > 0:
                grouped_emotions['Others'] = others

            # Determine the highest emotion
            max_emotion = max(probs, key=probs.get)

            # Define colors for each segment

            colors_list = {"sadness":'yellow', "anger":'orangered', "love":'pink', "surprise":'blue', "fear":'brown',"joy":'yellowgreen', "Others":"grey"}
            colors = []
            for i in grouped_emotions:
                colors.append(colors_list[i])
            # Create a pie chart with a hole in the middle (donut chart)
            fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(aspect="equal"))
            wedges, labels, autopct_texts = ax.pie(grouped_emotions.values(), labels=['',]*len(grouped_emotions), autopct=autopct_generator(group_threshold*100),
                                                startangle=140, colors=colors,
                                                wedgeprops=dict(width=0.3, edgecolor='w', linewidth=2, linestyle='-', antialiased=True))

            # Adjust the position of the labels
            label_distance = 1.05  # How far from the center of the pie the labels appear
            for label, pct_text in zip(labels, autopct_texts):
                label.set_horizontalalignment('center')
                label.set_verticalalignment('center')
                label.set_rotation_mode('anchor')
                label.set_position((label_distance * np.cos(np.deg2rad((label.get_rotation() - 90) % 360)), 
                                    label_distance * np.sin(np.deg2rad((label.get_rotation() - 90) % 360))))
                pct_text.set_visible(False)  # Hide the default percentage labels

            # Annotate each wedge with a text label
            for i, wedge in enumerate(wedges):
                ang = (wedge.theta2 - wedge.theta1) / 2. + wedge.theta1
                x = np.cos(np.deg2rad(ang))
                y = np.sin(np.deg2rad(ang))

                horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
                connectionstyle = "angle,angleA=0,angleB={}".format(ang)
                bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
                kw = dict(arrowprops=dict(arrowstyle="->", connectionstyle=connectionstyle),
                        bbox=bbox_props, zorder=0, va='center')

                emotion = list(grouped_emotions.keys())[i]
                total_no = (len(answers) - answers.count(''))
                probability = list(grouped_emotions.values())[i]/total_no
                label_text = f"{emotion} {probability*100:.1f}%"
                ax.annotate(label_text, xy=(x, y), xytext=(1.5*x, 1.2*y),
                            horizontalalignment=horizontalalignment, **kw)

            # Title and center text
            ax.set_title("Emotion Probabilities", fontsize=16, weight='bold', pad=35)
            ax.text(0, 0, max_emotion, ha='center', va='center', fontsize=14, style='italic', weight='bold')

            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

            # Display the chart in Streamlit
            st.pyplot(fig)



         # Initialize questions and answers
        

        def main():
            st.title("Mood Reflection Form")

            
            survey = ss.StreamlitSurvey("Mood Prediction")
            pages = survey.pages(7, on_submit=lambda: form_success(answers))
            with pages:
                if pages.current == 0:
                    st.subheader(f"Question {pages.current+1}")
                    answers[pages.current] = st.text_area(f"{questions[pages.current]}",value = f"{answers[pages.current]}", key = pages.current)

                elif pages.current == 1:
                    st.subheader(f"Question {pages.current+1}")
                    answers[pages.current] = st.text_area(f"{questions[pages.current]}",value = f"{answers[pages.current]}", key = pages.current)
                elif pages.current == 2:
                    st.subheader(f"Question {pages.current+1}")
                    answers[pages.current] = st.text_area(f"{questions[pages.current]}",value = f"{answers[pages.current]}", key = pages.current)
                elif pages.current == 3:
                    st.subheader(f"Question {pages.current+1}")
                    answers[pages.current] = st.text_area(f"{questions[pages.current]}",value = f"{answers[pages.current]}", key = pages.current)
                elif pages.current == 4:
                    st.subheader(f"Question {pages.current+1}")
                    answers[pages.current] = st.text_area(f"{questions[pages.current]}",value = f"{answers[pages.current]}", key = pages.current)
                elif pages.current == 5:
                    st.subheader(f"Question {pages.current+1}")
                    answers[pages.current] = st.text_area(f"{questions[pages.current]}",value = f"{answers[pages.current]}", key = pages.current)
                elif pages.current == 6:
                    st.subheader(f"Question {pages.current+1}")
                    answers[pages.current] = st.text_area(f"{questions[pages.current]}",value = f"{answers[pages.current]}", key = pages.current)
            
            



            
        main()  