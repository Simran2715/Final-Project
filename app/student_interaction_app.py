import streamlit as st 
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import LabelEncoder
from transformers import T5Tokenizer, T5ForConditionalGeneration
import tensorflow as tf
from keras.models import load_model
from PIL import Image 
import cv2 
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences


st.set_page_config(
    page_title="Student Interaction App",
    page_icon="👩‍💻",
    layout="wide",
    initial_sidebar_state="expanded")


choice=st.sidebar.selectbox('Navigator :',['Introduction','Pass or Fail','Score Range Predictor','Dropout Risk Analyzer','Digit Recognizer','Topic Detector','Topic Summarizer'])

if choice=='Introduction':
    st.title('👩‍🎓Student Interaction App')
    st.image('images.jpeg')
    st.write('''Welcome Here!

    This app is providing all the information related to if you want to check pass or fail grade, or either you want to check the Future Score.
    Along with that you can navigate to different windows for asking machine learning and ai models to check for dropout risk generator if any.
    Although one can review their handwritten digit recognizer markesheets and summarize any topic or paragraph in just few lines as needed.
    ''')

elif choice=='Score Range Predictor':
    st.title('📑Score Range prediction')

    score_range=pd.read_csv('Student_Performance.csv')
    model= pickle.load(open('model_selected.pkl', 'rb'))
    encoder= pickle.load(open('encoder.pkl', 'rb'))

    student_id=st.number_input('Student ID')
    time_spent_on_app=st.number_input('Time Spent on App')
    past_grades=st.number_input('Past Grades')
    attendance_rate=st.number_input('Attendance Rate')
    time_spent_on_quiz=st.number_input('Time Spent on Quiz')
    question_attemted=st.number_input('Question attempted')
    pass_fail = st.selectbox('Pass or Fail', score_range['pass_fail'].unique())
    topic_difficulty = st.selectbox('Topic Difficulty', score_range['topic_difficulty'].unique())

    pass_fail_encoder=encoder['pass_fail'].transform([pass_fail])[0]
    topic_difficulty_encoder=encoder['topic_difficulty'].transform([topic_difficulty])[0]

    if st.button('Predict Future Score'):
        input_data=np.array([[student_id,time_spent_on_quiz,time_spent_on_app,attendance_rate,past_grades, question_attemted,pass_fail_encoder,topic_difficulty_encoder]])
        # input_data=input_data.reshape(1,-1)
        predict_score=model.predict(input_data)
        st.success(f'Predicted Score : {predict_score[0]:.2f} marks')

elif choice=='Pass or Fail':
    st.title('📌Prediction for Pass or Fail')

    model= pickle.load(open('rf.pkl', 'rb'))

    student_id=st.number_input('Student ID')
    time_spent_on_app=st.number_input('Time Spent on App')
    past_grades=st.number_input('Past Grade')
    attendance_rate=st.number_input('Attendence Rate')
    time_spent_on_quiz=st.number_input('Time Spent on Quiz')
    question_attempted=st.number_input('Question Attempted')

    if st.button('Predict Pass or Fail'):
        input_data=np.array([[student_id,time_spent_on_quiz,time_spent_on_app,attendance_rate,past_grades, question_attempted]])
        # input_data=input_data.reshape(1,-1)
        prediction=model.predict(input_data)
        st.success(f'Prediction : {prediction}')


elif choice=='Dropout Risk Analyzer':
    st.title('‼️Dropout Risk Predictor')

    model= pickle.load(open('XGB_classifier.pkl', 'rb'))

    student_id=st.number_input('Student ID')
    inactivity_score=st.number_input('Inactivity Score')
    poor_performance_score=st.number_input('Poor Score')
    inconsistent_engagement_score=st.number_input('Inconsistent Engagement Score')
    study_hpurs_per_week=st.number_input('Study Hours Spent per Week')
    attendance_rate=st.number_input('Attendance Percentage')
    previous_failures=st.number_input('Previous Failures')

    if st.button('Dropout Predictor'):
            input_data=np.array([student_id,inactivity_score,poor_performance_score,inconsistent_engagement_score,study_hpurs_per_week,attendance_rate,previous_failures])
            input_data=input_data.reshape(1,-1)

            prediction=model.predict(input_data)
            st.success(f'Prediction : {prediction}')

elif choice=='Topic Summarizer':
    st.title('📜Text Summarizer')
    st.markdown("""
    Enter a student's answer in the text box below to summarize your answer. This model is related to summarizing any topic of your aubject area
    """)
    # Text input area
    user_input = st.text_area("Enter student answer here:", height=150,
    placeholder="E.g., 'The process of cellular respiration converts glucose into ATP.'")

    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    def summarize(text):
        inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        return tokenizer.decode(outputs[0])

    # Example usage:
    # text = st.text_input('Text')
    summary = summarize(user_input)
    # print(summary)

    if st.button ('Summarize'):
        st.write('Your summarized text: ')
        st.success(summary)

elif choice=='Digit Recognizer':
    # st.set_page_config(page_title='Student Digit Recognizer', layout='centered')

    model = load_model('my_model.keras')
    #-------Streamlit UI-----------------
    st.title("✍️ Handwritten Digit Recognition")
    st.markdown("""
    Welcome to the Handwritten Digit Recognizer!
    Upload an image of a handwritten digit (0-9) and let the trained CNN model classify it.
    """)

    st.write("---")

    # Image Upload Section
    st.subheader("Upload a Handwritten Digit Image")
    uploaded_file = st.file_uploader(
        "Choose an image (PNG, JPG, JPEG)",
        type=["png", "jpg", "jpeg"]
    )

    # Prediction Logic
    if uploaded_file is not None:
        try:
            # Read the image
            image = Image.open(uploaded_file).convert('L') # Convert to grayscale

            st.image(image, caption='Uploaded Image', use_column_width=True)
            st.write("")
            st.write("Classifying...")

            # Preprocess the image
            # Convert PIL Image to OpenCV format (numpy array)
            img_array = np.array(image)

            # Resize to 28x28 pixels
            img_resized = cv2.resize(img_array, (28, 28), interpolation=cv2.INTER_AREA)

            # Invert colors if necessary (MNIST has white digits on black background)
            # Check average pixel value to decide if inversion is needed
            if np.mean(img_resized) > 128: # If the image is mostly light (black digit on white background)
                img_processed = 255 - img_resized # Invert colors
            else:
                img_processed = img_resized

            # Normalize to 0-1 range and add batch and channel dimensions
            img_normalized = img_processed.astype('float32') / 255.0
            img_input = img_normalized.reshape(1, 28, 28, 1)

            # Make prediction
            prediction = model.predict(img_input)
            predicted_digit = np.argmax(prediction)
            confidence = np.max(prediction) * 100

            st.success(f"**Predicted Digit: {predicted_digit}**")
            st.info(f"Confidence: {confidence:.2f}%")


        except Exception as e:
            st.error(f"Error processing image: {e}")
            st.warning("Please ensure the uploaded image contains a clear handwritten digit.")

    st.write("---")
    st.markdown(
        """
        **How it works:**
        This app uses a Convolutional Neural Network (CNN) trained on the MNIST dataset.
        The uploaded image is converted to grayscale, resized to 28x28 pixels,
        and then fed into the model for classification.
        """
    )
    st.markdown(
        "**Note:** For best results, upload clear, centered images of single digits."
    )

elif choice=='Topic Detector':
    st.title("📚 Student Answer Topic Detector")
    st.markdown("""
    Enter a student's answer in the text box below to predict its subject category.
    This model is trained to classify text into topics like **Mathematics, Science, History, Literature, Geography, Art, and Computer Science**.
    """)
    # Text input area
    user_input = st.text_area("Enter student answer here:", height=150,
    placeholder="E.g., 'The process of cellular respiration converts glucose into ATP.'")


    MODEL_DIR = "trained_model"
    MODEL_PATH = os.path.join(MODEL_DIR, "topic_classifier_model.h5")
    model = tf.keras.models.load_model(MODEL_PATH)
    TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")
    LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
    MAXLEN_PATH = os.path.join(MODEL_DIR, "maxlen.txt") # To save maxlen value

    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
    with open(LABEL_ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)
    with open(MAXLEN_PATH, 'r') as f:
        maxlen = int(f.read())
        st.success("Model loaded successfully!")

    def predict_topic(text_input, model, tokenizer, label_encoder, maxlen):
        """
        Predicts the topic of a new text input using the trained model.
        """
        if not text_input:
            return None, None # Indicate no prediction if input is empty

        new_sequence = tokenizer.texts_to_sequences([text_input])
        new_padded_sequence = pad_sequences(new_sequence, maxlen=maxlen, padding='post', truncating='post')

        # Predict requires 2D input (batch size, sequence length)
        prediction_probs = model.predict(new_padded_sequence, verbose=0)
        predicted_class_index = np.argmax(prediction_probs, axis=1)[0]
        predicted_label = label_encoder.inverse_transform([predicted_class_index])[0]

        # Get confidence score for the predicted label
        confidence = prediction_probs[0][predicted_class_index] * 100

        return predicted_label, confidence

    if st.button("Predict Topic"):
        if user_input:
            with st.spinner("Analyzing text..."):
                predicted_topic, confidence = predict_topic(user_input, model, tokenizer, label_encoder, maxlen)

            if predicted_topic:
                st.success(f"**Predicted Topic:** {predicted_topic}")
                st.info(f"Confidence: {confidence:.2f}%")
            else:
                st.warning("Could not predict topic. Please try another input.")
        else:
            st.warning("Please enter some text to get a prediction!")


