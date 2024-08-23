import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import tempfile

# Load the trained model
new_model = tf.keras.models.load_model('new_model.h5')

# Define emotion labels corresponding to the model's output
emotion_labels = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Neutral',
    5: 'Sad',
    6: 'Surprise'
}

# Streamlit app
st.title('Human Facial Emotion Detection')
st.write('Upload an image of a human face to detect the emotion.(Image having only one face)')
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_image is not None:
    # Save the uploaded image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webp") as temp_file:
        temp_file.write(uploaded_image.read())
        temp_file_path = temp_file.name

    # Read the image using PIL and convert to OpenCV format
    #image_pil = Image.open(temp_file_path).convert("RGB")
    image = cv2.imread(temp_file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Load Haar Cascade for face detection
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        st.image(image, caption='Processed Image.', use_column_width=True)
        st.write("No face detected.")
    else:
        for x, y, w, h in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = image[y:y + h, x:x + w]
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            facess = faceCascade.detectMultiScale(roi_gray)
            if len(facess) == 0:
                st.image(image, caption='Processed Image.', use_column_width=True)
                st.write("No face detected.")
            else:
                for (ex, ey, ew, eh) in facess:
                    face_roi = roi_color[ey: ey + eh, ex:ex + ew]
        final_image = cv2.resize(face_roi, (224, 224))
        final_image = np.expand_dims(final_image, axis=0)
        final_image = final_image / 255.0
    #if len(faces) == 0:
     #   st.image(image, caption='Processed Image.', use_column_width=True)
      #  st.write("No face detected.")
    #else:
    #    for (x, y, w, h) in faces:
    #        roi_gray = gray[y:y + h, x:x + w]
    #        roi_color = image[y:y + h, x:x + w]
    #        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Process the detected face
    #        final_image = cv2.resize(roi_color, (224, 224))
    #        final_image = np.expand_dims(final_image, axis=0)
    #        final_image = final_image / 255.0

            # Predict the emotion
        emotion = np.argmax(new_model.predict(final_image))

        st.image(image, caption='Processed Image.', use_column_width=True)
        st.write(f'Predicted Emotion: {emotion_labels[emotion]}')
