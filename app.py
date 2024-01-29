import streamlit as st
import joblib
import tensorflow as tf
import dill

with open('to_feature_map.pkl', 'rb') as f:
    loaded_to_feature_map = dill.load(f)

# with open('to_feature.pkl', 'rb') as f:
#     to_feature = dill.load(f)

# Main function to run the app
def main():
    st.title('AI Text Detection App')

    # Input text from user
    user_input = st.text_area('Enter text here:', '')

    if st.button('Detect'):
        # Transform the input text using the loaded TF-IDF vectorizer
        sample_example = [user_input]
        test_data = tf.data.Dataset.from_tensor_slices((sample_example, [0]*len(sample_example)))

        # Apply the loaded to_feature_map function to the test data
        input_tfidf = test_data.map(loaded_to_feature_map).batch(1)

        # Make prediction using the loaded model
        # prediction = model.predict(input_tfidf)

        # Display the result
        prediction = [True]
        result = 'AI Generated' if prediction[0] == 1 else 'Human Generated'
        st.write(f'The input text is likely: {result}')

if __name__ == '__main__':
    main()