import streamlit as st
import streamlit.components.v1 as components
import h5py

# import sys
# sys.path.append('models')
import tensorflow as tf 


import tensorflow_hub as hub
from official.nlp.data import classifier_data_lib
from official.nlp.tools import tokenization

def define_vars():
    list_labels = [0,1] # Label categories
    max_seq_length = 512 # maximum length of (token) input sequences
    train_batch_size = 2
    bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2",trainable=True)
    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
    return list_labels, max_seq_length, train_batch_size, bert_layer, vocab_file, do_lower_case, tokenizer

list_labels, max_seq_length, train_batch_size, bert_layer, vocab_file, do_lower_case, tokenizer = define_vars()

def load_modelss():
    ai_detection_models = tf.keras.models.load_model('new_main_model.h5',custom_objects={'KerasLayer':bert_layer})
    print("Load model Succefully!!")
    return ai_detection_models

ai_detection_model = load_modelss()


def to_feature(text, label, label_list=list_labels, max_seq_length=max_seq_length, tokenizer=tokenizer):
  text_a = text.numpy()
  labels = label.numpy()
  example = classifier_data_lib.InputExample(guid=None,
                                             text_a=text_a,
                                             text_b=None,
                                             label=labels)
  feature = classifier_data_lib.convert_single_example(0, example, label_list, max_seq_length, tokenizer)

  return (feature.input_ids, feature.input_mask, feature.segment_ids, feature.label_id)

def to_feature_map(text, label):
  input_ids, input_mask, segment_ids, label_id = tf.py_function(to_feature, inp=[text,label],
                                                                Tout=[tf.int32,tf.int32,tf.int32,tf.int32])

  input_ids.set_shape([max_seq_length])
  input_mask.set_shape([max_seq_length])
  segment_ids.set_shape([max_seq_length])
  label_id.set_shape([])

  x = {
      "input_word_ids":input_ids,
      "input_mask":input_mask,
      "input_type_ids":segment_ids
  }

  return (x, label_id)

# Function to detect AI-generated text
def detect_ai_generated(text):
    proc_data = [text]
    term_data = tf.data.Dataset.from_tensor_slices((proc_data, [0]*len(proc_data)))
    term_data = (term_data.map(to_feature_map).batch(1))
    # Your AI detection logic here
    # Example: Use a pre-trained model
    prediction = ai_detection_model.predict(term_data)
    print("Prediction:", prediction)
    return 'AI Generated' if prediction[0] >= 0.5 else 'Human Generated'

# Streamlit App
def main():
    st.set_page_config(
        page_title="Text Generation with AI Detection",
        page_icon="🤖",
        layout="wide",
    )

    st.title('Text Generation with AI Detection')
    st.sidebar.title('Settings')

    # User Input
    user_input = st.text_area('Enter text here:', '')

    # Detect AI-generated text
    if st.button('Detect AI'):
        ai_result = detect_ai_generated(user_input)
        st.success(f'The input text is likely: {ai_result}')

    # Generate Text
    # st.sidebar.subheader('Text Generation:')
    # gen_prompt = st.sidebar.text_input('Enter a prompt for text generation:', 'Once upon a time,')
    # if st.sidebar.button('Generate Text', key='generate_button'):
    #     generated_text = generate_text(gen_prompt)
    #     st.subheader('Generated Text:')
    #     st.write(generated_text)

    # Additional Decoration
    components.html(
        """
        <style>
            body {
                background-color: #f5f5f5;
            }
            .stApp {
                max-width: 1200px;
                margin: 0 auto;
            }
            .sidebar .sidebar-content {
                background-color: #262730;
                padding: 20px;
                border-radius: 10px;
                color: white;
            }
        </style>
        """,
        height=0,
    )

if __name__ == '__main__':
    main()
