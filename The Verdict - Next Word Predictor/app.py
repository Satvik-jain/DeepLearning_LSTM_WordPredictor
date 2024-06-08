import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import numpy as np


model = load_model("The_Verdict.h5")
with open("tokenizer.pickle", "rb") as handle:
        tokenizer = pickle.load(handle)

def prediction(t='',l=1):
  text = t
  sentence_length = l
  for repeat in range(sentence_length):
    token_text = tokenizer.texts_to_sequences([text])
    padded_token_text = pad_sequences(token_text, maxlen = 230, padding = 'pre')
    pos = np.argmax(model.predict(padded_token_text))
    for (word,index) in tokenizer.word_index.items():
      if index == pos:
        text = text + " " + word
  return text

import gradio as gr

demo = gr.Interface(title = "The Verdict",
                    examples = [['It had always been'], ['I found the couple at'],['She glanced out almost']],
                    fn=prediction,
                    inputs=[gr.Textbox(lines = 2, label = 'Query', placeholder='Enter Here', value=""),
                            gr.Slider(1,100,step = 1, label = "How many Words to generate?", value = 1)],
                    outputs=gr.Text(lines = 7, ), allow_flagging = 'never', theme=gr.themes.Base())

demo.launch(share = True)