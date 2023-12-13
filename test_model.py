import os
import tensorflow as tf
import gradio as gr
import pandas as pd #Reading in tabular data like csv files


class toxicClassifier:
    def __init__(self):
        self.loadModel = None
        self.MAX_WORDS = 200000
        self.vectorizer = tf.keras.layers.TextVectorization(max_tokens=self.MAX_WORDS,
                               output_sequence_length=1800,
                               output_mode='int')
        self.df = pd.read_csv(os.path.join('jigsaw-toxic-comment-classification-challenge', 'train.csv', 'train.csv'))
        self.X = self.df['comment_text']
        self.vectorizer.adapt(self.X.values)

    
    def score_comment(self, comment):
        vectorized_comment = self.vectorizer([comment])
        result = self.loadModel.predict(vectorized_comment)
        print(result, comment)

        text = ''
        for index, col in enumerate(['toxic','severe_toxic','obscene','threat','insult','identity_hate']):
            text += "{}: {} ({})\n".format(col, result[0][index]>0.5, result[0][index])
        return text

    def displayGradio(self):
        gui = gr.Interface(fn=self.score_comment,
                   inputs=gr.Textbox(lines=2, placeholder='Comment to Score'),
                   outputs='text')
        gui.launch()


model = toxicClassifier()
model.loadModel = tf.keras.models.load_model(os.path.join('models', 'toxicClassifier.h5'))
model.displayGradio()





