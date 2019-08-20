from flask import Flask,jsonify,request
import flask
import pickle
import numpy as np
import sys
import logging
from keras.models import load_model
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from keras.preprocessing import sequence
import warnings
warnings.filterwarnings("ignore")
# load model_

model2 = load_model("./model_protein_sequence.h5")

model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
labelencoder = joblib.load('./labelencoder.pkl')
global graph
graph = tf.get_default_graph()

app = Flask(__name__)
app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)


char2index_dict=joblib.load('./char2index_dict.pkl')

def predict2(text):
    text=text.lower()
    final=[]
    seq1=[]
    for s in (text):
      x=char2index_dict[s]
      seq1.append(str(x))
    final.append(seq1)
    final_sequence = sequence.pad_sequences(final, maxlen=100,padding='post')
    nb_classes = 24
    targets = np.array(final_sequence)
    one_hot_train = np.eye(nb_classes)[targets]
    #one ---[None,100,24]
    with graph.as_default():
      res=model2.predict(one_hot_train)
      print("res",res)
    pred = labelencoder.inverse_transform([np.argmax(res)])
    print("Step 1 cleared")
    return (pred[0])

@app.route('/')
def index():
    return flask.render_template('index.html')


@app.route('/predict',methods=['GET','POST'])
def predict():
    input = request.form.to_dict()
    input = list(input.values())
    result = predict2(input[0])
    res = {'result':result}
    return flask.render_template('index2.html',user=res)

if __name__=='__main__':
    app.run(debug=True)
