import json
import logging
import sys
import model
import spacy

from flask import Flask, request
from flask import render_template
from flask_socketio import SocketIO, emit
from spacy import displacy
from IPython.core.display import display, HTML
from globals import VIZ_COLOR_OPTIONS

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
socketio = SocketIO(app)

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    tokens = request.form['data'].split()
    annotations = model.predict(request.form['data'])
    annotations = annotations[0]
    ex = generate_sens_viz_cont(tokens, annotations[:len(tokens)])
    ex['settings'] = {}
    html = displacy.render(ex, style='ent', manual=True, options={'colors':VIZ_COLOR_OPTIONS})
    return json.dumps({'status':'OK', 'html':html})

@socketio.on('connect', namespace='/socket')
def socket_connect():
    print('Client connected')

@socketio.on('predict', namespace='/socket')
def socket_predict(message):
    text = model.preprocess(message['data'])
    annotations = model.predict(text)
    annotations[0] = annotations[0][1:-1] # remove first and last [CLS] + [SEP]
    text_tokens, labels = model.align_tokenization_with_labels(text.split(), annotations)
    print(text_tokens)
    print(labels)
    ex = model.generate_sens_viz_cont(text_tokens[0], labels[0][:len(labels[0])])
    ex['settings'] = {}
    html = displacy.render(ex, style='ent', manual=True, options={'colors':VIZ_COLOR_OPTIONS})
    emit('annotations', {'data': html})

if __name__ == "__main__":
    app.run()