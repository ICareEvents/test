from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import pickle
import pandas as pd
import os

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/node-info/<node_name>', methods=['GET'])
def get_node_info(node_name):
    # Retrieve information based on node_name
    # Example load from a file renamed to x_train_alligned
    if not os.path.exists('x_train_alligned'):
        return f"No data file found for node lookup: {node_name}"
    with open('x_train_alligned', 'rb') as file:
        text = pickle.load(file)
    inds = [i for i in range(len(text)) if node_name in text[i].split()]

    df = pd.read_csv('df_raw.csv')
    output = '<br>'.join([f'Document {l}: {df["paragraphs"].values[l]}' for l in inds])
    node_info = f'Node: <b>{node_name}</b><br>Output:<br>{output}'
    return node_info

# Additional routes can go here if needed

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # Default to 8080
    app.run(debug=False, host='0.0.0.0', port=port)
