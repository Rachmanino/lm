from flask import Flask, render_template, request, jsonify
import random
import numpy as np
from attention import attention
app = Flask(__name__)

def normalize_and_convert_to_hex(matrix):
    normalized_matrix = matrix / np.max(matrix, axis=1)[:, None]

    gray_matrix = (255 - normalized_matrix * 255).astype(int)

    hex_matrix = np.array([['#{:02x}{:02x}{:02x}'.format(g, g, g) for g in row] for row in gray_matrix])
    
    return hex_matrix

def main(text):
    dis_matrix, tokens = attention(text)
    hex_matrix = normalize_and_convert_to_hex(dis_matrix) 
    num_tokens = len(tokens)
    color_dict = {}
    for i in range(num_tokens):
        colorlist = []
        for j in range(num_tokens):
            color = hex_matrix[i][j]
            colorlist.append(color)
        color_dict[i] = colorlist  
    return color_dict

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    data = request.get_json()
    text = data.get('text', '')
    color_dict = main(text)  
    return jsonify(color_dict)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
