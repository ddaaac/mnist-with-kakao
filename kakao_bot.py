# -*- coding: utf-8 -*-

from flask import Flask
from flask import request
from flask import jsonify
from flask import json
import urllib.request
import mnist

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "Hello World"


@app.route("/keyboard")
def keyboard():
        content = {
            'type' : 'text',
            }
        return jsonify(content)

@app.route("/message",methods=['GET', 'POST'])
def message():
        data = json.loads(request.data)
        img_url = data['content']
        
        urllib.request.urlretrieve(img_url, './number.png')
      
        result = mnist.mnist()
        
        response ={
                "message" :{
                        "text" : result
                }
        }

        response = json.dumps(response)

        return response

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)