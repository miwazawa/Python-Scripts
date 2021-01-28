from flask import Flask,render_template
import pandas as pd

app = Flask(__name__)

@app.route('/')
def hello():
    return render_template('index.html')
    
@app.route('/index')
def index():  
    s='Hello'
    return s

if __name__ == "__main__":
    app.run(debug=True)


