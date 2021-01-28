import flask
import pandas as pd

app = flask.Flask(__name__)

@app.route('/')
def index():
    df = pd.read_csv("KEN_ALL_ROME.CSV", encoding = "s-jis")
    for row in df:
        print(row)
    #csv_data=[]
    s = 'kasu'
    return s

if __name__ == "__main__":
    app.run(port=8000,debug=False)


