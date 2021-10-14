from flask import Flask
from flask import render_template
from flask import request
import detect


app = Flask(__name__)

@app.route("/")
def main():
    return render_template("main.html")

#
@app.route("/upload", methods=['POST'])
def upload():
    file = request.files["upload_image"]

    file.save("static/test.jpg")

    # savetype = 0 original savetype = 1 : merge, savetype = 2 : crop , savetype = save for static
    path = detect.detect(source="static/test.jpg",
                         weights='./weights/120e_32b.pt',
                         name="headlamp",
                         savetype=3,
                         )

    path = str(path)

    detect.detect(source=path,
                  weights="./weights/best_rain.pt",
                  name="rain",
                  savetype=4
                  )



    return render_template("result.html")

if __name__ =='__main__':
    app.run(host="0.0.0.0")