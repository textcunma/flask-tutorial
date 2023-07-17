import base64

from flask import Flask, Response
from segment import Model

# Flaskのインスタンス生成
app = Flask(__name__)

# URLにおいて, GETメソッドを用いたリクエスト処理
@app.route('/v1/segment', methods=['GET'])
def run():
    model = Model()
    encoded_str = model.render(url = "https://github.com/pytorch/hub/raw/master/images/deeplab1.png")

    # Base64をデコード
    binary_img = base64.b64decode(encoded_str)

    # レスポンス生成
    response = Response(binary_img, mimetype='image/jpeg')
    return response

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port = 8080)