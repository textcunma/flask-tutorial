import io
import torch
import base64
import urllib
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50

class Model:
    def __init__(self) -> None:
        self.seg_model = ImageSegmentModel()

    def render(self, url: str = "https://github.com/pytorch/hub/raw/master/images/deeplab1.png") -> str:
        # モデル実行, 結果をRGBに変換
        img = self.seg_model(url=url, is_show=False).convert("RGB")

        # 出力結果をバイナリデータに移行
        buffered = io.BytesIO()
        img.save(buffered, 'JPEG')

        # Base64にエンコード, バイナリデータから文字列化
        encoded_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return encoded_str


class ImageSegmentModel:
    """DeepLab-V3による画像セグメンテーション"""

    def __init__(self) -> None:
        self.model = deeplabv3_resnet50(weights='DEFAULT')
        self.model.eval()
        self.filename = ""
    
    def __call__(self, url: str, is_show: bool = False) ->  Image.Image:
        # 画像をダウンロード
        self._download_img(url)
        
        # 画像ファイルをモデル入力できるように変更
        img = Image.open(self.filename).convert("RGB")
        input_batch = self._convert_input(img)

        # GPUに伝送
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            self.model.to('cuda')

        # 推論実行
        with torch.no_grad():
            output = self.model(input_batch)['out'][0]
        output_predictions = output.argmax(0)

        # 着色
        colors = self._decide_color()
        colored_img = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(img.size)
        colored_img.putpalette(colors)

        # 表示
        if is_show:
            plt.imshow(colored_img)
            plt.show()

        return colored_img
  
    def _convert_input(self, img):
        """画像ファイルをモデル入力に対応する形式に変更"""
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(img)
        input_batch = input_tensor.unsqueeze(0)
        return input_batch
    
    def _decide_color(self):
        """各クラスの色を決定"""
        palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
        colors = (colors % 255).numpy().astype("uint8")
        return colors

    def _download_img(self, url):
        """画像ファイルをダウンロード"""
        url, filename = (url, "tmp.png")
        try: 
            urllib.URLopener().retrieve(url, filename)
        except: 
            urllib.request.urlretrieve(url, filename)
        
        self.filename = filename

if __name__ == '__main__':
    model = Model()
    model.render()