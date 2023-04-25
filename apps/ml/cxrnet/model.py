import torch.nn as nn
import timm
import torch
import warnings
import base64
from PIL import Image
from io import BytesIO

warnings.filterwarnings("ignore")

from apps.ml.cxrnet.augmentations import get_transform
from apps.ml.cxrnet.gradcam import gradcam

transform = get_transform(False, 224, 0)

class CXRNet(nn.Module):

    def __init__(self, num_classes=7, model_name="coatnet_1_rw_224", pretrained=False):
        super().__init__()

        self.features = timm.create_model(model_name, num_classes=num_classes, pretrained=pretrained)

    def forward(self, x):
        return self.features(x)


def create_model(num_classes, model_name, pretrained_path):
    model = CXRNet(num_classes, model_name, pretrained_path)
    return model

class CXRClassifier:
    def __init__(self):

        state_dict = torch.load("collectedstatic/model/0.8960.pth", map_location="cpu")
        self.model = CXRNet()
        self.model.load_state_dict(state_dict)
        self.transform = transform
        self.logits_to_prob = nn.Sigmoid()

    def preprocessing(self, input_data):
        self.image_buffer = input_data
        try:
            logits = self.transform(image=input_data)["image"]
        except:
            raise KeyError("Invalid input data")

        return logits.unsqueeze(0)

    def predict(self, input_data):
        heatmap = gradcam(self.image_buffer, self.model)
        heatmap = Image.fromarray(heatmap)
        self.image_buffer = self.__pil2base64(heatmap)
        return self.model(input_data)

    def postprocessing(self, input_data):
        answer = self.logits_to_prob(input_data)[0]
        return {
            "answer": answer, 
            "gradcam": self.image_buffer,
            "category": ['主動脈硬化(鈣化)','動脈彎曲','肺野異常','肺紋增加','脊椎病變','心臟肥大','肺尖肋膜增厚'], 
            "status": "OK"
        }
    

    def __pil2base64(self, image):
        img_buffer = BytesIO()
        image.save(img_buffer, format="JPEG")
        byte_data = img_buffer.getvalue()
        base64_str=base64.b64encode(byte_data)
        return base64_str
    
    def compute_prediction(self, input_data):

        try:
            input_data = self.preprocessing(input_data)
            prediction = self.predict(input_data)  # only one sample
            prediction = self.postprocessing(prediction)
        except Exception as e:
            return {"status": "Error", "message": str(e)}

        return prediction