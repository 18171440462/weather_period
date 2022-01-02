import torch.utils.data.distributed
import torchvision.transforms as transforms
import json
from torch.autograd import Variable
import os
from PIL import Image
import numpy as np
import tqdm

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)
classes_period = ('Afternoon', 'Dawn', 'Dusk', 'Morning', 'Night')
classes_weather = ('Cloudy', 'Fog', 'Rainy', 'Snow', 'Sunny')
save_json_path = "./result.json"
result = {}
total = []
transform_test = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
model_period = torch.load("train_period/973.526664576184589e-05period_model.pth")
model_period.eval()
model_period.to(DEVICE)
model_weather = torch.load("train_weather/1005.6687185563119796e-05weather_model.pth")
model_weather.eval()
model_weather.to(DEVICE)
path = './documents/test_images/'
testList = os.listdir(path)
for img_name in testList:
    img = Image.open(path + img_name)
    img = transform_test(img)
    img.unsqueeze_(0)
    img = Variable(img).to(DEVICE)
    out_period = model_period(img)
    out_weather = model_weather(img)
    # Predict
    _, pred_period = torch.max(out_period.data, 1)
    _, pred_weather = torch.max(out_weather.data, 1)
    each_obj = {}
    each_obj["filename"] = 'test_images\\'+ img_name
    each_obj["period"] = classes_period[pred_period.data.item()]
    each_obj["weather"] = classes_weather[pred_weather.data.item()]
    total.append(each_obj)
# print(total)
result["annotations"] = total
json.dump(result, open(save_json_path, 'w'), indent=4, cls=MyEncoder)