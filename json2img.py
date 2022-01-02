import json
import os
import shutil
path = "./documents/train.json"
dir = "/home/ie/桌面/ZDW/projects/weather_period/documents/train_images/"
data_dir = "/home/ie/桌面/ZDW/projects/weather_period/data/"

num = 0
with open(path,"r") as f:
    files = json.load(f)
    annotations = files["annotations"]
    for each_image in annotations:
        img_name = each_image["filename"].split("\\")[-1]
        weather = each_image["weather"]
        # period = each_image["period"]
        weather_ls = os.listdir(data_dir + "weather/train/")
        # period_ls = os.listdir(data_dir + "period/train/")
        if weather not in weather_ls:
            os.mkdir(data_dir+ "weather/train/" + weather)
        print(data_dir +"weather/train/" + weather +"/"+ img_name)
        shutil.copyfile(dir+img_name,data_dir +"weather/train/" + weather +"/"+ img_name)
        # if period not in period_ls:
        #     os.mkdir(data_dir + "period/train/"+ period)
        # shutil.copyfile(dir+img_name,data_dir +"period/train/"+ period +"/"+ img_name)
