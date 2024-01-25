#!/usr/bin/env python
# coding: utf-8

import requests


url = 'http://localhost:9696/predict'

#data = {'path': 'fruit-dataset-small/test/Banana/100_100.jpg'}
data = {'path': 'https://raw.githubusercontent.com/Horea94/Fruit-Images-Dataset/master/Test/Banana/100_100.jpg'}


response = requests.post(url, json=data).json()

print(response)
