# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 18:45:27 2018

@author: Jorge
"""

import cv2
import os

raw_path = '../../etc/raw_data'
processed_path = '../../etc/processed_faces'

for dirname, dirnames, filenames in os.walk(raw_path):
    for filename in filenames:
        img = cv2.imread(os.path.join(raw_path, filename))
        res = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(processed_path, filename), res)