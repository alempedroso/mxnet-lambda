import os
import boto3
import json
import tempfile
import urllib2

import mxnet as mx
import numpy as np
import cv2
from collections import namedtuple

def laplacian_variance(image):
  return cv2.Laplacian(image, cv2.CV_64F).var()

def has_blur(image):
  gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  return laplacian_variance(gray_img)

def lambda_handler(event, context):
  url = event['url']
  resp = urllib2.urlopen(url)
  image = np.asarray(bytearray(resp.read()), dtype='uint8')
  img = cv2.imdecode(image, 1)
  blur_variance = has_blur(img)
  payload = {
    "blurVariance": blur_variance
  }

  return json.dumps(payload)
