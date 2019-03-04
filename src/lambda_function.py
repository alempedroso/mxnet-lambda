import os
import boto3
import json
import tempfile
import urllib2

import numpy as np
import cv2
from collections import namedtuple

def laplacian_variance(image):
  return cv2.Laplacian(image, cv2.CV_64F).var()

def blur_variance(image):
  gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  return laplacian_variance(gray_img)

def lambda_handler(event, context):
  print("Received event: " + json.dumps(event["body"]))

  event_payload = json.loads(event["body"])
  url = event_payload["url"]
  resp = urllib2.urlopen(url)
  image = np.asarray(bytearray(resp.read()), dtype='uint8')
  img = cv2.imdecode(image, 1)
  payload = {
    "statusCode": 200,
    "body": json.dumps({
      "blurVariance": blur_variance(img)
    })
  }

  return payload
