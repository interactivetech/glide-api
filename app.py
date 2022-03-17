import os
# Using Flask since Python doesn't have built-in session management
from flask import Flask, session, render_template, request, jsonify
# Our target library
import requests
import json
from pathlib import Path
from base64 import encodebytes
from PIL import Image
import io
import numpy as np
from model import load_models, sample_model

import logging
from logging.handlers import RotatingFileHandler
from time import strftime
import traceback
from time import time
import torch as th

has_cuda = th.cuda.is_available()
device = th.device('cpu' if not has_cuda else 'cuda')
t0=time()

options, options_up,model,model_up,diffusion, diffusion_up = load_models(has_cuda,
                                                                         device,
                                                                         timestep_respacing='25',
                                                                         timestep_respacing_up='fast27')
print("Done Loading, time: {} sec.".format(time()-t0))

batch_size = 10
guidance_scale = 3.0

# Tune this parameter to control the sharpness of 256x256 images.
# A value of 1.0 is sharper, but sometimes results in grainy artifacts.
upsample_temp = 0.997


app = Flask(__name__)

# def get_response_image(image_path):
#     pil_img = Image.open(image_path, mode='r') # reads the PIL image
#     byte_arr = io.BytesIO()
#     pil_img.save(byte_arr, format='jpeg') # convert the PIL image to byte array
#     encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii') # encode as base64
#     return encoded_img
def get_response_image(im):
    byte_arr = io.BytesIO()
    im.save(byte_arr, format='jpeg') # convert the PIL image to byte array
    encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii') # encode as base64
    return encoded_img

@app.route('/')
def index():
    return "Welcome to dalleapi.com"

@app.route('/generate', methods=['POST'])
def generate():
    if request.method == "POST":
        
        input_json = request.get_json(force=True) 
        print("Prompt: {}".format(input_json['prompt']))
        timestamp = strftime('%Y-%b-%d-%H:%M')
        app.logger.info('%s,%s,%s,%s,%s,%s,%s,%s',
                timestamp,
                request.remote_addr,
                request.method,
                request.scheme,
                request.full_path,
                input_json['prompt'],
                input_json['n_images']
                )
        up_samples = sample_model(
                 input_json['prompt'],
                 input_json['n_images'],
                 guidance_scale,
                 upsample_temp,
                 model,
                 model_up,
                 diffusion,
                 diffusion_up,
                 options,
                 options_up
                 )
        # model.del_cache()
        # model_up.del_cache()
        # images =generate_images(
        #     input_json['prompt'],
        #     input_json['n_images'],
        #     model,
        #     tokenizer,
        #     vqgan,
        #     clip,
        #     processor,
        #     model_params, 
        #     vqgan_params, 
        #     clip_params,
        #     input_json['gen_top_k'],
        #     )
        encoded_images = [get_response_image(i) for i in up_samples]
        # encoded_imges = []
        # for image_path in range(10):
        #     encoded_imges.append(get_response_image_test())
        return jsonify({'prompt':input_json['prompt'],
                        'n_images':input_json['n_images'],
                        'result': encoded_images})
        # except Exception as e:
        #     print(e)
        #     return jsonify({"sorry": "Sorry, no results! Please try again."}), 500

# @app.after_request
# def after_request(response):
#     timestamp = strftime('[%Y-%b-%d %H:%M]')
#     app.logger.info('2-%s %s %s %s %s %s', timestamp, request.remote_addr, request.method, request.scheme, request.full_path, response.status)
#     return response

# @app.errorhandler(Exception)
# def exceptions(e):
#     tb = traceback.format_exc()
#     timestamp = strftime('[%Y-%b-%d %H:%M]')
#     app.logger.error('%s %s %s %s %s 5xx INTERNAL SERVER ERROR\n%s', timestamp, request.remote_addr, request.method, request.scheme, request.full_path, tb)
#     return e.status_code

if __name__ == '__main__':
    handler = RotatingFileHandler('app.csv', maxBytes=100000, backupCount=3)
    logger = logging.getLogger('tdm')
    app.logger.setLevel(logging.INFO)
    app.logger.addHandler(handler)
    app.logger.info("timestamp,request.remote_addr,request.method,request.scheme,request.full_path,prompt,n_images,gen_top_k")
    app.run(
        host="0.0.0.0",
        port=5000
    )