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
from waitress import serve

has_cuda = th.cuda.is_available()
device = th.device('cpu' if not has_cuda else 'cuda')
t0=time()

options, options_up,model,model_up,diffusion, diffusion_up = load_models(has_cuda,
                                                                         device,
                                                                         timestep_respacing='25',
                                                                         timestep_respacing_up='fast27')
options_100, options_up_100,model_100,model_up_100,diffusion_100, diffusion_up_100 = load_models(has_cuda,
                                                                         device,
                                                                         timestep_respacing='100',
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
    im_pillow = Image.fromarray(im)
    byte_arr = io.BytesIO()
    im_pillow.save(byte_arr, format='jpeg') # convert the PIL image to byte array
    encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii') # encode as base64
    return encoded_img

@app.route('/')
def index():
    return "Welcome to dalleapi.com"

@app.route('/generate', methods=['POST'])
def generate():
    if request.method == "POST":
        try:
            input_json = request.get_json(force=True) 
            assert len(input_json['prompt']) < 4097
            print("Prompt: {}".format(input_json['prompt']))
            timestamp = strftime('%Y-%b-%d-%H:%M')
            if input_json['type']=='fast':
                app.logger.info('%s,%s,%s,%s,%s,%s,%s,%s',
                        timestamp,
                        request.remote_addr,
                        request.method,
                        request.scheme,
                        request.full_path,
                        input_json['prompt'],
                        input_json['n_images'],
                        input_json['type']
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
                        options_up,
                        device
                        )
            elif input_json['type']=='high':
                app.logger.info('%s,%s,%s,%s,%s,%s,%s,%s',
                        timestamp,
                        request.remote_addr,
                        request.method,
                        request.scheme,
                        request.full_path,
                        input_json['prompt'],
                        input_json['n_images'],
                        input_json['type']
                        )
                up_samples = sample_model(
                        input_json['prompt'],
                        input_json['n_images'],
                        guidance_scale,
                        upsample_temp,
                        model_100,
                        model_up_100,
                        diffusion_100,
                        diffusion_up_100,
                        options_100,
                        options_up_100,
                        device
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

            encoded_images = [Image.fromarray(i) for i in up_samples]
            names = []
            url_paths = []
            SAVE_DIR = '/images'
            for ind,i in enumerate(encoded_images):
                names.append('{}.jpg'.format(ind))
                i.save(os.path.join(SAVE_DIR,names[ind]))
                url_paths.append('https://dalleapi.com/static/{}'.format(names[ind]))
            # encoded_imges = []
            # for image_path in range(10):
            #     encoded_imges.append(get_response_image_test())
            return jsonify({'prompt':input_json['prompt'],
                            'n_images':input_json['n_images'],
                            'result': url_paths})
        except Exception as e:
            print(e)
            timestamp = strftime('%Y-%b-%d-%H:%M')

            app.logger.info('%s,%s,%s,%s,%s,%s,%s,%s',
                        timestamp,
                        request.remote_addr,
                        request.method,
                        request.scheme,
                        request.full_path,
                        input_json['prompt'],
                        input_json['n_images'],
                        "ERROR"
                        )
            return jsonify({"sorry": "Sorry, no results! Please try again."}), 500

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
    th.multiprocessing.set_start_method('spawn', force=True)
    handler = RotatingFileHandler('app.csv', maxBytes=100000, backupCount=3)
    logger = logging.getLogger('tdm')
    app.logger.setLevel(logging.INFO)
    app.logger.addHandler(handler)
    app.logger.info("timestamp,request.remote_addr,request.method,request.scheme,request.full_path,prompt,n_images,type")
    # app.run(
    #     host="0.0.0.0",
    #     port=5000
    # )
    serve(app,host='0.0.0.0',port=5000,threads=1)