# coding=utf-8
# Created by Meteorix at 2019/8/9
from flask import Flask, jsonify, request
from service_streamer import ThreadedStreamer, Streamer
from model import load_models, sample_model
import torch as th
import numpy as np
from PIL import Image
import os
app = Flask(__name__)
# batch_size = 10
guidance_scale = 3.0
SAVE_DIR = '/images'
os.makedirs(SAVE_DIR,exist_ok=True)
# Tune this parameter to control the sharpness of 256x256 images.
# A value of 1.0 is sharper, but sometimes results in grainy artifacts.
upsample_temp = 0.997
has_cuda = th.cuda.is_available()
device = th.device('cpu' if not has_cuda else 'cuda')
options, options_up,model,model_up,diffusion, diffusion_up = load_models(has_cuda,
                                                                         device,
                                                                         timestep_respacing='25',
                                                                         timestep_respacing_up='fast27')
print(os.cpu_count())
streamer = Streamer(sample_model,
                    batch_size=1,
                    max_latency=60*30,
                    worker_num=2,
                    wait_for_worker_ready=True,
                    device=(0),
                    mp_start_method="fork")


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_json = request.get_json(force=True) 
        # img_bytes = file.read()
        up_samples = sample_model(
                        "a cat",
                        6,
                        guidance_scale,
                        upsample_temp,
                        model,
                        model_up,
                        diffusion,
                        diffusion_up,
                        options,
                        options_up,
                        device)
        return jsonify({'done': 'done'})


@app.route('/stream_predict', methods=['POST'])
def stream_predict():
    if request.method == 'POST':
        input_json = request.get_json(force=True) 
        up_samples = streamer.predict(
                        [("a cat",
                        6,
                        guidance_scale,
                        upsample_temp,
                        model,
                        model_up,
                        diffusion,
                        diffusion_up,
                        options,
                        options_up,
                        device)]
                        )
        # up_samples = up_samples[0]
        
        print(up_samples[0][0].shape)
        encoded_images = [Image.fromarray(i) for i in up_samples[0][0]]
        names = []
        url_paths = []

        for ind,i in enumerate(encoded_images):
            names.append('{}.jpg'.format(ind))
            i.save(os.path.join(SAVE_DIR,names[ind]))
            url_paths.append('https://dalleapi.com/static/{}'.format(names[ind]))
        # encoded_imges = []
        # for image_path in range(10):
        #     encoded_imges.append(get_response_image_test())
        s =  jsonify({'prompt':input_json['prompt'],
                        'n_images':input_json['n_images'],
                        'result': url_paths})
        print(s)
        # print(len(up_samples))
        return s


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5005)