# coding=utf-8
# Created by Meteorix at 2019/8/9
from flask import Flask, jsonify, request
from service_streamer import ThreadedStreamer
from model import load_models, sample_model
import torch as th

app = Flask(__name__)
# batch_size = 10
guidance_scale = 3.0

# Tune this parameter to control the sharpness of 256x256 images.
# A value of 1.0 is sharper, but sometimes results in grainy artifacts.
upsample_temp = 0.997
has_cuda = th.cuda.is_available()
device = th.device('cpu' if not has_cuda else 'cuda')
options, options_up,model,model_up,diffusion, diffusion_up = load_models(has_cuda,
                                                                         device,
                                                                         timestep_respacing='25',
                                                                         timestep_respacing_up='fast27')

streamer = ThreadedStreamer(sample_model, batch_size=1,max_latency=30)


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
        up_samples = up_samples[0]
        print(up_samples)
        return jsonify({'done': 'done'})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5005)