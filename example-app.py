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

streamer = ThreadedStreamer(sample_model(
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
                        device), batch_size=1)


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        class_id, class_name = get_prediction(img_bytes)
        return jsonify({'class_id': class_id, 'class_name': class_name})


@app.route('/stream_predict', methods=['POST'])
def stream_predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        class_id, class_name = streamer.predict([img_bytes])[0]
        return jsonify({'class_id': class_id, 'class_name': class_name})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5005)