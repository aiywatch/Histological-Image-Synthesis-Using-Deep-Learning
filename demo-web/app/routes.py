
from flask import render_template
from app import app
from flask import Flask, render_template, request, send_from_directory
from flask.ext.uploads import UploadSet, configure_uploads, IMAGES

from app.models import generator, discriminator, GenNucleiDataset

import torch
import os, time, pickle, argparse
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt
import scipy.misc

from multiprocessing import Value

counter = Value('i', 0)



@app.route('/')
@app.route('/index')
def index():

    return render_template('index.html')


photos = UploadSet('photos', IMAGES)

app.config['UPLOADED_PHOTOS_DEST'] = 'static/img'
configure_uploads(app, photos)


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST' and 'photo' in request.files:
        filename_mask = photos.save(request.files['photo'])
        mask = imread('./static/img/'+filename_mask)

        G = generator()
        D = discriminator()
        G = torch.load('./app/weights/G_model.pth', map_location={'cuda:0': 'cpu'})
        D = torch.load('./app/weights/D_model.pth', map_location={'cuda:0': 'cpu'})


        mask = resize(mask, (256, 256))
        mask = torch.Tensor(mask).view(-1, 256, 256)

        option_layer1 = torch.ones((1, mask.shape[1], mask.shape[2]))
        option_layer2 = torch.ones((1, mask.shape[1], mask.shape[2])) * 2

        mask1 = torch.cat([mask, option_layer1], 0)
        mask2 = torch.cat([mask, option_layer2], 0)

        out1 = G(mask1.view(1,2,256,256))[0,:,:,:].permute(1,2,0).cpu().detach().numpy()
        out1[out1<0] = 0

        out2 = G(mask2.view(1,2,256,256))[0,:,:,:].permute(1,2,0).cpu().detach().numpy()
        out2[out2<0] = 0

        out1_filename = f'out1_{counter.value}.jpg'
        out2_filename = f'out2_{counter.value}.jpg'
        with counter.get_lock():
            counter.value += 1

        scipy.misc.imsave(f'./static/img/{out1_filename}', out1)
        scipy.misc.imsave(f'./static/img/{out2_filename}', out2)

        return render_template('show_results.html', model_g=G, mask=filename_mask, out1=out1_filename, out2=out2_filename)
    return render_template('upload.html')

@app.route('/uploads/<filename>')
def send_file(filename):
    # os.chdir(os.path.dirname(__file__))
    current_dir = os.getcwd()
    return send_from_directory(current_dir+'/static/img', filename)

@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

if __name__ == '__main__':
    app.run(debug=True)
