#!flask/bin/python
from app import app
from app.models import generator, discriminator, GenNucleiDataset


app.run(debug=True)