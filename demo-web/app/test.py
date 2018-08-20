from models import generator, discriminator, GenNucleiDataset
import torch



G = generator()
D = discriminator()

G = torch.load('./app/weights/G_model.pth', map_location={'cuda:0': 'cpu'})
D = torch.load('./app/weights/D_model.pth', map_location={'cuda:0': 'cpu'})

print(G)