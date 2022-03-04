import os
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
from models.CNNv4 import Model, transform

classes = {
    0: "airplane", 
    1: "automobile", 
    2: "bird", 
    3: "cat", 
    4: "deer", 
    5: "dog", 
    6: "frog", 
    7: "horse", 
    8: "ship", 
    9: "truck"
}

model = Model()
model.load_state_dict(torch.load('model.pt'))
model.to("cuda")
model.eval()

ids = []
targets = []

file_names = os.listdir(os.path.join('data', 'test'))
pbar = tqdm(enumerate(file_names), total=len(file_names))
for it, file_name in pbar:
    file_path = os.path.join('data', 'test', file_name)

    img = Image.open(file_path)
    img = transform(img)
    img = img[None, :, :, :]
    img = img.to("cuda")

    probs = model(img)
    argmax = torch.argmax(probs).item()

    ids.append(file_name)
    targets.append(classes[argmax])

df = pd.DataFrame({'id': ids, 'target': targets})
df.to_csv('submission.csv', index=False)
