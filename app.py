import streamlit as st
import numpy as np
from PIL import Image
import torch
from torch.nn import DataParallel
import torch.nn as nn
import albumentations as A

# Hyper Params
INPUT_IMG_SIZE = 128
OUTPUT_CLASSES = 12

# Model
class SegNet(nn.Module):
    def __init__(self, num_classes):
        super(SegNet, self).__init__()
        self.num_classes = num_classes
        self.contracting_11 = self.conv_block(in_channels=3, out_channels=64)
        self.contracting_12 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.contracting_21 = self.conv_block(in_channels=64, out_channels=128)
        self.contracting_22 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.contracting_31 = self.conv_block(in_channels=128, out_channels=256)
        self.contracting_32 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.middle = self.conv_block(in_channels=256, out_channels=1024)
        self.expansive_11 = nn.ConvTranspose2d(in_channels=1024, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_12 = self.conv_block(in_channels=512, out_channels=256)
        self.expansive_21 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_22 = self.conv_block(in_channels=256, out_channels=128)
        self.expansive_31 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_32 = self.conv_block(in_channels=128, out_channels=64)
        self.output = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=3, stride=1, padding=1)




    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(num_features=out_channels),
                                    nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(num_features=out_channels))
        return block

    def forward(self, X):

        contracting_11_out = self.contracting_11(X)
        contracting_12_out = self.contracting_12(contracting_11_out)
        contracting_21_out = self.contracting_21(contracting_12_out)
        contracting_22_out = self.contracting_22(contracting_21_out)
        contracting_31_out = self.contracting_31(contracting_22_out)
        contracting_32_out = self.contracting_32(contracting_31_out)

        middle_out = self.middle(contracting_32_out)

        expansive_11_out = self.expansive_11(middle_out)
        expansive_12_out = self.expansive_12(torch.cat((expansive_11_out, contracting_31_out), dim=1))
        expansive_21_out = self.expansive_21(expansive_12_out)
        expansive_22_out = self.expansive_22(torch.cat((expansive_21_out, contracting_21_out), dim=1))
        expansive_31_out = self.expansive_31(expansive_22_out)
        expansive_32_out = self.expansive_32(torch.cat((expansive_31_out, contracting_11_out), dim=1))

        output_out = self.output(expansive_32_out)
        return output_out

# Load model
model = SegNet(OUTPUT_CLASSES)
model = DataParallel(model)
model.load_state_dict(torch.load('results/checkpoint.ckp')['state_dict'])

model = model.module
model.cpu()
model.eval()

transform = A.Compose([A.Resize(INPUT_IMG_SIZE, INPUT_IMG_SIZE),
                       A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])])

# Decorate the app
st.title("Semantic Segmentation")
st.write("This is a simple web app to predict the semantic segmentation of Cityscapes Dataset.")
st.write("The model is trained on 12 classes of Cityscapes Dataset.")

# Add GIF
st.header("A GIF to show how SemSeg works.")
st.image('https://miro.medium.com/v2/resize:fit:1100/1*kvh9u8W2sHlQoBPfwERggA.gif', use_column_width=True)

st.header("Upload an image to predict the Segmentation Mask.")
# Load image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    st.write("Done!")
    image = transform(image=np.array(image))['image']
    image = np.array(image)
    image = image.transpose(2, 0, 1)
    image = torch.from_numpy(image).float().unsqueeze(0)
    image = image.cpu()
    pred = model(image)
    p = pred[0].permute(1,2,0)
    p = torch.argmax(p, dim=2)
    p = p.reshape(INPUT_IMG_SIZE, INPUT_IMG_SIZE).cpu().numpy()*20
    st.image(p, caption='Prediction.', use_column_width=True)

# EOF
