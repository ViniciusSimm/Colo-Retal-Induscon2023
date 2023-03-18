import tensorflow as tf
import pandas as pd
import numpy as np
import os
import cv2
from complement_functions import *

IMG_HEIGHT = 288
IMG_WIDTH= 384
IMG_CHANNELS = 3
num_classes = 3

images_train, images_test, masks_train, masks_test = get_folders(['CVC-ClinicDB','Kvasir-SEG'],0.2)
X = get_files(images_train)
y = get_files(masks_train)
X_v = get_files(images_test)
y_v = get_files(masks_test)

# def block_conv(in_layer,filters,kernel=(3,3),padding='same',dropout=0.1):
#         c1 = tf.keras.layers.Conv2D(filters, kernel, activation='relu', kernel_initializer='he_normal', padding=padding)(in_layer)
#         c1 = tf.keras.layers.Dropout(dropout)(c1)
#         c1 = tf.keras.layers.Conv2D(filters, kernel, activation='relu', kernel_initializer='he_normal', padding=padding)(c1)
#         b1 = tf.keras.layers.BatchNormalization()(c1)
#         r1 = tf.keras.layers.ReLU()(b1)
#         return r1

# def block_up(in_layer,pair_block,filters,matrix=(2,2), strides=(2, 2), padding='same'):
#         u6 = tf.keras.layers.Conv2DTranspose(filters, matrix, strides=strides, padding=padding)(in_layer)
#         u6 = tf.keras.layers.concatenate([u6, pair_block])
#         u6 = tf.keras.layers.BatchNormalization()(u6)
#         u6 = tf.keras.layers.ReLU()(u6)
#         return u6


# r1 = block_conv(input,16)
# p1 = tf.keras.layers.MaxPooling2D((2, 2))(r1)
# r2 = block_conv(p1,32)
# p2 = tf.keras.layers.MaxPooling2D((2, 2))(r2)
# r3 = block_conv(p2,64)
# p3 = tf.keras.layers.MaxPooling2D((2, 2))(r3)
# r4 = block_conv(p3,128)
# p4 = tf.keras.layers.MaxPooling2D((2, 2))(r4)

# c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
# b5 = tf.keras.layers.BatchNormalization()(c5)
# r5 = tf.keras.layers.ReLU()(b5)
# c5 = tf.keras.layers.Dropout(0.3)(r5)
# c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

# u6 = block_up(c5,r4,filters,matrix=(2,2), strides=(2, 2), padding='same')

inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

#Contraction path
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
b1 = tf.keras.layers.BatchNormalization()(c1)
r1 = tf.keras.layers.ReLU()(b1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(r1)

c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
b2 = tf.keras.layers.BatchNormalization()(c2)
r2 = tf.keras.layers.ReLU()(b2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(r2)
 
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
b3 = tf.keras.layers.BatchNormalization()(c3)
r3 = tf.keras.layers.ReLU()(b3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(r3)
 
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
b4 = tf.keras.layers.BatchNormalization()(c4)
r4 = tf.keras.layers.ReLU()(b4)
p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(r4)
 
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
b5 = tf.keras.layers.BatchNormalization()(c5)
r5 = tf.keras.layers.ReLU()(b5)
c5 = tf.keras.layers.Dropout(0.3)(r5)
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

#Expansive path 
u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6, c4])
u6 = tf.keras.layers.BatchNormalization()(u6)
u6 = tf.keras.layers.ReLU()(u6)

 
u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(u6)
u7 = tf.keras.layers.concatenate([u7, c3])
u7 = tf.keras.layers.BatchNormalization()(u7)
u7 = tf.keras.layers.ReLU()(u7)

 
u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(u7)
u8 = tf.keras.layers.concatenate([u8, c2])
u8 = tf.keras.layers.BatchNormalization()(u8)
u8 = tf.keras.layers.ReLU()(u8)
 
u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(u8)
u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
u9 = tf.keras.layers.BatchNormalization()(u9)
u9 = tf.keras.layers.ReLU()(u9)

 
outputs = tf.keras.layers.Conv2D(num_classes, (1, 1), activation='sigmoid')(u9)

model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
        tf.keras.callbacks.TensorBoard(log_dir='logs')]

model.fit(X, y, validation_data=(X_v,y_v), batch_size=16, epochs=5, callbacks=callbacks)



# import torch 
# import torch.nn as nn

# class double_conv(nn.Module):
#     """(conv => BN => ReLU) * 2"""

#     def __init__(self, in_ch, out_ch):
#         super(double_conv, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, 3, padding=1),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#             # nn.Conv2d(out_ch, out_ch, 3, padding=1),
#             # nn.BatchNorm2d(out_ch),
#             # nn.ReLU(inplace=True),
#         )

#     def forward(self, x):
#         x = self.conv(x)
#         return x

# class inconv(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(inconv, self).__init__()
#         self.conv = double_conv(in_ch, out_ch)

#     def forward(self, x):
#         x = self.conv(x)
#         return x

# class down(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(down, self).__init__()
#         self.mpconv = nn.Sequential(nn.MaxPool2d(2), double_conv(in_ch, out_ch))

#     def forward(self, x):
#         x = self.mpconv(x)
#         return x


# class up(nn.Module):
#     def __init__(self, in_ch, out_ch, bilinear=True):
#         super(up, self).__init__()

#         if bilinear:
#             self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
#         else:
#             self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

#         self.conv = double_conv(in_ch, out_ch)

#     def forward(self, x1, x2):
#         x1 = self.up(x1)

#         # input is CHW
#         diffY = x2.size()[2] - x1.size()[2]
#         diffX = x2.size()[3] - x1.size()[3]

#         x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        
#         x = torch.cat([x2, x1], dim=1)
#         return self.conv(x)

# class outconv(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(outconv, self).__init__()
#         self.conv = nn.Conv2d(in_ch, out_ch, 1)

#     def forward(self, x):
#         x = self.conv(x)
#         return x


# class UNet(nn.Module):
#     def __init__(self, n_channels, n_classes):
#         super(UNet, self).__init__()
#         self.inc = inconv(n_channels, 64)
#         self.down1 = down(64, 128)
#         self.down2 = down(128, 256)
#         self.down3 = down(256, 512)
#         self.down4 = down(512, 512)
#         self.up1 = up(1024, 256, False)
#         self.up2 = up(512, 128, False)
#         self.up3 = up(256, 64, False)
#         self.up4 = up(128, 64, False)
#         self.outc = outconv(64, n_classes)

#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         x = self.outc(x)
#         return torch.sigmoid(x)
    
# model = UNet(n_channels=3, n_classes=4).float()
# model.cuda()
# print(model)

