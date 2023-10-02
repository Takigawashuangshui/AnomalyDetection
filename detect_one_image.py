#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import tifffile
import torch
from torchvision import transforms
import os
import random
from PIL import Image

def load_image(path):
    image = Image.open(path).convert(mode='RGB')
    return image


def perform(image, subdataset, path, output_dir, model_size):
    # constants
    seed = 42
    image_size = 256
    iheight, iweight = image.height, image.width

    # image pre-processing 
    default_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = default_transform(image)
    image = image[None, :]

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # load models
    if model_size == 'small':
        teacher = torch.load('output/1/trainings/mvtec_ad/'+subdataset+'/teacher_final.pth')
        student = torch.load('output/1/trainings/mvtec_ad/'+subdataset+'/student_final.pth')
    elif model_size == 'medium':
        teacher = torch.load('output/2/trainings/mvtec_ad/'+subdataset+'/teacher_final.pth')
        student = torch.load('output/2/trainings/mvtec_ad/'+subdataset+'/student_final.pth')
    else:
        raise Exception()
    autoencoder = torch.load('output/1/trainings/mvtec_ad/'+subdataset+'/autoencoder_final.pth')
    if torch.cuda.is_available():
        teacher.cuda()
        student.cuda()
        autoencoder.cuda()
    teacher_mean, teacher_std = teacher_normalization(teacher, image)
    teacher.eval()
    student.eval()
    autoencoder.eval()

    #perform anomaly detection on sample image
    q_st_start, q_st_end, q_ae_start, q_ae_end = map_normalization(
        image=image, teacher=teacher, student=student,
        autoencoder=autoencoder, teacher_mean=teacher_mean,
        teacher_std=teacher_std, desc='Final map normalization')
    map = test(
        image=image, path=path, teacher=teacher, student=student,
        autoencoder=autoencoder, teacher_mean=teacher_mean,
        teacher_std=teacher_std, q_st_start=q_st_start, q_st_end=q_st_end,
        q_ae_start=q_ae_start, q_ae_end=q_ae_end,
        h=iheight, w=iweight, desc='Final inference')
    return map

def save_map(path, test_output_dir, map):
    defect_class = os.path.basename(os.path.dirname(path))
    if test_output_dir is not None:
        img_nm = os.path.split(path)[1].split('.')[0]
        if not os.path.exists(os.path.join(test_output_dir, defect_class)):
                os.makedirs(os.path.join(test_output_dir, defect_class))
        file = os.path.join(test_output_dir, defect_class, img_nm + '.tiff')
        tifffile.imwrite(file, map)


def test(image, path, teacher, student, autoencoder, teacher_mean, teacher_std, 
         q_st_start, q_st_end, q_ae_start, q_ae_end, h, w, desc='Running inference'):
    orig_height, orig_width = h, w
    if torch.cuda.is_available():
            image = image.cuda()
    map_combined, map_st, map_ae = predict(
            image=image, teacher=teacher, student=student,
            autoencoder=autoencoder, teacher_mean=teacher_mean,
            teacher_std=teacher_std, q_st_start=q_st_start, q_st_end=q_st_end,
            q_ae_start=q_ae_start, q_ae_end=q_ae_end)
    map_combined = torch.nn.functional.pad(map_combined, (4, 4, 4, 4))
    map_combined = torch.nn.functional.interpolate(
            map_combined, (orig_height, orig_width), mode='bilinear')
    map_combined = map_combined[0, 0].cpu().numpy()
    return map_combined

@torch.no_grad()
def predict(image, teacher, student, autoencoder, teacher_mean, teacher_std,
            q_st_start=None, q_st_end=None, q_ae_start=None, q_ae_end=None):
    out_channels = 384
    teacher_output = teacher(image)
    teacher_output = (teacher_output - teacher_mean) / teacher_std
    student_output = student(image)
    autoencoder_output = autoencoder(image)
    map_st = torch.mean((teacher_output - student_output[:, :out_channels])**2,
                        dim=1, keepdim=True)
    map_ae = torch.mean((autoencoder_output -
                         student_output[:, out_channels:])**2,
                        dim=1, keepdim=True)
    if q_st_start is not None:
        map_st = 0.1 * (map_st - q_st_start) / (q_st_end - q_st_start)
    if q_ae_start is not None:
        map_ae = 0.1 * (map_ae - q_ae_start) / (q_ae_end - q_ae_start)
    map_combined = 0.5 * map_st + 0.5 * map_ae
    return map_combined, map_st, map_ae

@torch.no_grad()
def map_normalization(image, teacher, student, autoencoder,
                      teacher_mean, teacher_std, desc='Map normalization'):
    maps_st = []
    maps_ae = []
    # ignore augmented ae image
    if torch.cuda.is_available():
        image = image.cuda()
    map_combined, map_st, map_ae = predict(
        image=image, teacher=teacher, student=student,
        autoencoder=autoencoder, teacher_mean=teacher_mean,
        teacher_std=teacher_std)
    maps_st.append(map_st)
    maps_ae.append(map_ae)
    maps_st = torch.cat(maps_st)
    maps_ae = torch.cat(maps_ae)
    q_st_start = torch.quantile(maps_st, q=0.9)
    q_st_end = torch.quantile(maps_st, q=0.995)
    q_ae_start = torch.quantile(maps_ae, q=0.9)
    q_ae_end = torch.quantile(maps_ae, q=0.995)
    return q_st_start, q_st_end, q_ae_start, q_ae_end

@torch.no_grad()
def teacher_normalization(teacher, train_image):
    mean_outputs = []
    if torch.cuda.is_available():
        train_image = train_image.cuda()
    teacher_output = teacher(train_image)
    mean_output = torch.mean(teacher_output, dim=[0, 2, 3])
    mean_outputs.append(mean_output)
    channel_mean = torch.mean(torch.stack(mean_outputs), dim=0)
    channel_mean = channel_mean[None, :, None, None]

    mean_distances = []
    if torch.cuda.is_available():
        train_image = train_image.cuda()
    teacher_output = teacher(train_image)
    distance = (teacher_output - channel_mean) ** 2
    mean_distance = torch.mean(distance, dim=[0, 2, 3])
    mean_distances.append(mean_distance)
    channel_var = torch.mean(torch.stack(mean_distances), dim=0)
    channel_var = channel_var[None, :, None, None]
    channel_std = torch.sqrt(channel_var)

    return channel_mean, channel_std
