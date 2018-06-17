# Imports here
import json
import os
import random
import torch
import argparse
import numpy as np
from torch import nn
from PIL import Image
from matplotlib import pyplot
from torchvision import models
from torch.autograd import Variable
from collections import OrderedDict

# function that loads a checkpoint and rebuilds the model


def load_checkpoint(args):

    checkpoint = torch.load('saved_model')
    hidden_units = checkpoint['hidden_units']

    if checkpoint['arch'] == 'densenet121':
        model = models.densenet121()
        hidden_units = [1024] + hidden_units
    else:
        model = models.vgg16()
        hidden_units = [25088] + hidden_units

    # build the classifier part of model
    classier_net = []
    hidden_units.append(102)

    pair = []
    for i in range(len(hidden_units)-1):
        pair.append((hidden_units[i], hidden_units[i+1]))

    for i, x in enumerate(pair):
        classier_net.append(('fc' + str(i + 1), nn.Linear(x[0], x[1])))
        if i != len(pair):
            classier_net.append(('relu', nn.ReLU()))

    classier_net.append(('output', nn.LogSoftmax(dim=1)))
    classifier = nn.Sequential(OrderedDict(classier_net))
    model.classifier = classifier

    model.load_state_dict(checkpoint['state_dict'])

    # Check whether to train on gpu or not
    if torch.cuda.is_available():
        model.cuda()
        print('Running on GPU')
    else:
        print('Running on CPU')

    class_to_idx = checkpoint['class_to_idx']
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return model, class_to_idx, idx_to_class


def process_image(image):

    # TODO: Process a PIL image for use in a PyTorch model
    size = 256, 256
    image.thumbnail(size, Image.ANTIALIAS)
    image = image.crop((size[0] / 2 - 112, size[1] / 2 - 112, size[0] / 2 + 112, size[1] / 2 + 112))
    image = np.array(image)
    image = image / 255.

    img_0 = image[:, :, 0]                      # Scaling image
    img_1 = image[:, :, 1]
    img_2 = image[:, :, 2]

    img_0 = (img_0 - 0.485)/0.229                  # Normalizing image
    img_1 = (img_1 - 0.456)/0.224
    img_2 = (img_2 - 0.406)/0.225

    image[:, :, 0] = img_0
    image[:, :, 1] = img_1
    image[:, :, 2] = img_2

    image = np.transpose(image, (2, 0, 1))   # Transposing image
    return image


def predict(image_path, model, idx_to_class, cat_to_name, topk=5):

    # TODO: Implement the code to predict the class from an image file

    image = Image.open(image_path)
    image = process_image(image)
    image = torch.FloatTensor([image])
    image = image.cuda()

    model.eval()    # Evaluation mode

    output = model.forward(Variable(image))
    ps = torch.exp(output).data.cuda()
    ps = ps.cpu().numpy()[0]

    topk_index = np.argsort(ps)[-topk:][::-1]
    topk_class = [idx_to_class[x] for x in topk_index]
    name = [cat_to_name[x] for x in topk_class]
    topk = ps[topk_index]

    return topk, name


def create_plot(cl, pb, path, maps):

    image = Image.open(path)

    fig, (ax1, ax2) = pyplot.subplots(figsize=(9, 8), ncols=1, nrows=2)
    flower_name = maps[path.split('/')[-2]]
    ax1.set_title(flower_name)
    ax1.imshow(image)
    ax1.axis('off')

    y_pos = np.arange(len(pb))
    ax2.barh(y_pos, pb, align='center')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([x for x in cl])
    ax2.invert_yaxis()                         # labels read top-to-bottom
    ax2.set_title('Class Probability')
    pyplot.show()


def main(temp):

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--gpu', type=bool, default=True, help='Want to use GPU')
    parser.add_argument('--topk', type=int, default=5, help='Top K probabilities')
    parser.add_argument('--image_path', type=str, default=temp, help='Path of image to predicted')
    parser.add_argument('--mapping', type=str, default='cat_to_name.json', help='mapping file')

    args = parser.parse_args()

    with open(args.mapping, 'r') as f:
        cat_to_name = json.load(f)

    model, class_to_idx, idx_to_class = load_checkpoint(args)
    topk_prob, named_topk_class = predict(args.image_path, model, idx_to_class, cat_to_name, topk=args.topk)

    print('Top 5 predicted classes: ', named_topk_class)
    print('Respective probabilities: ', topk_prob)

    create_plot(named_topk_class, topk_prob, args.image_path, cat_to_name)


if __name__ == "__main__":

    test_set = 10

    for i in range(test_set):
        path = 'flower_data/test/'
        rand = np.random.random_integers(1, 101)
        path = path + str(rand) + '/'
        listing = os.listdir(path)
        list_fl = [x for x in listing]
        file = random.choice(list_fl)
        main(path+file)
