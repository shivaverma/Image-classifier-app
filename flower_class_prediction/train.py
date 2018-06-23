# Imports here
import json
import torch
import argparse
from torch import nn
from torch import optim
from predict import load_checkpoint
from collections import OrderedDict
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


def prepare_dataloaders():

    data_dir = 'flower_data'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define your transforms for the training, validation, and testing sets
    # Rotating, randomly cropping, flipping and normalizing the image
    data_transforms = transforms.Compose([transforms.RandomRotation(30), transforms.Resize(300),
                                          transforms.RandomCrop(224),
                                          transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=data_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=data_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=data_transforms)

    # Using the image datasets and the transforms, define the dataloaders
    # data will be shuffled for each epoch, 1 images from batch of five in test data
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=20, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)

    print("Total training images are " + str(len(trainloader)))
    print("Total validation images are " + str(len(validloader)))
    print("Total testing images are " + str(len(testloader)), "\n")
    return trainloader, validloader, testloader


def create_model(args):

    hidden_units = [x for x in args.hidden_units]

    if args.arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        hidden_units = [1024] + hidden_units
    else:
        model = models.vgg16(pretrained=True)
        hidden_units = [25088] + hidden_units

    # Freeze the parameters of densenet so that losses does not back propagate
    for param in model.parameters():
        param.requires_grad = False

    # Building and training network
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

    print("Layers: ", OrderedDict(classier_net), "\n")
    classifier = nn.Sequential(OrderedDict(classier_net))
    model.classifier = classifier
    return model


def train_network(args):

    if args.res_training:
        model, _, _ = load_checkpoint(args)
    else:
        model = create_model(args)

    if torch.cuda.is_available():
        model.cuda()
        print('Training model on GPU', "\n")

    loss_fn = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)

    steps = 0
    running_loss = 0
    validation_frequency = 30

    trainloader, validloader, testloader = prepare_dataloaders()
    try:
        for e in range(args.epochs):

            for images, labels in iter(trainloader):

                steps += 1
                inputs = Variable(images)
                inputs = inputs.cuda()
                targets = Variable(labels)
                targets = targets.cuda()
                optimizer.zero_grad()
                output = model.forward(inputs)
                loss = loss_fn(output, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.data

                if steps % validation_frequency == 0:

                    # model in evaluation mode
                    model.eval()
                    accuracy = 0
                    val_loss = 0
                    for ii, (image, label) in enumerate(validloader):

                        inp = Variable(image)
                        inp = inp.cuda()
                        label = Variable(label)
                        label = label.cuda()
                        out = model.forward(inp)
                        val_loss += loss_fn(out, label).data
                        ps = torch.exp(out).data
                        equality = (label.data == ps.max(1)[1])
                        accuracy += equality.type_as(torch.FloatTensor()).mean()
                        # print(ii + 1, "image validation done")

                    print("Training Loss: {:.3f}.. ".format(running_loss / validation_frequency),
                          "Validation Loss: {:.3f}.. ".format(val_loss / len(validloader)),
                          "Validation Accuracy: {:.3f}".format(accuracy / len(validloader)))

                    running_loss = 0
                    model.train()

                # print(str(steps) + " Image trained")

    except KeyboardInterrupt:
        pass

    print('Starting Validation on Test Set')
    # validation on the test set
    model.eval()

    accuracy = 0
    test_loss = 0
    for ii, (images, labels) in enumerate(testloader):

        inputs = Variable(images)
        inputs = inputs.cuda()
        labels = Variable(labels)
        labels = labels.cuda()
        output = model.forward(inputs)
        test_loss += loss_fn(output, labels).data
        ps = torch.exp(output).data
        equality = (labels.data == ps.max(1)[1])
        print("this: "+ str(equality.type_as(torch.FloatTensor()).mean()))
        accuracy += equality.type_as(torch.FloatTensor()).mean()
        print(ii, "Images tested")

    print("Test Loss: {:.3f}.. ".format(test_loss / len(testloader)),
          "Test Accuracy: {:.3f}".format(accuracy / len(testloader)))

    # Save the model
    checkpoint = {'lr': args.lr,
                  'arch': args.arch,
                  'output_size': 102,
                  'input_size': [3, 224, 224],
                  'state_dict': model.state_dict(),
                  'hidden_units': args.hidden_units,
                  'batch_size': trainloader.batch_size,
                  'optimizer_dict': optimizer.state_dict(),
                  'class_to_idx': trainloader.dataset.class_to_idx
                  }

    print('Saving Model')
    torch.save(checkpoint, 'saved_model')


def main():

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--res_training', type=bool, default=False, help='Train from checkpoint')
    parser.add_argument('--arch', type=str, default='vgg16', help='vgg16 or densenet121')
    parser.add_argument('--lr', type=float, default=0.001, help='What is learning rate?')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs')
    parser.add_argument('--hidden_units', type=list, default=[500], help='hidden units for fc')
    parser.add_argument('--mapping', type=str, default='cat_to_name.json', help='mapping file')
    args = parser.parse_args()
    train_network(args)


if __name__ == "__main__":

    main()
