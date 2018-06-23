# Imports here
import torch
from torch import nn
from train import  prepare_dataloaders
from predict import load_checkpoint
from torch.autograd import Variable


def test_network(a=1):

    model, _, _ = load_checkpoint(a)

    if torch.cuda.is_available():
        model.cuda()
        print('Training model on GPU', "\n")

    loss_fn = nn.NLLLoss()

    _, _, testloader = prepare_dataloaders()

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
        accuracy += equality.type_as(torch.FloatTensor()).mean()
        print(ii, "Images tested")

    print("Test Loss: {:.3f}.. ".format(test_loss / len(testloader)),
          "Test Accuracy: {:.3f}".format(accuracy / len(testloader)))


if __name__ == "__main__":

    test_network()
