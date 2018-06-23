import ast
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models

resnet18 = models.resnet18(pretrained=True)
alexnet = models.alexnet(pretrained=True)
vgg16 = models.vgg16(pretrained=True)

my_models = {'resnet': resnet18, 'alexnet': alexnet, 'vgg': vgg16}

# obtain ImageNet labels

with open('imagenet1000_clsid_to_human.txt') as imagenet_classes_file:
    imagenet_classes_dict = ast.literal_eval(imagenet_classes_file.read())


def classifier(img_path, model_name):

    img_pil = Image.open(img_path)

    # define transforms
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # preprocess the image
    img_tensor = preprocess(img_pil)

    # resize the tensor (add dimension for batch)
    img_tensor.unsqueeze_(0)
    
    # pytorch versions 0.4 & higher - Variable depreciated so that it returns
    # a tensor. So to address tensor as output (not wrapper) and to mimic the
    # affect of setting volatile = True (because we are using pertained models
    # for inference) we can set requires_gradient to False. Here we just set
    # requires_grad_ to False on our tensor

    img_tensor.requires_grad_(False)

    # apply model to input
    model = my_models[model_name]

    # puts model in evaluation mode
    # instead of (default)training mode
    model = model.eval()
    output = model(img_tensor)

    # return index corresponding to predicted class
    pred_idx = output.data.numpy().argmax()

    return imagenet_classes_dict[pred_idx]
