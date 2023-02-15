import timm
import albumentations as A
import albumentations.pytorch


def build_model(config):
    # set up model name
    if config['model_name'].startswith('swin'):
        model_name = f"{config['model_name']}_patch4_window7_224"
    elif config['model_name'].startswith('vit'):
        model_name = f"{config['model_name']}_patch16_224"
    elif config['model_name'].startswith('resnet'):
        model_name = config['model_name']
    else:
        raise NotImplementedError('Unknown model')

    # create model
    model = timm.create_model(model_name, pretrained=config['pretrained'], num_classes=1)

    # freeze model
    if config['freeze']:
        submodules = [n for n, _ in model.named_children()]
        timm.freeze(model, submodules[:submodules.index(config['freeze_until']) + 1])

    return model


def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)

    return result


def get_transformations():
    data_transforms = {
        'train': A.Compose([
            A.Rotate(limit=30),
            A.RandomResizedCrop(224, 224, ratio=(1.0, 1.0), scale=(0.9, 1.0)),
            A.HorizontalFlip(),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            A.pytorch.transforms.ToTensorV2()
        ]),
        'valid': A.Compose([
            A.Resize(256, 256),
            A.CenterCrop(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            A.pytorch.transforms.ToTensorV2()
        ]),
    }

    return data_transforms
