import timm


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
