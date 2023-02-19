import numpy as np
import torch


def scale_cam(cam, size):
    cam = (cam - cam.min()) / (1e-7 + cam.max())
    cam = cam[None, :, :, :]
    cam = torch.nn.functional.interpolate(cam, size, mode='bilinear')
    cam = cam.squeeze()

    return cam


def get_target_width_height(input_tensor):
    width, height = input_tensor.size(-1), input_tensor.size(-2)

    return width, height


class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targeted intermediate layers """

    def __init__(self, model, target_layer, reshape_transform):
        self.model = model
        self.gradient = None
        self.activation = None
        self.reshape_transform = reshape_transform
        self.handles = []

        self.handles.append(target_layer.register_forward_hook(self.save_activation))
        self.handles.append(target_layer.register_forward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output

        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)

        self.activation = activation

    def save_gradient(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # You can only register hooks on tensor requires grad.
            return

        # Gradients are computed in reverse order
        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            self.gradient = grad

        output.register_hook(_store_grad)

    def __call__(self, x):
        self.gradient = None
        self.activation = None

        return self.model(x)

    def release(self):
        for handle in self.handles:
            handle.remove()


class GCAM:
    def __init__(self, model, target_layer, use_cuda, reshape_transform=None, compute_input_gradient=False,
                 uses_gradients=True):
        self.model = model.eval()
        self.target_layer = target_layer
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.reshape_transform = reshape_transform
        self.compute_input_gradient = compute_input_gradient
        self.uses_gradients = uses_gradients
        self.activations_and_grads = ActivationsAndGradients(self.model, target_layer, reshape_transform)

    def get_cam_weights(self, input_tensor, target_layer, targets, activation, grad):
        raise Exception("Not Implemented")

    def get_cam_image(self, activation, grad):
        elementwise_activations = torch.maximum(grad * activation, torch.zeros(grad.shape, device=grad.device))
        cam = elementwise_activations.sum(axis=1)

        return cam

    def forward(self, input_tensor, targets):

        if self.cuda:
            input_tensor = input_tensor.cuda()

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor, requires_grad=True)

        outputs = self.activations_and_grads(input_tensor).reshape(-1)

        # all values in targets that are 1 will stay 1, but 0 will become -1
        targets = targets * 2 - 1

        if self.uses_gradients:
            self.model.zero_grad()
            loss = sum(targets * outputs)
            loss.backward(retain_graph=True)

        cam = self.get_cam_image(self.activations_and_grads.activation, self.activations_and_grads.gradient)
        cam = torch.maximum(cam, torch.zeros(cam.shape, device=cam.device))
        target_size = get_target_width_height(input_tensor)
        cam = scale_cam(cam, target_size)

        return cam

    def __call__(self, input_tensor, targets):
        return self.forward(input_tensor, targets)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True
