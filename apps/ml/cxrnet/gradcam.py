from apps.ml.cxrnet.augmentations import get_transform

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image

def gradcam(image, model):

    model_children = list(model.features.children())
    target_layers = [model_children[-2]]

    transform = get_transform(train=False, img_size=224, rotate_degree=0)
    image = transform(image=image)["image"]
    origin = image.numpy().transpose(1,2,0)
    origin = origin - origin.min()
    origin = origin / origin.max()
    image = image.unsqueeze(0)

    input_tensor = image
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)

    targets = None

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(origin, grayscale_cam, use_rgb=True)

    return visualization