import argparse
import cv2
import numpy as np
import torch

from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit
from swin_model.classifier_Swin_I3D import Classifier


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument(
        '--image-path',
        type=str,
        default='./examples/both.png',
        help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
        'of cam_weights*activations')

    parser.add_argument(
        '--method',
        type=str,
        default='gradcam',
        help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


# def reshape_transform(tensor, height=7, width=7):
#     result = tensor.reshape(tensor.size(0),
#                             height, width, tensor.size(2))
#     #TODO: reshape
#     # Bring the channels to the first dimension,
#     # like in CNNs.
#     result = result.transpose(2, 3).transpose(1, 2)
#     return result

# def reshape_transform(tensor, height=7, width=7):
#     # 假设 tensor 的形状为 [BATCH, TIME, HEIGHT, WIDTH, CHANNELS]
#     tensor = tensor.mean(1)  # 形状变为 [BATCH, HEIGHT, WIDTH, CHANNELS]

#     # 接下来，调整张量的维度顺序以符合 Grad-CAM 的期望
#     tensor = tensor.permute(0, 3, 1, 2)  # 形状变为 [BATCH, CHANNELS, HEIGHT, WIDTH]
#     print(tensor.shape)
#     return tensor


def reshape_transform(tensor):
    # 假设 tensor 的形状为 [BATCH, TIME, HEIGHT, WIDTH, CHANNELS]
    # 其中 HEIGHT 和 WIDTH 都是 224
    batch, time, height, width, channels = tensor.size()
    print(tensor.shape)

    # 选择一个特定的时间步，例如第一个时间步
    tensor = tensor[:, 0, :, :, :]  # 形状变为 [BATCH, HEIGHT, WIDTH, CHANNELS]

    # 调整张量维度以符合 Grad-CAM 的期望格式
    tensor = tensor.permute(0, 3, 1, 2)  # 形状变为 [BATCH, CHANNELS, HEIGHT, WIDTH]
    tensor = torch.randn(1, 1024, 7, 7) 
    print(tensor.shape)
    return tensor



if __name__ == '__main__':
    """ python swinT_example.py -image-path <path_to_image>
    Example usage of using cam-methods on a SwinTransformers network.
    """

    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}

    if args.method not in list(methods.keys()):
        raise Exception(f"method should be one of {list(methods.keys())}")



    model = Classifier(transformer_output_size=1*1024*16*7*7, num_classes=700)
    model.eval()

    if args.use_cuda:
        model = model.cuda()


    target_layers = [model.video_transformer.layers[-1].blocks[-1].norm2]
    print(target_layers)


    if args.method not in methods:
        raise Exception(f"Method {args.method} not implemented")

    if args.method == "ablationcam":
        cam = methods[args.method](model=model,
                                   target_layers=target_layers,
                                   use_cuda=args.use_cuda,
                                   reshape_transform=reshape_transform,
                                   ablation_layer=AblationLayerVit())
    else:
        cam = methods[args.method](model=model,
                                   target_layers=target_layers,
                                   use_cuda=args.use_cuda,
                                   reshape_transform=reshape_transform
                                   )

    # rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
    # rgb_img = cv2.resize(rgb_img, (224, 224))
    # rgb_img = np.float32(rgb_img) / 255
    # input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5],
    #                                 std=[0.5, 0.5, 0.5])

    # AblationCAM and ScoreCAM have batched implementations.
    # You can override the internal batch size for faster computation.
    cam.batch_size = 32
    input_tensor = torch.randn(1, 3, 32, 224, 224)

    grayscale_cam = cam(input_tensor=input_tensor,
                        targets=None,
                        eigen_smooth=args.eigen_smooth,
                        aug_smooth=args.aug_smooth)

    # Here grayscale_cam has only one image in the batch
    grayscale_cam = grayscale_cam[0, :]

    # cam_image = show_cam_on_image(grayscale_cam, grayscale_cam)
    cv2.imwrite(f'{args.method}_cam.jpg', grayscale_cam)

