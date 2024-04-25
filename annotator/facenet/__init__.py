import torch
import einops
import numpy as np
from PIL import Image
from torchvision import transforms as pth_transforms
from facenet_pytorch import MTCNN, InceptionResnetV1


class FaceNet():

    def __init__(self):

        self.resnet = None
        self.mtcnn = None

    def load_model(self):

        # Create an inception resnet (in eval mode) for ID feature:
        if self.resnet is None:
            self.resnet = InceptionResnetV1(pretrained='vggface2', device='cuda').eval()
        # Create a face detection pipeline using MTCNN:
        if self.mtcnn is None:
            self.mtcnn = MTCNN(image_size=160, device='cuda')

    def unload_model(self):

        if self.resnet is not None:
            self.resnet = self.resnet.to('cpu')

        if self.mtcnn is not None:
            self.mtcnn = self.mtcnn.to('cpu')

    @torch.no_grad()
    def __call__(self, img):

        if self.resnet is None:
            self.load_model()

        transform = pth_transforms.Compose([
            pth_transforms.Resize((160, 160)),
            pth_transforms.ToTensor(),
        ])
        img = Image.fromarray(img)
        # Get cropped and prewhitened image tensor
        img_cropped = self.mtcnn(img)
        if img_cropped is not None:
            img_cropped = img_cropped.to(self.resnet.device)
        else:
            print('fail to detect faces')
            img_cropped = transform(img).to(self.resnet.device)
        cropped_img = Image.fromarray((einops.rearrange(img_cropped, 'c h w -> h w c') * 127.5 + 127.5).squeeze().cpu().numpy().clip(0, 255).astype(np.uint8))
        # Calculate embedding (unsqueeze to add batch dimension)
        img_embedding = self.resnet(img_cropped.unsqueeze(0))

        img_embedding = img_embedding.unsqueeze(0)

        return img_embedding, cropped_img
