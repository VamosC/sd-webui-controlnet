import os
import torch
import einops
from modules import devices
from annotator.annotator_path import models_path
from .decalib.deca import DECA
from .decalib.utils.config import cfg as deca_cfg
from .decalib.datasets import datasets as deca_dataset
from PIL import Image


class FlameHeadModel:
    """
    A class for reconstructing the human head in images using the FLAME head model.

    Attributes:
        model_dir (str): Path to the directory where the pose models are stored.
    """

    model_dir = os.path.join(models_path, "FLAME")

    def __init__(self):
        self.device = devices.get_device_for("controlnet")
        self.deca = None

    def load_model(self):
        """
        Load the 3D head model
        """
        # Build DECA
        if self.deca is None:
            deca_cfg.model.use_tex = True
            deca_cfg.model.tex_path = os.path.join(self.model_dir, "FLAME_texture.npz")
            deca_cfg.model.tex_type = "FLAME"
            deca_cfg.rasterizer_type = "pytorch3d"
            self.deca = DECA(config=deca_cfg)

    def unload_model(self):
        """
        Unload the head model by moving them to the CPU.
        """
        if self.deca is not None:
            self.deca = self.deca.to('cpu')

    def create_inter_data(self, dataset, modes, meanshape_path=""):

        if self.deca is None:
            self.load_model()

        meanshape = None
        if os.path.exists(meanshape_path):
            print("use meanshape: ", meanshape_path)
            with open(meanshape_path, "rb") as f:
                meanshape = pickle.load(f)
        else:
            print("not use meanshape")

        img2 = dataset[-1]["image"].unsqueeze(0).to("cuda")
        with torch.no_grad():
            code2 = self.deca.encode(img2)
        image2 = dataset[-1]["original_image"].unsqueeze(0).to("cuda")
        tform2 = dataset[-1]["tform"].unsqueeze(0)
        tform2 = torch.inverse(tform2).transpose(1, 2).to("cuda")
        code2["tform"] = tform2

        for i in range(len(dataset) - 1):

            img1 = dataset[i]["image"].unsqueeze(0).to("cuda")

            with torch.no_grad():
                code1 = self.deca.encode(img1)

            # To align the face when the pose is changing
            ffhq_center = None

            tform = dataset[i]["tform"].unsqueeze(0)
            tform = torch.inverse(tform).transpose(1, 2).to("cuda")
            original_image = dataset[i]["original_image"].unsqueeze(0).to("cuda")

            code1["tform"] = tform
            if meanshape is not None:
                code1["shape"] = meanshape

            for mode in modes:

                code = {}
                for k in code1:
                    code[k] = code1[k].clone()

                if "position" in mode:
                    code["tform"] = code2["tform"]
                    code["cam"] = code2["cam"]
                if "pose" in mode:
                    code["pose"][:, :3] = code2["pose"][:, :3]
                if "lighting" in mode:
                    code["light"] = code2["light"]
                if "expression" in mode:
                    code["exp"] = code2["exp"]
                    code["pose"][:, 3:] = code2["pose"][:, 3:]

                opdict, _ = self.deca.decode(
                    code,
                    render_orig=True,
                    original_image=original_image,
                    tform=code["tform"],
                )

                rendered = opdict["rendered_images"].detach()
                normal = opdict["normal_images"].detach()
                albedo = opdict["albedo_images"].detach()

                o = einops.rearrange(torch.cat([normal, albedo, rendered], dim=1).squeeze(), 'c h w -> h w c').cpu().numpy()

                return o

    def __call__(
        self,
        oriImg,
        refImg,
        res,
        modes=('position', 'pose')
    ):
        """
        Reconstruct 3D Face Model in the given image and render

        Args:
            oriImg (numpy.ndarray): The input image for pose detection and drawing.

        Returns:
            numpy.ndarray: Normal, albedo, and rendered images from 3D Face
        """
        imagepath_list = [oriImg, refImg]
        dataset = deca_dataset.TestData(imagepath_list, iscrop=True, size=res)
        data = self.create_inter_data(dataset, modes=[modes])
        return data
