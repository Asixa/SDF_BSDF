import torch
import torch.nn as nn
import numpy as np
import os
from models.bsdf import DisneyBRDF

def smithG1(cosTheta, alpha):
    sinTheta = torch.sqrt(1.0 - cosTheta * cosTheta)
    tanTheta = sinTheta / (cosTheta + 1e-10)
    root = alpha * tanTheta
    return 2.0 / (1.0 + torch.hypot(root, torch.ones_like(root)))


class BSDFRenderer(nn.Module):
    def __init__(self, use_cuda=False):
        super().__init__()

        self.bsdf_fn = DisneyBRDF()


    def forward(self, light, distance, normal, viewdir, bsdf_params):

        light_intensity = light / (distance * distance + 1e-10)

        diffuse, specular, sheen, clearcoat = self.bsdf_fn(
            viewdir, 
            normal, 
            bsdf_params["base_color"], 
            bsdf_params["subsurface"], 
            bsdf_params["metallic"], 
            bsdf_params["specular"], 
            bsdf_params["specular_tint"], 
            bsdf_params["roughness"], 
            bsdf_params["anisotropic"], 
            bsdf_params["sheen"], 
            bsdf_params["sheen_tint"],
            bsdf_params["clearcoat"], 
            bsdf_params["clearcoat_gloss"])

        ret = {"diffuse_rgb": diffuse, 
               "specular_rgb": specular, 
               "sheen_rgb": sheen,
                "clearcoat_rgb": clearcoat,
               "rgb": diffuse + specular + sheen + clearcoat}
        return ret
