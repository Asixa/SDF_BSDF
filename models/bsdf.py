import torch
import math


def sqr(x):
    return x * x

def Fd_Burley(NdotL, NdotV):
    FL = SchlickFresnel(NdotL)
    FV = SchlickFresnel(NdotV)
    return (1.0 / math.pi) * FL * FV

def Fss(x):
    x = torch.clamp(x, min=0.0, max=1.0)
    return torch.lerp(1.0, torch.pow(x, 5.0), 0.5)

def Fresnel(f0, x):
    return f0 + (1.0 - f0) * torch.pow(1.0 - x, 5.0)

def SchlickFresnel(x):
    x = torch.clamp(x, min=0.0, max=1.0)
    return torch.pow(1.0 - x, 5.0)

def GTR1(NdotH, a):
    if a >= 1.0:
        return (1.0 / math.pi)
    aa = a * a
    t = 1.0 + (aa - 1.0) * NdotH * NdotH
    return (aa - 1.0) / (math.pi * torch.log(aa) * t)

def GTR2_aniso(NdotH, phi, ax, ay):
    sinae = torch.sin(phi)
    cosae = torch.cos(phi)
    ee = (cosae*ax)**2 + (sinae*ay)**2 + NdotH**2
    return (1.0 / math.pi) * (1.0 / (ax*ay*(ee - (1.0 - NdotH**2) / ee))**1.5)

def Vis_SmithGGXCorrelated(NdotL, NdotV, roughness):
    a = roughness * roughness
    GGXL = (2.0 * NdotL) / (NdotL + torch.sqrt(a + (1.0 - a) * NdotL * NdotL))
    GGXV = (2.0 * NdotV) / (NdotV + torch.sqrt(a + (1.0 - a) * NdotV * NdotV))
    return GGXL * GGXV


class DisneyBRDF(torch.nn.Module):
    def __init__(self):
        super(DisneyBRDF, self).__init__()

    def forward(self, view_dir, normal, base_color, subsurface, metallic, specular, specular_tint, roughness, anisotropic, sheen, sheen_tint, clearcoat, clearcoat_gloss):
        # Normalize input directions
        view_dir = view_dir / torch.norm(view_dir, dim=-1, keepdim=True)
        normal = normal / torch.norm(normal, dim=-1, keepdim=True)

        # Compute halfway vector
        halfway_vec = view_dir + normal
        halfway_vec = halfway_vec / torch.norm(halfway_vec, dim=-1, keepdim=True)

        # Compute dot products
        NdotL = torch.clamp(torch.sum(normal * view_dir, dim=-1), min=0.0)
        NdotV = torch.clamp(torch.sum(normal * view_dir, dim=-1), min=0.0)
        NdotH = torch.clamp(torch.sum(normal * halfway_vec, dim=-1), min=0.0)
        LdotH = torch.clamp(torch.sum(view_dir * halfway_vec, dim=-1), min=0.0)

        # Diffuse term
        diffuse = base_color / math.pi

        # Subsurface term
        Fss90 = torch.lerp(1.0, Fss(LdotH), subsurface)
        Fss = torch.lerp(1.0, Fss90, Fd_Burley(NdotL, NdotV))

        # Specular term
        aspect = torch.sqrt(1.0 - anisotropic * 0.9)
        ax = torch.max(0.001, sqr(roughness) / aspect)
        ay = torch.max(0.001, sqr(roughness) * aspect)
        Ds = GTR2_aniso(NdotH, torch.atan2(halfway_vec[..., 1], halfway_vec[..., 0]), ax, ay)
        Fs = Fresnel(specular, LdotH)
        Fs = torch.lerp(Fs, torch.ones_like(Fs), metallic)

        # Sheen term
        Fsheen = sheen * Fd_Burley(NdotL, NdotV)
        Fsheen = torch.lerp(Fsheen, Fsheen * base_color, sheen_tint)

        # Clearcoat term
        Dr = GTR1(NdotH, torch.lerp(0.1, 0.001, clearcoat_gloss))
        Fr = torch.lerp(0.04, 1.0, Fresnel(0.5, LdotH))
        Fcc = clearcoat * Dr * Fr

        # Combine terms
        specular = (1.0 / math.pi) * NdotL * Fs * Ds * Vis_SmithGGXCorrelated(NdotL, NdotV, roughness)
        sheen = Fsheen
        clearcoat = Fcc

        return diffuse + specular + sheen + clearcoat, diffuse, specular, sheen, clearcoat
        return diffuse + specular + sheen + clearcoat
