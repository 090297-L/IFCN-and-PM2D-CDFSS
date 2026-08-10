from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple, Union
import inspect
import types
import warnings
import numpy as np

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from model import resnet_GPRN as resnet


PseudoMaskProvider = Callable[[Tensor, Tensor], Tensor]


class SimilarityWeightedFeature(nn.Module):
    def __init__(self, similarity_type="cosine"):
        super().__init__()
        self.similarity_type = similarity_type  # 支持dot/cosine

    def forward(self, image_features, proto_features):

        B, C, H, W = image_features.shape

        # 特征预处理
        proto = proto_features.view(B, C, 1)  # [B, C, 1]
        image_flat = image_features.view(B, C, H * W)  # [B, C, H*W]
        # 计算相似度
        if self.similarity_type == "dot":
            similarity = torch.bmm(proto.transpose(1, 2), image_flat)  # [B, 1, H*W]
        elif self.similarity_type == "cosine":
            # 归一化处理
            proto_norm = torch.norm(proto, dim=1, keepdim=True) + 1e-6
            image_norm = torch.norm(image_flat, dim=1, keepdim=True) + 1e-6
            similarity = torch.bmm(proto.transpose(1, 2), image_flat) / (proto_norm * image_norm)

        # 调整形状并计算权重
        similarity = similarity.view(B, 1, H, W)  # [B, 1, H, W]
        weight = 1 - similarity

        # 特征加权（自动广播）
        return image_features * weight


class SimilarityComputer(nn.Module):
    def __init__(self, similarity_metric='cosine'):
        super().__init__()

        self.similarity_metric = similarity_metric

    def forward(self, x1, x2):

        if self.similarity_metric == 'cosine':
            return self._cosine_similarity(x1, x2)
        else:
            raise ValueError(f"Unsupported similarity metric: {self.similarity_metric}")

    def _cosine_similarity(self, x1, x2):
        # 计算模长
        x1_norm = torch.norm(x1, dim=1, keepdim=True)  # [batchsize, 1, 1, 1]
        x2_norm = torch.norm(x2, dim=1, keepdim=True)  # [batchsize, 1, h, w]

        # 计算点积
        dot_product = (x1 * x2).sum(dim=1, keepdim=True)  # [batchsize, 1, h, w]

        # 计算余弦相似度
        cosine_similarity = dot_product / (x1_norm * x2_norm + 1e-8)  # [batchsize, 1, h, w]

        # 用余弦相似度矩阵调制 x2
        result = (1 - cosine_similarity) * x1  # [batchsize, c, h, w]
        return result


class FBCI(nn.Module):
    def __init__(self, dim):
        super(FBCI, self).__init__()

        self.dim = dim

        self.k = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.q = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)

        self.similarity = SimilarityWeightedFeature()

        # self.similarity = SimilarityComputer()

    def forward(self, x1, x2, x3):
        batch_size = x1.size(0)

        x_kl = self.similarity(x1, x2)

        k = self.k(x_kl).view(batch_size, self.dim, -1)
        kt = k.permute(0, 2, 1)

        q = self.q(x_kl).view(batch_size, self.dim, -1)

        v = self.v(x3).view(batch_size, self.dim, -1)
        vt = v.permute(0, 2, 1)

        ktq_matmul = torch.matmul(kt, q)
        ktq_matmul = F.softmax(ktq_matmul, dim=-1)

        vtq_matmul = torch.matmul(vt, q)
        vtq_matmul = F.softmax(vtq_matmul, dim=-1)

        # print(k.size(), q.size(), v.size(), ktq_matmul.size())

        out1 = torch.matmul(v, ktq_matmul)
        # # out1 = F.softmax(out1, dim=-1)
        y1 = out1.view(batch_size, self.dim, *x1.size()[2:])
        out2 = torch.matmul(q, ktq_matmul)
        # out2 = F.softmax(out2, dim=-1)
        y2 = out2.view(batch_size, self.dim, *x1.size()[2:])
        out3 = torch.matmul(v, vtq_matmul)
        # out3 = F.softmax(out3, dim=-1)
        y3 = out3.view(batch_size, self.dim, *x1.size()[2:])
        out4 = torch.matmul(q, vtq_matmul)
        # # out4 = F.softmax(out4, dim=-1)
        y4 = out4.view(batch_size, self.dim, *x1.size()[2:])
        out = (y1 + y2 + y3 + y4) / 4
        # out = (y2+y3)/2

        out = out + x_kl
        # out = out + x3

        return out, x3


class CSC(nn.Module):

    def __init__(self, dim: int = 64, alpha: float = 0.1, eps: float = 1e-5):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.refine = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )

    def _masked_mean(self, features: Tensor, mask: Tensor) -> Tensor:
        mask = mask.float()
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        mask = F.interpolate(mask, size=features.shape[-2:], mode="nearest")
        numerator = (features * mask).sum(dim=(-2, -1), keepdim=True)
        denominator = mask.sum(dim=(-2, -1), keepdim=True).clamp_min(self.eps)
        return numerator / denominator

    def forward(
        self,
        query: Tensor,
        source: Tensor,
        source_mask: Tensor,
        target_pseudo_mask: Tensor,
    ) -> Tensor:
        source_region = self._masked_mean(source, source_mask)
        target_region = self._masked_mean(query, target_pseudo_mask)
        shift = target_region - source_region
        corrected = query - self.alpha * shift
        return query + self.refine(corrected - query)


class MobileSAMPseudoMaskProvider(nn.Module):

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model_type: str = "vit_t",
        device: Optional[Union[str, torch.device]] = None,
        threshold: float = 0.5,
    ):
        super().__init__()
        try:
            from mobile_sam import SamPredictor, sam_model_registry
        except ImportError as exc:
            raise ImportError(
                "MobileSAM is unavailable. Install the official mobile_sam package "
                "and timm, or set use_mobilesam=False."
            ) from exc
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.threshold = threshold
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path).to(self.device).eval()
        for parameter in self.sam.parameters():
            parameter.requires_grad_(False)
        self.predictor = SamPredictor(self.sam)
        self._patch_prompt_encoder_signature()
        if checkpoint_path is None:
            warnings.warn("MobileSAM is running without a checkpoint; use mobile_sam.pt for real inference.")

    def _patch_prompt_encoder_signature(self) -> None:
        prompt_encoder = self.sam.prompt_encoder
        parameters = inspect.signature(prompt_encoder.forward).parameters
        if "protos" not in parameters:
            return
        original_forward = prompt_encoder.forward

        def forward_compat(module, points=None, boxes=None, protos=None, masks=None):
            return original_forward(points, boxes, protos, masks)

        prompt_encoder.forward = types.MethodType(forward_compat, prompt_encoder)

    @staticmethod
    def _to_rgb_uint8(image: Tensor) -> np.ndarray:
        image = image.detach().float().clamp(0, 1)
        return image.permute(1, 2, 0).mul(255).byte().cpu().numpy()

    @torch.no_grad()
    def forward(self, images: Tensor, probabilities: Tensor) -> Tensor:
        masks = []
        for image, probability in zip(images, probabilities):
            image_np = self._to_rgb_uint8(image)
            binary = probability.squeeze(0).detach().float().cpu().numpy() > self.threshold
            ys, xs = np.where(binary)
            if len(xs) == 0:
                height, width = binary.shape
                ys, xs = np.array([height // 2]), np.array([width // 2])
            point_coords = np.array([[float(xs.mean()), float(ys.mean())]], dtype=np.float32)
            point_labels = np.array([1], dtype=np.int32)
            box = np.array(
                [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())],
                dtype=np.float32,
            )
            self.predictor.set_image(image_np)
            sam_masks, scores, _ = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=False,
            )
            selected = sam_masks[int(np.argmax(scores))].astype(np.float32)
            masks.append(torch.from_numpy(selected).to(probability.device).unsqueeze(0))
        return torch.stack(masks, dim=0)

class IFCN(nn.Module):

    def __init__(
        self,
        backbone: Optional[Union[str, int]] = None,
        shot: int = 1,
        args=None,
        layers: int = 50,
        classes: int = 2,
        criterion: Optional[nn.Module] = None,
        pretrained: bool = False,
        feature_dim: int = 64,
        alpha: float = 0.1,
        pseudo_mask_provider: Optional[PseudoMaskProvider] = None,
        use_mobilesam: bool = True,
        mobile_sam_checkpoint: Optional[str] = None,
        mobile_sam_model_type: str = 'vit_t',
        mobile_sam_device: Optional[Union[str, torch.device]] = None,
        mobile_sam_threshold: float = 0.5,
        **kwargs,
    ):
        super().__init__()
        del args, kwargs
        self.shot = shot
        self.classes = classes
        self.feature_dim = feature_dim
        self.criterion = criterion or nn.CrossEntropyLoss(ignore_index=255)
        if pseudo_mask_provider is not None:
            self.pseudo_mask_provider = pseudo_mask_provider
        elif use_mobilesam:
            self.mobile_sam = MobileSAMPseudoMaskProvider(
                checkpoint_path=mobile_sam_checkpoint,
                model_type=mobile_sam_model_type,
                device=mobile_sam_device,
                threshold=mobile_sam_threshold,
            )
            self.pseudo_mask_provider = self.mobile_sam
        else:
            self.mobile_sam = None
            self.pseudo_mask_provider = None

        backbone_name = backbone
        if backbone_name is None:
            backbone_name = f"resnet{layers}"
        if isinstance(backbone_name, int):
            backbone_name = f"resnet{backbone_name}"
        backbone_model = resnet.__dict__[backbone_name](pretrained=pretrained)
        self.layer0 = nn.Sequential(
            backbone_model.conv1,
            backbone_model.bn1,
            backbone_model.relu,
            backbone_model.maxpool,
        )
        self.layer1 = backbone_model.layer1
        self.layer2 = backbone_model.layer2
        self.layer3 = backbone_model.layer3
        self.layer4 = nn.Identity()

        backbone_dim = 1024 if layers >= 50 else 256
        self.project = nn.Sequential(
            nn.Conv2d(backbone_dim, feature_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
        )
        self.fbci = FBCI(feature_dim)
        self.csc = CSC(feature_dim, alpha=alpha)
        self.classifier = nn.Sequential(
            nn.Conv2d(feature_dim * 3, feature_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(feature_dim, classes, kernel_size=1),
        )
        self.coarse_classifier = nn.Conv2d(feature_dim * 2, classes, kernel_size=1)

    @staticmethod
    def _masked_mean(features: Tensor, mask: Tensor, eps: float = 1e-5) -> Tensor:
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        mask = F.interpolate(mask.float(), size=features.shape[-2:], mode="nearest")
        numerator = (features * mask).sum(dim=(-2, -1), keepdim=True)
        denominator = mask.sum(dim=(-2, -1), keepdim=True).clamp_min(eps)
        return numerator / denominator

    @staticmethod
    def _as_support_tensor(value: Tensor) -> Tensor:
        if value.ndim == 4:
            value = value.unsqueeze(1)
        return value.float()

    @staticmethod
    def _as_mask_tensor(value: Tensor) -> Tensor:
        if value.ndim == 5 and value.shape[2] == 1:
            value = value.squeeze(2)
        if value.ndim == 3:
            value = value.unsqueeze(1)
        return value.float()

    def _encode(self, image: Tensor) -> Tensor:
        feature = self.layer0(image)
        feature = self.layer1(feature)
        feature = self.layer2(feature)
        feature = self.layer3(feature)
        return self.project(feature)

    def _encode_support(self, support: Tensor, masks: Tensor) -> Tuple[Tensor, Tensor]:
        support_features: List[Tensor] = []
        foreground: List[Tensor] = []
        background: List[Tensor] = []
        for index in range(support.shape[1]):
            support_feature = self._encode(support[:, index])
            support_features.append(support_feature)
            support_mask = masks[:, index]
            foreground.append(self._masked_mean(support_feature, support_mask))
            background.append(self._masked_mean(support_feature, 1.0 - support_mask))
        support_feature = torch.stack(support_features, dim=1).mean(dim=1)
        foreground_proto = torch.stack(foreground, dim=1).mean(dim=1)
        background_proto = torch.stack(background, dim=1).mean(dim=1)
        return support_feature, foreground_proto, background_proto

    def _make_pseudo_mask(self, image: Tensor, coarse_logits: Tensor) -> Tensor:
        probability = coarse_logits.softmax(dim=1)[:, 1:2]
        if self.pseudo_mask_provider is not None:
            pseudo = self.pseudo_mask_provider(image, probability)
            if pseudo.ndim == 3:
                pseudo = pseudo.unsqueeze(1)
            return pseudo.float().to(probability.device)
        return probability.detach()

    def forward(
        self,
        img_s_list=None,
        mask_s_list=None,
        nom=None,
        img_q=None,
        mask_q=None,
        x=None,
        s_x=None,
        s_y=None,
        y=None,
        return_aux: bool = False,
    ):
        if x is not None:
            img_q = x
        if s_x is not None:
            img_s_list = s_x
        if s_y is not None:
            mask_s_list = s_y
        if y is not None:
            mask_q = y
        if img_q is None or img_s_list is None or mask_s_list is None:
            raise ValueError("query, support images, and support masks are required")
        if nom is None:
            nom = torch.zeros_like(img_q)

        support = self._as_support_tensor(img_s_list)
        support_masks = self._as_mask_tensor(mask_s_list)
        query_feature = self._encode(img_q.float())
        normal_feature = self._encode(nom.float())
        support_feature, foreground_proto, background_proto = self._encode_support(
            support, support_masks
        )

        normal_proto = self._masked_mean(normal_feature, torch.ones_like(normal_feature[:, :1]))
        query_cognition = self.fbci(query_feature, normal_proto, support_feature)
        prototype_difference = foreground_proto - background_proto
        prototype_map = prototype_difference.expand_as(query_cognition)
        coarse_logits = self.coarse_classifier(torch.cat([query_cognition, prototype_map], dim=1))
        pseudo_mask = self._make_pseudo_mask(img_q, coarse_logits)
        query_corrected = self.csc(
            query_cognition,
            support_feature,
            support_masks[:, 0],
            pseudo_mask,
        )
        final_features = torch.cat(
            [query_corrected, prototype_map, query_corrected - query_cognition], dim=1
        )
        final_logits = self.classifier(final_features)
        final_logits = F.interpolate(
            final_logits, size=img_q.shape[-2:], mode="bilinear", align_corners=False
        )
        coarse_logits = F.interpolate(
            coarse_logits, size=img_q.shape[-2:], mode="bilinear", align_corners=False
        )

        if not self.training:
            if return_aux:
                return {"logits": final_logits, "coarse_logits": coarse_logits, "pseudo_mask": pseudo_mask}
            return final_logits

        if mask_q is None:
            raise ValueError("query mask is required during training")
        target = mask_q.long()
        if target.ndim == 4:
            target = target.squeeze(1)
        loss_final = self.criterion(final_logits, target)
        loss_coarse = self.criterion(coarse_logits, target)
        loss = loss_final + 0.3 * loss_coarse
        prediction = final_logits.argmax(dim=1)
        if return_aux:
            return prediction, loss, {
                "loss_final": loss_final.detach(),
                "loss_coarse": loss_coarse.detach(),
                "pseudo_mask": pseudo_mask.detach(),
            }
        return prediction, loss


IFCN_Net = IFCN




