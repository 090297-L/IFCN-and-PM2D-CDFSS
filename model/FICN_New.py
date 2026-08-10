import model.resnet_GPRN as resnet
import sys
import os
import torch
from torch import nn
import torch.nn.functional as F
import pdb
from SAM2pred import *

import numpy as np
from skimage.measure import label

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

from peft import LoraConfig, get_peft_model

from model.SPAC import SPAC
from model.FBCI import FBCI
from model.DSTCP import DSTCP

class SAMMaskGenerator_without_Prompt(nn.Module):
    def __init__(self,
                 model_type="vit_b",
                 checkpoint_path="/gly/yury/lhy/CFNet/initmodel/sam_vit_b_01ec64.pth",
                 points_per_side=8,
                 pred_iou_thresh=0.82,
                 device="cuda"):
        super().__init__()
        self.device = device

        # 初始化 SAM 模型
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=device)

        for param in sam.parameters():
            param.requires_grad_(False)
        sam.eval()  # 设置为评估模式

        # 配置自动掩码生成器
        self.generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=0.85,
            crop_n_layers=1,
            points_per_batch=64         # 提高批处理点数加速推理
        )

        # 图像归一化参数
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def denormalize(self, tensor):
        """将归一化的图像张量转换到 0-255 范围"""
        return torch.clamp((tensor * self.std + self.mean) * 255, 0, 255).byte()

    def forward(self, images, strategy="largest"):
        """
        输入:
            images - [B, C, H, W] 归一化后的图像张量 (0-1)
            strategy - 掩码选择策略: "largest" | "best_quality" | "merge_all"
        输出:
            masks - [B, 1, H, W] 二进制掩码
        """
        batch_masks = []
        for img in images:
            # 转换到 SAM 输入格式
            img_np = self.denormalize(img.unsqueeze(0))[0]  # [3, H, W]
            img_np = img_np.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]

            # 生成掩码
            sam_masks = self.generator.generate(img_np)

            if len(sam_masks) == 0:
                mask = torch.zeros(img.shape[1], img.shape[2], device=self.device)
            else:
                # 选择掩码策略
                if strategy == "largest":
                    selected = sorted(sam_masks, key=lambda x: x['area'], reverse=True)[0]
                elif strategy == "best_quality":
                    selected = sorted(sam_masks, key=lambda x: x['predicted_iou'], reverse=True)[0]
                elif strategy == "merge_all":
                    merged_mask = np.zeros_like(sam_masks[0]['segmentation'], dtype=np.uint8)
                    for m in sam_masks:
                        merged_mask |= m['segmentation']
                    selected = {'segmentation': merged_mask}

                mask = torch.from_numpy(selected['segmentation']).to(self.device)

            batch_masks.append(mask.float().unsqueeze(0))  # [1, H, W]

        return torch.stack(batch_masks, dim=0)  # [B, 1, H, W]
    
class SAMMaskGenerator(nn.Module):
    def __init__(self,
                 model_type="vit_b",
                 checkpoint_path="/gly/yury/lhy/CFNet/initmodel/sam_vit_b_01ec64.pth",
                 points_per_side=8,
                 pred_iou_thresh=0.82,
                 device="cuda"):
        super().__init__()
        self.device = device
        # 初始化 SAM 模型
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=device)
        for param in sam.parameters():
            param.requires_grad_(False)
        sam.eval()
        # 配置双生成器
        self.auto_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=0.85,
            crop_n_layers=1,
            points_per_batch=64
        )
        self.predictor = SamPredictor(sam)
        # 图像归一化参数
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def denormalize(self, tensor):
        """将归一化的图像张量转换到0-255范围"""
        return torch.clamp((tensor * self.std + self.mean) * 255, 0, 255).byte()

    def forward(self, images, strategy="largest", prompts=None):
        """
        输入:
            images: [B, C, H, W] 归一化后的图像张量
            strategy: 掩码选择策略 ("largest"|"best_quality"|"merge_all")
            prompts: list of dicts，每个dict包含:
                - points: [N, 2] 点坐标
                - labels: [N] 点标签 (1前景, 0背景)
                - box: [4,] 边界框 [x0,y0,x1,y1]
        输出:
            masks: [B, 1, H, W] 二进制掩码
        """
        batch_masks = []
        for i, img in enumerate(images):
            # 图像预处理
            img_np = self.denormalize(img.unsqueeze(0))[0]
            img_np = img_np.permute(1, 2, 0).cpu().numpy()

            if prompts is not None:
                # Prompt模式
                prompt = prompts[i] if i < len(prompts) else {}
                self.predictor.set_image(img_np)
                masks, scores, _ = self.predictor.predict(
                    point_coords=prompt.get('points', None),
                    point_labels=prompt.get('labels', None),
                    box=prompt.get('box', None),
                    multimask_output=strategy in ["best_quality", "largest", "merge_all"]
                )
                sam_masks = [
                    {'segmentation': m, 'area': m.sum(), 'predicted_iou': s}
                    for m, s in zip(masks, scores)
                ]
            else:
                # 自动生成模式
                sam_masks = self.auto_generator.generate(img_np)
            # 统一策略处理
            if not sam_masks:
                mask = torch.zeros(img.shape[1], img.shape[2], device=self.device)
            else:
                if strategy == "largest":
                    selected = max(sam_masks, key=lambda x: x['area'])
                elif strategy == "best_quality":
                    selected = max(sam_masks, key=lambda x: x['predicted_iou'])
                elif strategy == "merge_all":
                    merged = np.zeros_like(sam_masks[0]['segmentation'], dtype=np.uint8)
                    for m in sam_masks:
                        merged |= m['segmentation']
                    selected = {'segmentation': merged}
                mask = torch.from_numpy(selected['segmentation']).to(self.device)
            batch_masks.append(mask.float().unsqueeze(0))
        return torch.stack(batch_masks, dim=0)

class SAMPromptGenerator(nn.Module):
    def __init__(self,
                 prompt_type="both",  # "point"|"box"|"both"
                 point_strategy="center",  # "center"|"random"
                 num_points=1,
                 add_background=False,
                 threshold=0.5,
                 connectivity=2):
        super().__init__()
        self.prompt_type = prompt_type
        self.point_strategy = point_strategy
        self.num_points = num_points
        self.add_background = add_background
        self.threshold = threshold
        self.connectivity = connectivity

    def _find_main_object(self, mask):
        """提取最大连通区域"""
        if np.count_nonzero(mask) == 0:
            return mask

        labeled = label(mask, connectivity=self.connectivity)
        if labeled.max() == 0:
            return mask
        largest_label = np.argmax(np.bincount(labeled.flat)[1:]) + 1
        return (labeled == largest_label).astype(np.uint8)

    def _generate_points(self, mask, fg_coords):
        """生成点提示策略"""
        points, labels = [], []

        # 前景点生成
        if len(fg_coords) > 0 and not np.all(fg_coords == 0):
            if self.point_strategy == "center":
                # 计算质心坐标
                y_center = int(np.round(fg_coords[:, 0].mean()))
                x_center = int(np.round(fg_coords[:, 1].mean()))
                points.append([x_center, y_center])
                labels.append(1)
            else:  # 随机采样
                indices = np.random.choice(len(fg_coords),
                                           size=min(self.num_points, len(fg_coords)))
                for idx in indices:
                    y, x = fg_coords[idx]
                    points.append([x, y])
                    labels.append(1)

        # 背景点生成
        if self.add_background and (len(points) > 0):
            bg_coords = np.column_stack(np.where(mask == 0))
            if len(bg_coords) > 0 and not np.all(bg_coords == 0):
                indices = np.random.choice(len(bg_coords), size=self.num_points)
                for idx in indices:
                    y, x = bg_coords[idx]
                    points.append([x, y])
                    labels.append(0)

        return np.array(points), np.array(labels)

    def forward(self, masks):
        """
        输入:
            masks: [B, 1, H, W] 概率掩码 (0-1范围)
        输出:
            prompts: list of dicts，每个dict包含:
                - points: [N, 2] 点坐标 (绝对坐标)
                - labels: [N] 点标签 (1=前景, 0=背景)
                - box: [4,] 边界框 [x0,y0,x1,y1]
        """
        batch_prompts = []
        with torch.no_grad():
            masks_np = masks.detach().cpu().numpy()
            
            for mask in masks_np:
                binary_mask = (mask[0] > self.threshold).astype(np.uint8)
                processed_mask = self._find_main_object(binary_mask)
                fg_coords = np.column_stack(np.where(processed_mask))
                
                prompt = {"points": None, "labels": None, "box": None}
                
                # 边界框生成（增加维度验证）
                if self.prompt_type in ["box", "both"]:
                    if fg_coords.size > 0 and fg_coords.ndim == 2:
                        try:
                            y_min, x_min = np.min(fg_coords, axis=0)
                            y_max, x_max = np.max(fg_coords, axis=0)
                            prompt["box"] = np.array(
                                [x_min, y_min, x_max, y_max],
                                dtype=np.float32
                            )
                        except ValueError as e:
                            print(f"Ignoring box generation due to: {str(e)}")
                            prompt["box"] = None
                    else:
                        prompt["box"] = None
                
                # 点生成（增加空坐标检查）
                if self.prompt_type in ["point", "both"]:
                    if fg_coords.size > 0 and fg_coords.ndim == 2:
                        points, labels = self._generate_points(processed_mask, fg_coords)
                        # 有效性过滤
                        valid_points = [
                            p for p in points 
                            if not np.any(np.isnan(p)) and p[0] < processed_mask.shape[1] and p[1] < processed_mask.shape[0]
                        ]
                        if len(valid_points) > 0:
                            prompt["points"] = np.array(valid_points, dtype=np.float32)
                            prompt["labels"] = labels[:len(valid_points)].astype(np.int64)
                
                batch_prompts.append(prompt)
        
        return batch_prompts

# sam = sam_model_registry["vit_b"](checkpoint="/gly/yury/lhy/CFNet/initmodel/sam_vit_b_01ec64.pth")
# print("Image Encoder Modules:")
# for name, _ in sam.image_encoder.named_modules():
#     print(name)

class SAMMaskGenerator_without_Prompt_LoRA(nn.Module):
    def __init__(self,
                 model_type="vit_b",
                 checkpoint_path="/gly/yury/lhy/CFNet/initmodel/sam_vit_b_01ec64.pth",
                 points_per_side=8,
                 pred_iou_thresh=0.9,
                 lora_rank=8,
                 lora_alpha=16,
                 lora_dropout=0.1,
                 device="cuda"):
        super().__init__()
        self.device = device
        # 初始化 SAM 模型
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        # 创建LoRA配置（注意：target_modules需要根据实际模型结构调整）
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["proj"],  # 关键修改：指定要注入LoRA的模块
            lora_dropout=lora_dropout,
            bias="none",
        )
        # 对image_encoder应用LoRA适配器
        sam.image_encoder = get_peft_model(sam.image_encoder, lora_config)
        
        # 将模型转移到设备
        sam.to(device=device)
        # 自动冻结非LoRA参数（通过peft自动处理）
        # 打印可训练参数数量
        sam.image_encoder.print_trainable_parameters()
        # 配置自动掩码生成器
        self.generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=0.85,
            crop_n_layers=1,
            points_per_batch=64
        )
        # 图像归一化参数
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    def denormalize(self, tensor):
        """将归一化的图像张量转换到 0-255 范围"""
        return torch.clamp((tensor * self.std + self.mean) * 255, 0, 255).byte()
    def forward(self, images, strategy="largest"):
        """
        输入:
            images - [B, C, H, W] 归一化后的图像张量 (0-1)
            strategy - 掩码选择策略: "largest" | "best_quality" | "merge_all"
        输出:
            masks - [B, 1, H, W] 二进制掩码
        """
        batch_masks = []
        for img in images:
            # 转换到 SAM 输入格式
            img_np = self.denormalize(img.unsqueeze(0))[0]  # [3, H, W]
            img_np = img_np.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
            # 生成掩码
            sam_masks = self.generator.generate(img_np)
            if len(sam_masks) == 0:
                mask = torch.zeros(img.shape[1], img.shape[2], device=self.device)
            else:
                # 选择掩码策略
                if strategy == "largest":
                    selected = sorted(sam_masks, key=lambda x: x['area'], reverse=True)[0]
                elif strategy == "best_quality":
                    selected = sorted(sam_masks, key=lambda x: x['predicted_iou'], reverse=True)[0]
                elif strategy == "merge_all":
                    merged_mask = np.zeros_like(sam_masks[0]['segmentation'], dtype=np.uint8)
                    for m in sam_masks:
                        merged_mask |= m['segmentation']
                    selected = {'segmentation': merged_mask}
                mask = torch.from_numpy(selected['segmentation']).to(self.device)
            batch_masks.append(mask.float().unsqueeze(0))  # [1, H, W]
        return torch.stack(batch_masks, dim=0)  # [B, 1, H, W]

class FICN_Net(nn.Module):
    def __init__(self, backbone, shot=1, args = None):
        super(FICN_Net, self).__init__()
        backbone = resnet.__dict__[backbone](pretrained=True)
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1, self.layer2, self.layer3 = backbone.layer1, backbone.layer2, backbone.layer3
        self.refine = False
        self.shot = shot
        self.iter_refine = False
        self.adapter = nn.Linear(1024, 256, bias = False)

        self.cls = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(1024, 2, kernel_size=1)
        )

        self.fbci = FBCI(dim=1024)

        self.spac = SPAC(1024, 1024)

        # self.dstcp = DSTCP(in_channels=1024, proj_channels=1024)

        # self.prompt_gan = SAMPromptGenerator(prompt_type="both",  # "point"|"box"|"both"
        #          point_strategy="center",  # "center"|"random"
        #          num_points=1)

        # self.sam = SAMMaskGenerator(points_per_side=8, pred_iou_thresh=0.82)

        # self.sam_lora = SAMMaskGenerator_without_Prompt_LoRA()

        self.csc_validation_use_gt_mask = bool(getattr(args, "csc_validation_use_gt_mask", False)) if args is not None else False
        self.sam_no_prompt = None if self.csc_validation_use_gt_mask else SAMMaskGenerator_without_Prompt(points_per_side=8, pred_iou_thresh=0.82)

        self.criterion = nn.CrossEntropyLoss(ignore_index=255)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.export_csc_validation = bool(getattr(args, "export_csc_validation", False)) if args is not None else False
        self.csc_export_dir = getattr(args, "csc_export_dir", "csc_validation_exports") if args is not None else "csc_validation_exports"
        self.csc_export_max = int(getattr(args, "csc_export_max", 20)) if args is not None else 20
        self.csc_export_count = 0

    def _to_numpy(self, tensor):
        if tensor is None:
            return None
        return tensor.detach().float().cpu().numpy()

    def _export_csc_validation(
        self,
        source_before,
        target_before,
        source_after,
        target_after,
        supp_sam_mask,
        qry_sam_mask,
        img_q,
        mask_q,
        heat_before=None,
        heat_after=None,
    ):
        if not self.export_csc_validation:
            return
        if self.csc_export_count >= self.csc_export_max:
            return
        os.makedirs(self.csc_export_dir, exist_ok=True)
        path = os.path.join(self.csc_export_dir, f"csc_validation_{self.csc_export_count:04d}.npz")
        payload = {
            "source_before": self._to_numpy(source_before),
            "target_before": self._to_numpy(target_before),
            "source_after": self._to_numpy(source_after),
            "target_after": self._to_numpy(target_after),
            "supp_sam_mask": self._to_numpy(supp_sam_mask),
            "qry_sam_mask": self._to_numpy(qry_sam_mask),
            "image": self._to_numpy(img_q[0].permute(1, 2, 0)) if img_q is not None else None,
            "gt_mask": self._to_numpy(mask_q[0]) if mask_q is not None else None,
            "heat_before": self._to_numpy(heat_before[0]) if heat_before is not None else None,
            "heat_after": self._to_numpy(heat_after[0]) if heat_after is not None else None,
        }
        payload = {k: v for k, v in payload.items() if v is not None}
        np.savez_compressed(path, **payload)
        self.csc_export_count += 1

    def enhance_feature(self, feature_q, normal_prototype, qry_sam_masks):
        def l2_normalize(tensor):
            norm = tensor.norm(p=2, dim=(2, 3), keepdim=True)
            return tensor / norm
        
        # if self.args.dataset in ('isic', 'lung'):
        #     qry_sam_masks = qry_sam_masks[:, 1:, :, :]
        b, m, w, h = qry_sam_masks.shape
        index_mask = torch.zeros_like(qry_sam_masks[:, 0]).long() + m # b x h x w
        for i in range(m):
            index_mask[qry_sam_masks[:, i]==1] = i
        masks = torch.nn.functional.one_hot(index_mask)[:, :, :, :m].permute((0, 3, 1, 2))

        #masks = qry_sam_masks

        if self.training:
            target_masks = F.interpolate(masks.float(), feature_q.shape[-2:], mode='nearest')
        else:
            target_masks = masks.float()
        map_features = self.masked_average_pooling2(feature_q, target_masks) # b x m x c, 用mask生成的prototypes 

        # graph_prompt = self.graph_attention(map_features, map_features, map_features)

        feat_FBCI, kl = self.fbci(feature_q, normal_prototype, feature_q)

        w = 0.1

        # map_features = map_features + w * feat_FBCI
        b, m, w, h = target_masks.shape
        _, _, c = map_features.shape
        _map_features = map_features.permute(0, 2, 1).contiguous() # b x c x m
        feature_sum = _map_features @ target_masks.view(b, m, -1) # 
        feature_sum = feature_sum.view(b, c, w, h)

        # sum_mask = target_masks.sum(dim=1, keepdim=True)
        # enabled_feat = torch.div(feature_sum, sum_mask + 1e-5)
        enabled_feat = feature_sum + w * feat_FBCI
        # enabled_feat = feature_sum

        return enabled_feat


    def forward(self, img_s_list, mask_s_list, nom, img_q, mask_q):
        b, c, h, w = img_q.shape
        # keep_num = self.args.keep_num
        # qry_sam_masks = qry_sam_masks[:, :keep_num, :, :]
        # for i in range(self.shot):
        #     supp_sam_masks[i] = supp_sam_masks[i][:, :keep_num, :, :]

        # feature maps of support images

        img_s_list, mask_s_list = self.get_nshot(img_s_list, mask_s_list)

        feature_s_list = []
        origin_feat_s = []
        #supp_dis = []
        supp_mask_list = []
        loss_dstcp = 0

        with torch.no_grad():
            # self.query_feat = query_feat
            n_0 = self.layer0(nom)
            n_0 = self.layer1(n_0)
        n_0 = self.layer2(n_0)
        feature_n = self.layer3(n_0)

        normal_prototype = self.gap(feature_n)

        for k in range(len(img_s_list)):
            with torch.no_grad():
                # img_s_add = img_s_list[k] * (mask_s_list[k]).unsqueeze(1)
                s_0 = self.layer0(img_s_list[k])
                s_0 = self.layer1(s_0)
                if self.csc_validation_use_gt_mask:
                    supp_sam_mask = mask_s_list[k].unsqueeze(1).float()
                else:
                    supp_sam_mask = self.sam_no_prompt(img_s_list[k])
                # supp_sam_mask = self.sam_lora(img_s_list[k])
            s_0 = self.layer2(s_0)
            s_0 = self.layer3(s_0)
            origin_feat_s.append(s_0)
            target_size1 = (s_0.size(2), s_0.size(3))
            # 使用双线性插值（适合浮点型特征）
            supp_sam_mask = F.interpolate(
                supp_sam_mask,
                size=target_size1,
                mode='bilinear',
                align_corners=False
            )
            supp_mask_list.append(supp_sam_mask)
            enhance_s = self.enhance_feature(s_0, normal_prototype, supp_sam_mask)
            s_0 = s_0 + enhance_s

            feature_s_list.append(s_0)
            del s_0

        feature_s_ls = torch.cat(feature_s_list, dim=0)
        origin_feat_s = torch.cat(origin_feat_s, dim=0)
        supp_mask_ls = torch.cat(supp_mask_list, dim=0)

        # feature map of query image
        with torch.no_grad():
            # self.query_feat = query_feat
            q_0 = self.layer0(img_q)
            q_0 = self.layer1(q_0)
        q_0 = self.layer2(q_0)
        feature_q = self.layer3(q_0)
        with torch.no_grad():
            if self.csc_validation_use_gt_mask:
                qry_sam_mask = mask_q.unsqueeze(1).float()
            else:
                qry_sam_mask = self.sam_no_prompt(img_q)
            # qry_sam_mask = self.sam_lora(img_q)
        target_size2 = (feature_q.size(2), feature_q.size(3))
        # 使用双线性插值（适合浮点型特征）
        qry_sam_mask = F.interpolate(
            qry_sam_mask,
            size=target_size2,
            mode='bilinear',
            align_corners=False
        )
        enhance_q = self.enhance_feature(feature_q, normal_prototype, qry_sam_mask)
        z = feature_q.clone()
        feature_q = feature_q + enhance_q 
        
        # DSTCP
        # if self.shot > 1:
        #     for i in range(self.shot):
        #         feature_s_single = (feature_s_ls[0+i*b:b+i*b, :, :, :]).float()
        #         supp_mask_single = (supp_mask_ls[0+i*b:b+i*b, :, :, :]).float().unsqueeze(1)
        #         loss_dstcp += self.dstcp(feature_s_single, feature_q, supp_mask_single, qry_sam_mask)
        # else: 
        #     loss_dstcp = self.dstcp(feature_s_ls, feature_q, supp_mask_ls, qry_sam_mask)

        # loss_dstcp = loss_dstcp/self.shot

        # foreground(target class) and background prototypes pooled from K support features
        feature_fg_list = []
        feature_bg_list = []
        supp_out_ls = []

        for k in range(len(img_s_list)):
            feature_fg = self.masked_average_pooling(feature_s_list[k],
                                                               (mask_s_list[k] == 1).float())[None, :]
            feature_bg = self.masked_average_pooling(feature_s_list[k],
                                                               (mask_s_list[k] == 0).float())[None, :]
            
            feature_fg_list.append(feature_fg)
            feature_bg_list.append(feature_bg)

            if self.training:
                supp_similarity_fg = F.cosine_similarity(feature_s_list[k], feature_fg.squeeze(0)[..., None, None], dim=1)
                supp_similarity_bg = F.cosine_similarity(feature_s_list[k], feature_bg.squeeze(0)[..., None, None], dim=1)
                supp_out = torch.cat((supp_similarity_bg[:, None, ...], supp_similarity_fg[:, None, ...]), dim=1) * 10.0

                supp_out = F.interpolate(supp_out, size=(h, w), mode="bilinear", align_corners=True)
                supp_out_ls.append(supp_out)

        # average K foreground prototypes and K background prototypes
        FP = torch.mean(torch.cat(feature_fg_list, dim=0), dim=0).unsqueeze(-1).unsqueeze(-1)
        BP = torch.mean(torch.cat(feature_bg_list, dim=0), dim=0).unsqueeze(-1).unsqueeze(-1)

        if self.training:

            ### iter = 1 (BFP)
            if self.refine:
                out_refine, out_1, supp_out_1, new_FP, new_BP, FP_1, FP_2, out_0 = self.iter_BFP(FP, BP, feature_s_ls, feature_q, self.refine)
            else:
                out_1, supp_out_1, new_FP, new_BP = self.iter_BFP(FP, BP, feature_s_ls, feature_q, self.refine)
            out_1 = F.interpolate(out_1, size=(h, w), mode="bilinear", align_corners=True)
            supp_out_1 = F.interpolate(supp_out_1, size=(h, w), mode="bilinear", align_corners=True)
          
        else:
            if self.refine:
                out_refine, out_1,FP_1, FP_2, out_0,BP_1= self.iter_BFP(FP, BP, feature_s_ls, feature_q, self.refine)
            else:
                out_1 = self.iter_BFP(FP, BP, feature_s_ls, feature_q, self.refine)
            out_1 = F.interpolate(out_1, size=(h, w), mode="bilinear", align_corners=True)

        if self.refine:
            out_refine_origin = out_refine.clone()
            out_refine = F.interpolate(out_refine, size=(h, w), mode="bilinear", align_corners=True)
            out_0 = F.interpolate(out_0, size=(h, w), mode="bilinear", align_corners=True)
            out_ls = [out_refine, out_1]
        else:
            out_ls = [out_1]

        if self.export_csc_validation:
            feature_fg_before_list = []
            feature_bg_before_list = []
            for k in range(len(mask_s_list)):
                feature_s_before = origin_feat_s[k * b:(k + 1) * b]
                feature_fg_before = self.masked_average_pooling(
                    feature_s_before, (mask_s_list[k] == 1).float()
                )[None, :]
                feature_bg_before = self.masked_average_pooling(
                    feature_s_before, (mask_s_list[k] == 0).float()
                )[None, :]
                feature_fg_before_list.append(feature_fg_before)
                feature_bg_before_list.append(feature_bg_before)
            FP_before = torch.mean(torch.cat(feature_fg_before_list, dim=0), dim=0).unsqueeze(-1).unsqueeze(-1)
            BP_before = torch.mean(torch.cat(feature_bg_before_list, dim=0), dim=0).unsqueeze(-1).unsqueeze(-1)
            out_before = self.similarity_func(z, FP_before, BP_before)
            out_before = F.interpolate(out_before, size=(h, w), mode="bilinear", align_corners=True)
            heat_before = out_before.softmax(1)[:, 1]
            heat_after = out_ls[0].softmax(1)[:, 1]
            self._export_csc_validation(
                source_before=origin_feat_s,
                target_before=z,
                source_after=feature_s_ls,
                target_after=feature_q,
                supp_sam_mask=supp_mask_ls,
                qry_sam_mask=qry_sam_mask,
                img_q=img_q,
                mask_q=mask_q,
                heat_before=heat_before,
                heat_after=heat_after,
            )

        if self.training:
            fg_q = self.masked_average_pooling(feature_q, (mask_q == 1).float())[None, :].squeeze(0)
            bg_q = self.masked_average_pooling(feature_q, (mask_q == 0).float())[None, :].squeeze(0)

            self_similarity_fg = F.cosine_similarity(feature_q, fg_q[..., None, None], dim=1)
            self_similarity_bg = F.cosine_similarity(feature_q, bg_q[..., None, None], dim=1)
            self_out = torch.cat((self_similarity_bg[:, None, ...], self_similarity_fg[:, None, ...]), dim=1) * 10.0

            self_out = F.interpolate(self_out, size=(h, w), mode="bilinear", align_corners=True)
            supp_out = torch.cat(supp_out_ls, 0)

            out_ls.append(self_out)
            out_ls.append(supp_out)
        
        mask_s = torch.cat(mask_s_list, dim=0)
        mask_s = mask_s.long()

        pred = torch.argmax(out_ls[0], dim = 1)

        # loss = self.criterion(out_ls[0], mask_q) + 0.1*loss_dstcp

        # loss = self.criterion(out_ls[0], mask_q)

        if self.training:
            loss = self.criterion(out_ls[0], mask_q) + self.criterion(out_ls[1], mask_q) + self.criterion(out_ls[2], mask_s) * 0.4
        else:
            loss = self.criterion(out_ls[0], mask_q)

        return pred, loss

    def get_nshot(self, suppport_images, support_masks):
        
        support_masks_list = []
        support_images_list = []

        for i in range(self.shot):
            mask = (support_masks[:, i, :, :]).float().squeeze(1)

            images_s = (suppport_images[:, i, :, :, :]).float().squeeze(1)

            support_masks_list.append(mask)

            support_images_list.append(images_s)
            
        return support_images_list, support_masks_list

    def SSP_func(self, feature_q, out, flag = True):
        device = feature_q.device
        bs,c= feature_q.shape[:2]
        pred_1 = out.softmax(1)
        pred_2 = pred_1.view(bs, 2, -1)
        pred_fg = pred_2[:, 1]
        pred_bg = pred_2[:, 0]
        fg_ls = []
        bg_ls = []
        fg_local_ls = []
        bg_local_ls = []
        for epi in range(bs):
            f_h, f_w = feature_q[epi].shape[-2:]
            fg_mask = torch.zeros(f_h, f_w).to(torch.int64).to(device)
            bg_mask = torch.zeros(f_h, f_w).to(torch.int64).to(device=device)

            fg_thres = 0.7
            bg_thres = 0.6
            cur_feat = feature_q[epi].view(c, -1)
            
            if (pred_fg[epi] > fg_thres).sum() > 0:
                fg_feat = cur_feat[:, (pred_fg[epi]>fg_thres)] #.mean(-1)
                fg_mask = pred_1[epi, 1, :, :] > fg_thres
            else:
                topk_fg = torch.topk(pred_fg[epi], 12).indices
                fg_feat = cur_feat[:, topk_fg] #.mean(-1)
                topk_coords = torch.stack((topk_fg// f_w, topk_fg % f_w), dim=1)
                fg_mask[topk_coords[:,0], topk_coords[:, 1]] = 1
            if (pred_bg[epi] > bg_thres).sum() > 0:
                bg_feat = cur_feat[:, (pred_bg[epi]>bg_thres)] #.mean(-1)
                bg_mask = pred_1[epi, 0, :, :] > bg_thres
            else:
                topk_bg = torch.topk(pred_bg[epi], 12).indices
                bg_feat = cur_feat[:, topk_bg] #.mean(-1)
                topk_coords_b = torch.stack((topk_bg// f_w, topk_bg % f_w), dim=1)
                bg_mask[topk_coords_b[:,0], topk_coords_b[:, 1]] = 1
            
            # global proto
            fg_proto = fg_feat.mean(-1)
            bg_proto = bg_feat.mean(-1)
            fg_ls.append(fg_proto.unsqueeze(0))
            bg_ls.append(bg_proto.unsqueeze(0))

            # local proto
            fg_feat_norm = fg_feat / torch.norm(fg_feat, 2, 0, True) # 1024, N1
            bg_feat_norm = bg_feat / torch.norm(bg_feat, 2, 0, True) # 1024, N2
            cur_feat_norm = cur_feat / torch.norm(cur_feat, 2, 0, True) # 1024, N3

            cur_feat_norm_t = cur_feat_norm.t() # N3, 1024
            fg_sim = torch.matmul(cur_feat_norm_t, fg_feat_norm) * 2.0 # N3, N1
            bg_sim = torch.matmul(cur_feat_norm_t, bg_feat_norm) * 2.0 # N3, N2

            fg_sim = fg_sim.softmax(-1)
            bg_sim = bg_sim.softmax(-1)

            fg_proto_local = torch.matmul(fg_sim, fg_feat.t()) # N3, 1024
            bg_proto_local = torch.matmul(bg_sim, bg_feat.t()) # N3, 1024

            fg_proto_local = fg_proto_local.t().view(c, f_h, f_w).unsqueeze(0) # 1024, N3
            bg_proto_local = bg_proto_local.t().view(c, f_h, f_w).unsqueeze(0) # 1024, N3

            fg_local_ls.append(fg_proto_local)
            bg_local_ls.append(bg_proto_local)

        # global proto
        new_fg = torch.cat(fg_ls, 0).unsqueeze(-1).unsqueeze(-1)
        new_bg = torch.cat(bg_ls, 0).unsqueeze(-1).unsqueeze(-1)

        # local proto
        new_fg_local = torch.cat(fg_local_ls, 0).unsqueeze(-1).unsqueeze(-1)
        new_bg_local = torch.cat(bg_local_ls, 0)

        return new_fg, new_bg, new_fg_local, new_bg_local
    

    def similarity_func(self, feature_q, fg_proto, bg_proto):
        similarity_fg = F.cosine_similarity(feature_q, fg_proto, dim=1)
        similarity_bg = F.cosine_similarity(feature_q, bg_proto, dim=1)

        out = torch.cat((similarity_bg[:, None, ...], similarity_fg[:, None, ...]), dim=1) * 10.0
        return out

    def masked_average_pooling(self, feature, mask):
        mask = F.interpolate(mask.unsqueeze(1), size=feature.shape[-2:], mode='bilinear', align_corners=True)
        masked_feature = torch.sum(feature * mask, dim=(2, 3)) \
                         / (mask.sum(dim=(2, 3)) + 1e-5)
        return masked_feature
    
    def masked_average_pooling2(self, feature, mask):
        b, c, w, h = feature.shape
        _, m, _, _ = mask.shape

        _mask = mask.view(b, m, -1)
        _feature = feature.view(b, c, -1).permute(0, 2, 1).contiguous() # b, h*w, c
        feature_sum = _mask @ _feature # b x m x c
        masked_sum = torch.sum(_mask, dim=2, keepdim=True) # b x m x 1

        masked_average_pooling = torch.div(feature_sum, masked_sum + 1e-5)
        return masked_average_pooling

    
    def iter_BFP(self, FP, BP, feature_s_ls, feature_q, refine=True):
        ###### input FP and BP are support prototype
        ###### SSP on query side
        ### find the most similar part in query feature
        out_0 = self.similarity_func(feature_q, FP, BP)
        ### SSP in query feature
        SSFP_1, SSBP_1, ASFP_1, ASBP_1 = self.SSP_func(feature_q, out_0, True)
        ### update prototype for query prediction
        FP_1 = FP * 0.5 + SSFP_1 * 0.5
        FP_use = FP_1.clone()
        BP_1 = SSBP_1 * 0.3 + ASBP_1 * 0.7
        BP_use = SSBP_1.clone()
        ### use updated prototype to search target in query feature
        out_1 = self.similarity_func(feature_q, FP_1, BP_1)
        ###### Refine (only for the 1st iter)
        if refine:
            ### use updated prototype to find the most similar part in query feature again
            SSFP_2, SSBP_2, ASFP_2, ASBP_2 = self.SSP_func(feature_q, out_1, True)
            ### update prototype again for query regine
            FP_2 = FP * 0.5 + SSFP_2 * 0.5
            BP_2 = SSBP_2 * 0.3 + ASBP_2 * 0.7
            FP_2 = FP * 0.5 + FP_1 * 0.2 + FP_2 * 0.3
            BP_2 = BP * 0.5 + BP_1 * 0.2 + BP_2 * 0.3
            ### use updated prototype to search target in query feature again
            out_refine = self.similarity_func(feature_q, FP_2, BP_2)
            out_refine = out_refine * 0.7 + out_1 * 0.3

        ###### SSP on support side
        if self.training:
            ### duplicate query prototype for support SSP if shot > 1
            if self.shot > 1:
                FP_nshot = FP.repeat_interleave(self.shot, dim=0)
                FP_1 = FP_1.repeat_interleave(self.shot, dim=0)
                BP_1 = BP_1.repeat_interleave(self.shot, dim=0)
            ### find the most similar part in support feature list
            supp_out_0 = self.similarity_func(feature_s_ls, FP_1, BP_1)
            ### SSP in support feature list
            SSFP_supp, SSBP_supp, ASFP_supp, ASBP_supp = self.SSP_func(feature_s_ls, supp_out_0, False)
            ### update prototype for support prediction
            if self.shot > 1:
                FP_supp = FP_nshot * 0.5 + SSFP_supp * 0.5
            else:
                FP_supp = FP * 0.5 + SSFP_supp * 0.5

            BP_supp = SSBP_supp * 0.3 + ASBP_supp * 0.7
            ### use updated prototype to search target in support feature list
            supp_out_1 = self.similarity_func(feature_s_ls, FP_supp, BP_supp)

            ### process prototype if shot > 1
            if self.shot > 1:
                for i in range(FP_supp.shape[0]//self.shot):
                    for j in range(self.shot):
                        # print("each FP_supp", FP_supp[i*self.shot+j])
                        if j == 0:
                            FP_supp_avg = FP_supp[i*self.shot+j]
                            pass
                            BP_supp_avg = BP_supp[i*self.shot+j]
                        else:
                            FP_supp_avg = FP_supp_avg + FP_supp[i*self.shot+j]
                            BP_supp_avg = BP_supp_avg + BP_supp[i*self.shot+j]

                    FP_supp_avg = FP_supp_avg/self.shot
                    BP_supp_avg = BP_supp_avg/self.shot
                    FP_supp_avg = FP_supp_avg.reshape(1,FP_supp.shape[1],FP_supp.shape[2],FP_supp.shape[3])
                    BP_supp_avg = BP_supp_avg.reshape(1,BP_supp.shape[1],BP_supp.shape[2],BP_supp.shape[3])
                    if i == 0:
                        new_FP_supp = FP_supp_avg
                        new_BP_supp = BP_supp_avg
                    else:
                        new_FP_supp = torch.cat((new_FP_supp,FP_supp_avg), dim=0)
                        new_BP_supp = torch.cat((new_BP_supp,BP_supp_avg), dim=0)

                FP_supp = new_FP_supp
                BP_supp = new_BP_supp          

        if refine:
            if self.training:
                return out_refine, out_1, supp_out_1, FP_supp, BP_supp, FP_use, FP_2, out_0
            else:
                return out_refine, out_1, FP_use, FP_2, out_0, BP_use
        else:
            if self.training:
                return out_1, supp_out_1, FP_supp, BP_supp
            else:
                return out_1
            
