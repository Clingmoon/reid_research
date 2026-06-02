import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
import os.path
import torch.nn.functional as F


from .vivim import MambaLayer
#from .spmamba import VSSBlock
from mamba.mamba_ssm.modules.srmamba import SRMamba
from mamba.mamba_ssm.modules.bimamba import BiMamba
from mamba.mamba_ssm.modules.mamba_simple import Mamba
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)
            
import clip.clip as clip
def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size):
    url = clip._MODELS[backbone_name]
    model_path1 = '/dataset_cc/Pretrain-models/ViT-B-16.pt'  # 不用下载,用下载好的
    model_path2 = '/YCY/Pretrained_models/ViT-B-16.pt'  # 不用下载,用下载好的
    if os.path.exists(model_path1):
        model_path = model_path1
    elif os.path.exists(model_path2):
        model_path = model_path2
    else:
        model_path = clip._download(url)
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)

    return model

class MTMBranch(nn.Module):
    """Multi-Temporal Mamba branch for a single stride.

    Processes video frames grouped into fragments. Each fragment's cls tokens
    are averaged, patch tokens are collected and reordered by similarity to the
    fragment cls, then fed into BiMamba for spatiotemporal mining.
    """

    def __init__(self, d_model=768, d_state=16, d_conv=4, expand=2, use_occlusion_mask=False):
        super(MTMBranch, self).__init__()
        self.d_model = d_model
        self.use_occlusion_mask = use_occlusion_mask
        self.sp_mamba_bi = nn.Sequential(
            nn.LayerNorm(d_model),
            BiMamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            ),
        )
        self.norm = nn.LayerNorm(d_model)
        self.sp_attention = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.Tanh(),
            nn.Linear(d_model // 4, 1)
        )
        if self.use_occlusion_mask:
            self.occlusion_attention = nn.Sequential(
                nn.Linear(d_model * 2, d_model // 4),
                nn.GELU(),
                nn.Linear(d_model // 4, 1),
                nn.Sigmoid()
            )
            # 第一层: Kaiming Normal (fan_out), bias=0
            nn.init.kaiming_normal_(self.occlusion_attention[0].weight, a=0, mode='fan_out')
            nn.init.constant_(self.occlusion_attention[0].bias, 0.0)
            # 第二层: Xavier Normal + 正偏置(+2.0)，初始信任所有 token
            nn.init.xavier_normal_(self.occlusion_attention[2].weight)
            nn.init.constant_(self.occlusion_attention[2].bias, 2.0)

    def reorder(self, reference, raw, return_debug=False):
        reference_norm = F.normalize(reference, dim=-1).unsqueeze(1)  # B, 1, D
        raw_norm = F.normalize(raw, dim=-1)                           # B, L, D
        raw_norm = torch.transpose(raw_norm, 1, 2)                    # B, D, L
        sim_raw = torch.bmm(reference_norm, raw_norm).squeeze(1)      # B, L  原始余弦相似度

        # ---- Occlusion-aware mask attention ----
        sim = sim_raw
        w_t = torch.ones_like(sim_raw)                                # 默认全信任
        if self.use_occlusion_mask:
            z_cls_expanded = reference.unsqueeze(1).expand(-1, raw.size(1), -1)  # B, L, D
            concat_feat = torch.cat([z_cls_expanded, raw], dim=-1)    # B, L, 2D
            w_t = self.occlusion_attention(concat_feat).squeeze(-1)   # B, L
            sim = sim_raw * w_t                                       # B, L  校准后相似度
        # -----------------------------------------

        _, indices = torch.sort(sim, descending=True)

        selected_patch_embedding = []
        for i in range(indices.size(0)):
            all_patch_embeddings_i = raw[i, :, :].squeeze()           # L, D
            top_k_embedding = torch.index_select(all_patch_embeddings_i, 0, indices[i])  # L, D
            top_k_embedding = top_k_embedding.unsqueeze(0)            # 1, L, D
            selected_patch_embedding.append(top_k_embedding)
        selected_patch_embedding = torch.cat(selected_patch_embedding, 0)  # B, L, D

        if return_debug:
            return selected_patch_embedding, sim, indices, sim_raw, w_t
        return selected_patch_embedding

    def forward(self, cls_tokens, patch_tokens, stride, return_debug=False):
        """
        cls_tokens:   (B, T, D)       -- class token of each frame
        patch_tokens: (B, T, N, D)    -- patch tokens of each frame
        stride:       int             -- fragment stride S
        Returns:      (B, D)          -- branch feature vector
        """
        B, T, N, D = patch_tokens.shape
        effective_T = (T // stride) * stride
        if effective_T < stride:
            # Not enough frames even for one fragment; fall back to average cls
            frag_cls = cls_tokens.mean(dim=1)                          # B, D
            frag_patches = patch_tokens.reshape(B, -1, D)              # B, T*N, D
            reorder_result = self.reorder(frag_cls, frag_patches, return_debug=return_debug)
            if return_debug:
                reordered_patches, sim, indices, sim_raw, w_t = reorder_result
            else:
                reordered_patches = reorder_result
            seq = torch.cat([frag_cls.unsqueeze(1), reordered_patches], dim=1)  # B, 1+T*N, D
            mamba_out = self.sp_mamba_bi(seq)                          # B, 1+T*N, D
            mamba_out = self.norm(mamba_out)
            attn = self.sp_attention(mamba_out)                        # B, 1+T*N, 1
            attn = torch.transpose(attn, 1, 2)                         # B, 1, 1+T*N
            attn = F.softmax(attn, dim=-1)
            frag_feat = torch.bmm(attn, mamba_out).squeeze(1)          # B, D
            if return_debug:
                return frag_feat, {"sim": sim, "indices": indices, "sim_raw": sim_raw, "w_t": w_t, "attn_weights": attn}
            return frag_feat

        cls_tokens = cls_tokens[:, :effective_T, :]
        patch_tokens = patch_tokens[:, :effective_T, :, :]
        num_fragments = effective_T // stride

        fragment_features = []
        debug_cache = None

        for i in range(num_fragments):
            start = i * stride
            end = start + stride
            frag_cls = cls_tokens[:, start:end, :].mean(dim=1)         # B, D
            frag_patches = patch_tokens[:, start:end, :, :].reshape(B, -1, D)  # B, stride*N, D

            reorder_result = self.reorder(
                frag_cls, frag_patches,
                return_debug=(return_debug and i == 0)
            )
            if return_debug and i == 0:
                reordered_patches, sim, indices, sim_raw, w_t = reorder_result
                debug_cache = {"sim": sim, "indices": indices, "sim_raw": sim_raw, "w_t": w_t}
            else:
                reordered_patches = reorder_result

            seq = torch.cat([frag_cls.unsqueeze(1), reordered_patches], dim=1)  # B, 1+stride*N, D
            mamba_out = self.sp_mamba_bi(seq)                          # B, 1+stride*N, D
            mamba_out = self.norm(mamba_out)
            attn = self.sp_attention(mamba_out)                        # B, 1+stride*N, 1
            attn = torch.transpose(attn, 1, 2)                         # B, 1, 1+stride*N
            attn = F.softmax(attn, dim=-1)
            frag_feat = torch.bmm(attn, mamba_out).squeeze(1)          # B, D
            fragment_features.append(frag_feat)

            if return_debug and i == 0:
                debug_cache["attn_weights"] = attn

        branch_feat = torch.stack(fragment_features, dim=1).mean(dim=1)  # B, D

        if return_debug:
            return branch_feat, debug_cache
        return branch_feat


class CLIMB(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg):
        super(CLIMB, self).__init__()
        self.model_name = cfg.MODEL.NAME

        self.in_planes = 768
        self.in_planes_proj = 512
        self.camera_num = camera_num
        self.view_num = view_num
        self.sie_coe = cfg.MODEL.SIE_COE   

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)
        
        self.classifier = nn.Linear(self.in_planes_proj+self.in_planes, num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0]-16)//cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1]-16)//cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]
        clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)
        clip_model.to("cuda")

        self.image_encoder = clip_model.visual
        
        # Trick: freeze patch projection for improved stability
        # https://arxiv.org/pdf/2104.02057.pdf
        for _, v in self.image_encoder.conv1.named_parameters():
            v.requires_grad_(False)     #冻结了第一层卷积 (conv1) 即 Patch Projection 层的权重
        print('Freeze patch projection layer with shape {}'.format(self.image_encoder.conv1.weight.shape))

        if cfg.MODEL.SIE_CAMERA and cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_CAMERA:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(view_num))

        self.classifier2 = nn.Linear(self.in_planes, num_classes, bias=False)
        self.classifier2.apply(weights_init_classifier)
        self.bottleneck_proj_sp = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_proj_sp.bias.requires_grad_(False)
        self.bottleneck_proj_sp.apply(weights_init_kaiming)
        self.sp_mamba_bi = nn.Sequential(
            nn.LayerNorm(768),
            BiMamba(
                d_model=768,
                d_state=16,
                d_conv=4,
                expand=2,
            ),
        )
        self.sp_mamba_raw = nn.Sequential(
            nn.LayerNorm(768),
            Mamba(
                d_model=768,
                d_state=16,
                d_conv=4,
                expand=2,
            ),
        )
        self.norm2_mamba = nn.LayerNorm(768)
        self.norm3_mamba = nn.LayerNorm(768)
        self.sp_attention = nn.Sequential(
            nn.Linear(768, 192),
            nn.Tanh(),
            nn.Linear(192, 1)
        )

        # Multi-Temporal Mamba branches (video only)
        use_occlusion_mask = getattr(cfg.MODEL, 'USE_OCCLUSION_MASK', False)
        self.use_occlusion_mask = use_occlusion_mask
        self.mtm_branches = nn.ModuleDict({
            's1': MTMBranch(d_model=768, d_state=16, d_conv=4, expand=2, use_occlusion_mask=use_occlusion_mask),
            's4': MTMBranch(d_model=768, d_state=16, d_conv=4, expand=2, use_occlusion_mask=use_occlusion_mask),
            's8': MTMBranch(d_model=768, d_state=16, d_conv=4, expand=2, use_occlusion_mask=use_occlusion_mask),
        })
        if self.use_occlusion_mask:
            self.occlusion_attention = nn.Sequential(
                nn.Linear(768 * 2, 768 // 4),
                nn.GELU(),
                nn.Linear(768 // 4, 1),
                nn.Sigmoid()
            )
            # 第一层: Kaiming Normal (fan_out), bias=0
            nn.init.kaiming_normal_(self.occlusion_attention[0].weight, a=0, mode='fan_out')
            nn.init.constant_(self.occlusion_attention[0].bias, 0.0)
            # 第二层: Xavier Normal + 正偏置(+2.0)，初始信任所有 token
            nn.init.xavier_normal_(self.occlusion_attention[2].weight)
            nn.init.constant_(self.occlusion_attention[2].bias, 2.0)
        self.mif_fusion = nn.Linear(768, 768)
        self.bottleneck_mtm = nn.BatchNorm1d(768)
        self.bottleneck_mtm.bias.requires_grad_(False)
        self.bottleneck_mtm.apply(weights_init_kaiming)

    def reorder(self, reference, raw, return_debug=False):

        # attention_map = attention_map.mean(axis=1)  # torch.Size([64, 50, 50])
        reference_norm = F.normalize(reference, dim=-1).unsqueeze(1)  # bt, 1, 768
        raw_norm = F.normalize(raw, dim=-1)  # bt, 128, 768
        raw_norm = torch.transpose(raw_norm, 1, 2) # bt, 768, 128
        sim_raw = torch.bmm(reference_norm, raw_norm).squeeze(1)  # [bt, 1, 768] [bt, 768, 128]= [bt, 128] 原始余弦相似度

        # ---- Occlusion-aware mask attention ----
        sim = sim_raw
        w_t = torch.ones_like(sim_raw)  # 默认全信任
        if self.use_occlusion_mask:
            z_cls_expanded = reference.unsqueeze(1).expand(-1, raw.size(1), -1)  # bt, 128, 768
            concat_feat = torch.cat([z_cls_expanded, raw], dim=-1)  # bt, 128, 1536
            w_t = self.occlusion_attention(concat_feat).squeeze(-1)  # bt, 128
            sim = sim_raw * w_t  # bt, 128  校准后相似度
        # -----------------------------------------

        _, indices = torch.sort(sim, descending=True)

        selected_patch_embedding = []
        for i in range(indices.size(0)):   #bs
          all_patch_embeddings_i = raw[i, :,:].squeeze()  # torch.Size([128, 768])
          top_k_embedding = torch.index_select(all_patch_embeddings_i, 0, indices[i])  # torch.Size([128, 768])
          top_k_embedding = top_k_embedding.unsqueeze(0)  # torch.Size([1, 128, 768])
          selected_patch_embedding.append(top_k_embedding)
        selected_patch_embedding = torch.cat(selected_patch_embedding, 0)  # torch.Size([64, 128, 768])

        if return_debug:
            return selected_patch_embedding, sim, indices, sim_raw, w_t
        return selected_patch_embedding

    def forward(self, x, get_image=False, cam_label=None, view_label=None, return_visuals=False):
        # Detect video input: (B, T, C, H, W) vs (B, C, H, W)
        is_video = len(x.shape) == 5
        if is_video:
            B, T, C, H, W = x.shape
            x = x.view(-1, C, H, W)  # (B*T, C, H, W)
            if cam_label is not None:
                cam_label = cam_label.repeat_interleave(T)
            if view_label is not None:
                view_label = view_label.repeat_interleave(T)

        if get_image == True:
            if hasattr(self, "cv_embed") and cam_label != None and view_label!=None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label * self.view_num + view_label]
            elif hasattr(self, "cv_embed") and cam_label != None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label]
            elif hasattr(self, "cv_embed") and view_label!=None:
                cv_embed = self.sie_coe * self.cv_embed[view_label]
            else:
                cv_embed = None
            _, image_features, image_features_proj, = self.image_encoder(x, cv_embed)
            img_feature = image_features[:,0]
            img_feature_proj = image_features_proj[:,0]

            if is_video:
                img_feature = img_feature.view(B, T, -1).mean(1)
                img_feature_proj = img_feature_proj.view(B, T, -1).mean(1)

            feat = self.bottleneck(img_feature)
            feat_proj = self.bottleneck_proj(img_feature_proj)

            out_feat = torch.cat([feat, feat_proj], dim=1)
            return out_feat

        if hasattr(self, "cv_embed") and cam_label != None and view_label != None:
            cv_embed = self.sie_coe * self.cv_embed[cam_label * self.view_num + view_label]
        elif hasattr(self, "cv_embed") and cam_label != None:
            cv_embed = self.sie_coe * self.cv_embed[cam_label]
        elif hasattr(self, "cv_embed") and view_label != None:
            cv_embed = self.sie_coe * self.cv_embed[view_label]
        else:
            cv_embed = None
        _, image_features, image_features_proj, = self.image_encoder(x, cv_embed)
        img_feature = image_features[:, 0]
        img_feature_proj = image_features_proj[:, 0]

        if is_video:
            img_feature = img_feature.view(B, T, -1).mean(1)
            img_feature_proj = img_feature_proj.view(B, T, -1).mean(1)

        feat = self.bottleneck(img_feature)
        feat_proj = self.bottleneck_proj(img_feature_proj)

        out_feat = torch.cat([feat, feat_proj], dim=1)

        collect_visuals = return_visuals and not self.training

        if is_video:
            # ===== MTM: Multi-Temporal Mamba =====
            feats_for_mamba = image_features.detach()                    # (B*T, 1+N, D)
            cls_tokens = feats_for_mamba[:, 0, :].view(B, T, -1)         # (B, T, D)
            patch_tokens = feats_for_mamba[:, 1:, :].view(B, T, -1, feats_for_mamba.size(-1))  # (B, T, N, D)

            if collect_visuals:
                o_s1, debug_s1 = self.mtm_branches['s1'](cls_tokens, patch_tokens, stride=1, return_debug=True)
                reorder_sim = debug_s1["sim"]
                reorder_indices = debug_s1["indices"]
                reorder_sim_raw = debug_s1.get("sim_raw", None)
                reorder_w_t = debug_s1.get("w_t", None)
                A = debug_s1["attn_weights"]
            else:
                o_s1 = self.mtm_branches['s1'](cls_tokens, patch_tokens, stride=1)
                reorder_sim, reorder_indices, reorder_sim_raw, reorder_w_t, A = None, None, None, None, None

            o_s4 = self.mtm_branches['s4'](cls_tokens, patch_tokens, stride=4)
            o_s8 = self.mtm_branches['s8'](cls_tokens, patch_tokens, stride=8)

            feat_sp = self.bottleneck_mtm(self.mif_fusion(o_s1 + o_s4 + o_s8))
        else:
            # ===== Image branch: S=1 IRM (original logic) =====
            feats_for_mamba = image_features.detach()
            feats_for_mamba_sp = feats_for_mamba[:, 1:, :].detach()
            feats_for_mamba_cls = feats_for_mamba[:, 0, :].detach()

            if collect_visuals:
                re_order_mamba_sp, reorder_sim, reorder_indices, reorder_sim_raw, reorder_w_t = self.reorder(
                    feats_for_mamba_cls, feats_for_mamba_sp, return_debug=True
                )
            else:
                re_order_mamba_sp = self.reorder(feats_for_mamba_cls, feats_for_mamba_sp)
                reorder_sim, reorder_indices, reorder_sim_raw, reorder_w_t = None, None, None, None

            mamba_sp_out = self.sp_mamba_bi(re_order_mamba_sp)
            mamba_sp_out = torch.cat((feats_for_mamba_cls.unsqueeze(1), mamba_sp_out), dim=1)
            mamba_sp_out2 = self.norm2_mamba(mamba_sp_out)
            A = self.sp_attention(mamba_sp_out2)
            A = torch.transpose(A, 1, 2)
            A = F.softmax(A, dim=-1)
            mamba_sp_out2 = torch.bmm(A, mamba_sp_out2)
            mamba_sp_out2 = mamba_sp_out2.squeeze(1)
            feat_sp = self.bottleneck_proj_sp(mamba_sp_out2)

        if self.training:
            logit = self.classifier(out_feat)
            logitsp = self.classifier2(feat_sp)
            return out_feat, logit, feat_sp, logitsp
        else:
            feat_concat = torch.cat((out_feat, feat_sp), dim=1)
            if return_visuals:
                visual_tensors = {
                    "sim": reorder_sim,
                    "sim_raw": reorder_sim_raw,
                    "w_t": reorder_w_t,
                    "indices": reorder_indices,
                    "attn_weights": A,
                }
                return feat_concat, out_feat, feat_sp, visual_tensors
            return feat_concat, out_feat, feat_sp
            


    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if not self.training and 'classifier' in i:
                continue # ignore classifier weights in evaluation
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


def make_model(cfg, num_classes, camera_num, view_num):
    model = CLIMB(num_classes, camera_num, view_num, cfg)
    return model
