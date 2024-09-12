import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import *
from .regnet import CostRegNet

Align_Corners_Range = False

class OffsetNet(nn.Module):
    def __init__(self, residual):
        super(OffsetNet, self).__init__()
        if residual != 0:
           self.residual_net = Refinement(residual)

    def forward(self, inputs, feat):

        depth_init_res = inputs["depth"]
        inputs["depth_init_res"] = inputs["depth"]
        res, depth = self.residual_net(depth_init_res.detach(), feat)
        inputs["depth"] = depth
        inputs["res"] = res

        return inputs

class DepthNet(nn.Module):
    def __init__(self, flag, num_groups):
        super(DepthNet, self).__init__()
        self.flag = flag
        self.num_groups = num_groups
        if self.flag:
            self.reg = RegVis()
        self.reg_fuse = CostRegNet(in_channels=self.num_groups, base_channels=self.num_groups)

    def forward(self, features, proj_matrices, depth_values, num_depth, depth_interval=None, view_weights=None):
        proj_matrices = torch.unbind(proj_matrices, 1)

        assert depth_values.shape[1] == num_depth, "depth_values.shape[1]:{}  num_depth:{}".format(depth_values.shape[1], num_depth)

        # step 1. feature extraction
        # in: images; out: 32-channel feature maps
        ref_feature, src_features = features
        num_views = len(src_features)
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

        # step 2. differentiable homograph, build cost volume
        batch, feature_channel, height, width = ref_feature.size()
        ref_feature = ref_feature.view(batch, self.num_groups, feature_channel//self.num_groups, height, width)
        volume_sum = 0
        view_weight_sum = 0
        if view_weights == None:
            view_weights = []
            for src_fea, src_proj in zip(src_features, src_projs):
                #warpped features
                src_proj_new = src_proj[:, 0].clone()
                src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
                ref_proj_new = ref_proj[:, 0].clone()
                ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])
                warped_volume = homo_warping(src_fea, src_proj_new, ref_proj_new, depth_values)
                warped_volume = warped_volume.view(batch, self.num_groups, feature_channel//self.num_groups, num_depth, height, width)
                cost_volume = (warped_volume * ref_feature.unsqueeze(3)).sum(2)
                view_weight = self.reg(cost_volume)
                view_weight = torch.sigmoid(view_weight.squeeze(1))
                view_weight, _ = torch.max(view_weight, dim=1, keepdim=True)
                view_weights.append(view_weight)
                if self.training:
                    volume_sum = volume_sum + cost_volume * view_weight.unsqueeze(1)
                    view_weight_sum = view_weight_sum + view_weight.unsqueeze(1)
                else:
                    # TODO: this is only a temporal solution to save memory, better way?
                    volume_sum += cost_volume * view_weight
                    view_weight_sum += view_weight
                del warped_volume, cost_volume

            volume_sum = volume_sum / view_weight_sum
            view_weights = torch.cat(view_weights, dim=1)  # [B, N, 1, H, W]

            prob_volume = self.reg_fuse(volume_sum)

            prob_volume = F.softmax(prob_volume.squeeze(1), dim=1)
            depth = depth_regression(prob_volume, depth_values=depth_values)

            with torch.no_grad():
                # photometric confidence
                prob_volume_sum4 = 4 * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1), stride=1, padding=0).squeeze(1)
                depth_index = depth_regression(prob_volume, depth_values=torch.arange(num_depth, device=prob_volume.device, dtype=torch.float)).long()
                depth_index = depth_index.clamp(min=0, max=num_depth-1)
                photometric_confidence = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1)).squeeze(1)

            return {"depth": depth, "photometric_confidence": photometric_confidence.squeeze(1), "view_weights": view_weights.detach()}
        else:
            i = 0
            for src_fea, src_proj in zip(src_features, src_projs):
                #warpped features
                src_proj_new = src_proj[:, 0].clone()
                src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
                ref_proj_new = ref_proj[:, 0].clone()
                ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])
                warped_volume = homo_warping(src_fea, src_proj_new, ref_proj_new, depth_values)
                warped_volume = warped_volume.view(batch, self.num_groups, feature_channel//self.num_groups, num_depth, height, width)
                cost_volume = (warped_volume * ref_feature.unsqueeze(3)).sum(2)
                view_weight = view_weights[:, i]
                i = i + 1
                if self.training:
                    volume_sum = volume_sum + cost_volume * view_weight.unsqueeze(1).unsqueeze(1)
                    view_weight_sum = view_weight_sum + view_weight.unsqueeze(1).unsqueeze(1)
                else:
                    # TODO: this is only a temporal solution to save memory, better way?
                    volume_sum += cost_volume * view_weight.unsqueeze(1).unsqueeze(1)
                    view_weight_sum += view_weight.unsqueeze(1).unsqueeze(1)
                del warped_volume, cost_volume

            volume_sum = volume_sum / view_weight_sum

            prob_volume = self.reg_fuse(volume_sum)

            prob_volume = F.softmax(prob_volume.squeeze(1), dim=1)
            depth = depth_regression(prob_volume, depth_values=depth_values)


            with torch.no_grad():
                # photometric confidence
                prob_volume_sum4 = 4 * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1), stride=1, padding=0).squeeze(1)
                depth_index = depth_regression(prob_volume, depth_values=torch.arange(num_depth, device=prob_volume.device, dtype=torch.float)).long()
                depth_index = depth_index.clamp(min=0, max=num_depth-1)
                photometric_confidence = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1)).squeeze(1)
            return {"depth": depth, "photometric_confidence": photometric_confidence.squeeze(1)}


class MVSNet(nn.Module):
    def __init__(self, ndepths=[48, 32, 8], depth_interals_ratio=[4, 2, 1], num_groups=[8, 8, 8, 8 ,8], offsetnet_only=False, backbone_only=False):
        super(MVSNet, self).__init__()
        self.ndepths = ndepths
        self.depth_interals_ratio = depth_interals_ratio
        self.num_stage = len(ndepths)
        self.num_groups = num_groups

        self.offsetnet_only = offsetnet_only
        self.backbone_only = backbone_only

        assert len(ndepths) == len(depth_interals_ratio)

        self.feature = FeatExt3()
        self.ref_feature = FeatExt3_ref()
        self.feat_channel_ref = [0, 32, 0, 16, 0] 
        self.DepthNet = nn.ModuleList([DepthNet(i==0, self.num_groups[i]) for i in range(self.num_stage)])
        self.OffsetNet = nn.ModuleList([OffsetNet(self.feat_channel_ref[i]) for i in range(self.num_stage)])

    def forward(self, imgs, proj_matrices, depth_min, depth_max, depth_interval):
        depth_min, depth_max, depth_interval = depth_min.type(torch.float32), depth_max.type(torch.float32), depth_interval.type(torch.float32)
        # step 1. feature extraction
        B,N,C,H,W = imgs.shape

        if self.offsetnet_only:
           feats = self.ref_feature(imgs[:,0])
           with torch.no_grad():
               imgs = imgs.transpose(0, 1).reshape(N*B, C, H, W)
               features = self.feature(imgs)

        elif self.backbone_only:
            imgs = imgs.transpose(0, 1).reshape(N*B, C, H, W)
            features = self.feature(imgs)

        else:
            feats = self.ref_feature(imgs[:,0])

            imgs = imgs.transpose(0, 1).reshape(N*B, C, H, W)
            features = self.feature(imgs)

        outputs = {}
        depth, depth_start, view_weights = None, None, None
        for stage_idx in range(self.num_stage):
            if stage_idx<(self.num_stage-1):
               idx = stage_idx//2
            else:
               idx = stage_idx - 2 
            features_stage = features["stage{}".format(str(idx+1))]
            proj_matrices_stage = proj_matrices["stage{}".format(str(idx+1))]
            ref_feat, *src_feat = [features_stage[i * B:(i + 1) * B] for i in range(N)]
            features_stage = [ref_feat, src_feat]

            if (self.feat_channel_ref[stage_idx] != 0) & (self.backbone_only==False):
               feat = feats["stage{}".format(str(idx+1))]
               #print(feat.shape, stage_idx)
            else:
               feat = None

            if depth is not None:
                depth_start = depth.detach()
                if self.backbone_only:
                    depth_start = F.interpolate(depth_start.unsqueeze(1), size=(ref_feat.shape[-2], ref_feat.shape[-1]), mode='bilinear', align_corners=False) - self.ndepths[stage_idx] * depth_interval * self.depth_interals_ratio[stage_idx] / 2
                else:
                    depth_start = depth_start.unsqueeze(1) - self.ndepths[stage_idx] * depth_interval * self.depth_interals_ratio[stage_idx] / 2
                view_weights = F.interpolate(view_weights, size=(ref_feat.shape[-2], ref_feat.shape[-1]), mode='bilinear', align_corners=False)
            else:
                depth_start = depth_min.view(-1,1,1,1).repeat(1, 1, ref_feat.shape[-2], ref_feat.shape[-1])
                depth_interval = depth_interval.view(-1, 1, 1, 1)
            depth_range_samples = get_depth_range_samples(depth_start=depth_start, depth_num=self.ndepths[stage_idx], depth_interval=self.depth_interals_ratio[stage_idx] * depth_interval, dtype=ref_feat.dtype)
            if self.offsetnet_only:
                with torch.no_grad():
                    outputs_stage = self.DepthNet[stage_idx](features_stage, proj_matrices_stage,
                                          depth_values=depth_range_samples, depth_interval= self.depth_interals_ratio[stage_idx] * depth_interval, num_depth=self.ndepths[stage_idx], view_weights=view_weights)
                if self.feat_channel_ref[stage_idx] != 0:
                    outputs_stage = self.OffsetNet[stage_idx](outputs_stage, feat)
            elif self.backbone_only:
                outputs_stage = self.DepthNet[stage_idx](features_stage, proj_matrices_stage,
                                          depth_values=depth_range_samples, depth_interval= self.depth_interals_ratio[stage_idx] * depth_interval, num_depth=self.ndepths[stage_idx], view_weights=view_weights)
            else:
                outputs_stage = self.DepthNet[stage_idx](features_stage, proj_matrices_stage,
                                          depth_values=depth_range_samples, depth_interval= self.depth_interals_ratio[stage_idx] * depth_interval, num_depth=self.ndepths[stage_idx], view_weights=view_weights)
                if self.feat_channel_ref[stage_idx] != 0:
                    outputs_stage = self.OffsetNet[stage_idx](outputs_stage, feat)

            depth = outputs_stage['depth']
            outputs_stage['photometric_confidence'] = F.interpolate(outputs_stage['photometric_confidence'].unsqueeze(1), [H//2, W//2], mode='bilinear', align_corners=False).squeeze(1)
            outputs["stage{}".format(stage_idx + 1)] = outputs_stage
            outputs.update(outputs_stage)
            if view_weights == None:
                view_weights = outputs_stage['view_weights']

        return outputs
