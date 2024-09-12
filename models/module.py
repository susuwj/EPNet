import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("..")


def homo_warping(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, Ndepth] o [B, Ndepth, H, W]
    # out: [B, C, Ndepth, H, W]
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    num_depth = depth_values.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
        #print(rot_xyz.shape, depth_values.shape)
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.view(batch, 1, num_depth,
                                                                                            -1)  # [B, 3, Ndepth, H*W]
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        proj_xy = proj_xyz[:, :2, :, :] / (proj_xyz[:, 2:3, :, :] + 1e-9) # [B, 2, Ndepth, H*W]
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy

    warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2), mode='bilinear',
                                   padding_mode='zeros',align_corners=True)
    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)

    return warped_src_fea

class Conv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, **kwargs):
        super(Conv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Conv3d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu=True, bn=True, **kwargs):
        super(Conv3d, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm3d(out_channels) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class FeatExt3(nn.Module):
    def __init__(self):
        super(FeatExt3, self).__init__()
        base_channels = 8
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 8, 5, 2, 2, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU())
        self.conv0_0 = Conv2d(base_channels, base_channels * 2, 3, stride=1, padding=1)
        self.conv0_1 = Conv2d(base_channels * 2, base_channels * 2, 3, stride=1, relu=False, padding=1)
        self.conv0_2 = Conv2d(base_channels, base_channels * 2, 1, stride=1, relu=False)

        self.conv1_0 = Conv2d(base_channels * 2, base_channels * 2, 3, stride=1, padding=1)
        self.conv1_1 = Conv2d(base_channels * 2, base_channels * 2, 3, stride=1, relu=False, padding=1)

        self.conv2_0 = Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1)
        self.conv2_1 = Conv2d(base_channels * 4, base_channels * 4, 3, stride=1, relu=False, padding=1)
        self.conv2_2 = Conv2d(base_channels * 2, base_channels * 4, 1, stride=2, relu=False)

        self.conv3_0 = Conv2d(base_channels * 4, base_channels * 4, 3, stride=1, padding=1)
        self.conv3_1 = Conv2d(base_channels * 4, base_channels * 4, 3, stride=1, relu=False, padding=1)

        self.conv4_0 = Conv2d(base_channels * 4, base_channels * 8, 3, stride=2, padding=1)
        self.conv4_1 = Conv2d(base_channels * 8, base_channels * 8, 3, stride=1, relu=False, padding=1)
        self.conv4_2 = Conv2d(base_channels * 4, base_channels * 8, 1, stride=2, relu=False)

        self.conv5_0 = Conv2d(base_channels * 8, base_channels * 8, 3, stride=1, padding=1)
        self.conv5_1 = Conv2d(base_channels * 8, base_channels * 8, 3, stride=1, relu=False, padding=1)

        self.conv6_0 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 3, 2, 1, 1, bias=False)
        self.conv6_1 = nn.Conv2d(base_channels * 8, base_channels * 4, 3, 1, 1, bias=False)
        self.conv6_2 = Conv2d(base_channels * 4, base_channels * 4, 3, stride=1, padding=1)
        self.conv6_3 = Conv2d(base_channels * 4, base_channels * 4, 3, stride=1, relu=False, padding=1)

        self.conv7_0 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 3, 2, 1, 1, bias=False)
        self.conv7_1 = nn.Conv2d(base_channels * 4, base_channels * 2, 3, 1, 1, bias=False)
        self.conv7_2 = Conv2d(base_channels * 2, base_channels * 2, 3, stride=1, padding=1)
        self.conv7_3 = Conv2d(base_channels * 2, base_channels * 2, 3, stride=1, relu=False, padding=1)

        self.final_conv_1 = nn.Conv2d(64, 32, 3, 1, 1, bias=False)
        self.final_conv_2 = nn.Conv2d(32, 32, 3, 1, 1, bias=False)
        self.final_conv_3 = nn.Conv2d(16, 16, 3, 1, 1, bias=False)

    def forward(self, x):
        x = self.init_conv(x)

        residual = x
        x = self.conv0_1(self.conv0_0(x))
        x += self.conv0_2(residual)
        x = nn.ReLU(inplace=True)(x)

        residual = x
        x = self.conv1_1(self.conv1_0(x))
        x += residual
        out3 = nn.ReLU(inplace=True)(x)


        residual = out3
        x = self.conv2_1(self.conv2_0(out3))
        x += self.conv2_2(residual)
        x = nn.ReLU(inplace=True)(x)
        residual = x
        x = self.conv3_1(self.conv3_0(x))
        x += residual
        out2 = nn.ReLU(inplace=True)(x)

        residual = out2
        x = self.conv4_1(self.conv4_0(out2))
        x += self.conv4_2(residual)
        x = nn.ReLU(inplace=True)(x)
        residual = x
        x = self.conv5_1(self.conv5_0(x))
        x += residual
        out1 = nn.ReLU(inplace=True)(x)

        x = self.conv6_0(out1)
        x = torch.cat([x, out2], 1)
        x = self.conv6_1(x)
        residual = x
        x = self.conv6_3(self.conv6_2(x))
        x += residual
        out2 = nn.ReLU(inplace=True)(x)

        x = self.conv7_0(out2)
        x = torch.cat([x, out3], 1)
        x = self.conv7_1(x)
        residual = x
        x = self.conv7_3(self.conv7_2(x))
        x += residual
        out3 = nn.ReLU(inplace=True)(x)

        outputs = {}
        outputs["stage1"] = self.final_conv_1(out1)
        outputs["stage2"] = self.final_conv_2(out2)
        outputs["stage3"] = self.final_conv_3(out3)

        return outputs


class RegVis(nn.Module):

    def __init__(self):
        super(RegVis, self).__init__()
        base_channels = 8

        self.conv = Conv3d(base_channels, 1, kernel_size=1, stride=1, padding=0)
        self.final_conv = nn.Conv3d(1, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        x = self.conv(x)
        x = self.final_conv(x)

        return x

class FeatExt3_ref(nn.Module):
    def __init__(self):
        super(FeatExt3_ref, self).__init__()
        base_channels = 8
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU())
        self.conv0_0 = Conv2d(base_channels * 1, base_channels * 1, 3, stride=1, padding=1)
        self.conv0_1 = Conv2d(base_channels * 1, base_channels * 1, 3, stride=1, relu=False, padding=1)
        self.conv0_2 = Conv2d(base_channels * 1, base_channels * 1, 1, stride=1, relu=False)

        self.conv1_0 = Conv2d(base_channels * 1, base_channels * 1, 3, stride=1, padding=1)
        self.conv1_1 = Conv2d(base_channels * 1, base_channels * 1, 3, stride=1, relu=False, padding=1)

        self.conv2_0 = Conv2d(base_channels * 1, base_channels * 2, 3, stride=2, padding=1)
        self.conv2_1 = Conv2d(base_channels * 2, base_channels * 2, 3, stride=1, relu=False, padding=1)
        self.conv2_2 = Conv2d(base_channels * 1, base_channels * 2, 1, stride=2, relu=False)

        self.conv3_0 = Conv2d(base_channels * 2, base_channels * 2, 3, stride=1, padding=1)
        self.conv3_1 = Conv2d(base_channels * 2, base_channels * 2, 3, stride=1, relu=False, padding=1)

        self.conv4_0 = Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1)
        self.conv4_1 = Conv2d(base_channels * 4, base_channels * 4, 3, stride=1, relu=False, padding=1)
        self.conv4_2 = Conv2d(base_channels * 2, base_channels * 4, 1, stride=2, relu=False)

        self.conv5_0 = Conv2d(base_channels * 4, base_channels * 4, 3, stride=1, padding=1)
        self.conv5_1 = Conv2d(base_channels * 4, base_channels * 4, 3, stride=1, relu=False, padding=1)

        self.conv6_0 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 3, 2, 1, 1, bias=False)
        self.conv6_1 = nn.Conv2d(base_channels * 4, base_channels * 2, 3, 1, 1, bias=False)
        self.conv6_2 = Conv2d(base_channels * 2, base_channels * 2, 3, stride=1, padding=1)
        self.conv6_3 = Conv2d(base_channels * 2, base_channels * 2, 3, stride=1, relu=False, padding=1)

        self.final_conv_1 = nn.Conv2d(32, 32, 3, 1, 1, bias=False)
        self.final_conv_2 = nn.Conv2d(16, 16, 3, 1, 1, bias=False)

    def forward(self, x):
        x = self.init_conv(x)

        residual = x
        x = self.conv0_1(self.conv0_0(x))
        x += self.conv0_2(residual)
        x = nn.ReLU(inplace=True)(x)

        residual = x
        x = self.conv1_1(self.conv1_0(x))
        x += residual
        out3 = nn.ReLU(inplace=True)(x)


        residual = out3
        x = self.conv2_1(self.conv2_0(out3))
        x += self.conv2_2(residual)
        x = nn.ReLU(inplace=True)(x)
        residual = x
        x = self.conv3_1(self.conv3_0(x))
        x += residual
        out2 = nn.ReLU(inplace=True)(x)

        residual = out2
        x = self.conv4_1(self.conv4_0(out2))
        x += self.conv4_2(residual)
        x = nn.ReLU(inplace=True)(x)
        residual = x
        x = self.conv5_1(self.conv5_0(x))
        x += residual
        out1 = nn.ReLU(inplace=True)(x)

        x = self.conv6_0(out1)
        x = torch.cat([x, out2], 1)
        x = self.conv6_1(x)
        residual = x
        x = self.conv6_3(self.conv6_2(x))
        x += residual
        out2 = nn.ReLU(inplace=True)(x)

        outputs = {}
        outputs["stage1"] = self.final_conv_1(out1)
        outputs["stage2"] = self.final_conv_2(out2)

        return outputs

class Refinement(nn.Module):
    def __init__(self, feat_channels):
        super(Refinement, self).__init__()
        base_channels = 8

        self.conv1_0 = nn.Sequential(
            Conv2d(1, base_channels, 3, 1, padding=1),
            Conv2d(base_channels, base_channels * 2, 3, 1, padding=1),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1))

        self.conv1_2 = nn.ConvTranspose2d(base_channels * 2, base_channels * 2, 3, 2, 1, 1, bias=False)

        self.conv2_0 = Conv2d(feat_channels+base_channels * 2, base_channels * 4, 3, stride=2, padding=1)
        self.conv2_1 = Conv2d(base_channels * 4, base_channels * 4, 3, stride=1, relu=False, padding=1)
        self.conv2_2 = Conv2d(feat_channels+base_channels * 2, base_channels * 4, 1, stride=2, relu=False)

        self.conv3_0 = Conv2d(base_channels * 4, base_channels * 4, 3, stride=1, padding=1)
        self.conv3_1 = Conv2d(base_channels * 4, base_channels * 4, 3, stride=1, relu=False, padding=1)

        self.conv4_0 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 3, 2, 1, 1, bias=False)
        self.conv4_1 = nn.Conv2d(feat_channels+base_channels * 4, base_channels * 2, 3, 1, 1, bias=False)
        self.conv4_2 = Conv2d(base_channels * 2, base_channels * 2, 3, stride=1, padding=1)
        self.conv4_3 = Conv2d(base_channels * 2, base_channels * 2, 3, stride=1, relu=False, padding=1)

        self.final_conv = nn.Conv2d(base_channels * 2, 1, 3, padding=1, bias=False)
        
    def forward(self, depth, img_feat):
        depth_mean = torch.mean(depth.reshape(depth.shape[0],-1), -1, keepdim=True)
        depth_std = torch.std(depth.reshape(depth.shape[0],-1), -1, keepdim=True)
        depth = (depth.unsqueeze(1) - depth_mean.unsqueeze(-1).unsqueeze(-1)) / depth_std.unsqueeze(-1).unsqueeze(-1)
        depth_min, _ = torch.min(depth.reshape(depth.shape[0],-1), -1, keepdim=True)
        depth_max,_ = torch.max(depth.reshape(depth.shape[0],-1), -1, keepdim=True)

        depth_feat = self.conv1_2(self.conv1_0(depth))
        cat = torch.cat((img_feat, depth_feat), dim=1)

        residual = cat
        x = self.conv2_1(self.conv2_0(cat))
        x += self.conv2_2(residual)
        x = nn.ReLU(inplace=True)(x)
        residual = x
        x = self.conv3_1(self.conv3_0(x))
        x += residual
        out1 = nn.ReLU(inplace=True)(x)

        x = self.conv4_0(out1)
        x = torch.cat([x, cat], 1)
        x = self.conv4_1(x)
        residual = x
        x = self.conv4_3(self.conv4_2(x))
        x += residual
        out2 = nn.ReLU(inplace=True)(x)

        res = self.final_conv(out2)

        res_ = torch.zeros_like(res)
        for i in range(res.shape[0]):
            res_[i] = torch.clamp(res[i], min=depth_min[i].cpu().item(), max=depth_max[i].cpu().item())
        depth = (res_ + F.interpolate(depth, scale_factor=2, mode='bilinear', align_corners=False)) * depth_std.unsqueeze(-1).unsqueeze(-1) + depth_mean.unsqueeze(-1).unsqueeze(-1)
        
        return res_.squeeze(1), depth.squeeze(1)


def get_depth_range_samples(depth_start, depth_num, depth_interval, dtype):
    depth = depth_start + depth_interval * torch.arange(0, depth_num, dtype=dtype,
                            device=depth_start.device).view(1, depth_num, 1, 1)
    return depth

def depth_regression(p, depth_values):
    #print(p.shape, depth_values.shape)
    if depth_values.dim() <= 2:
        # print("regression dim <= 2")
        depth_values = depth_values.view(*depth_values.shape, 1, 1)
    depth = torch.sum(p * depth_values, 1)

    return depth

from .homography import inverse_warping
def photometric_loss(inputs, depth_gt_ms, mask_ms, proj_matrices_ms, imgs, depth_loss_weights, backbone_only, weight):
    total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
    imgs = torch.unbind(imgs, 1)
    for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
        stage_idx = int(stage_key.replace("stage", "")) - 1
        if stage_idx<4:
           idx = stage_idx//2
        else:
           idx = stage_idx - 2 
        if backbone_only:
           depth_est = stage_inputs["depth"]
        else:
            if (stage_idx == 1) or (stage_idx == 3):
               depth_est = stage_inputs["depth_init_res"]
            else:
               depth_est = stage_inputs["depth"]

        proj_matrices = proj_matrices_ms["stage{}".format(str(idx+1))]
        proj_matrices = torch.unbind(proj_matrices, 1)
        depth_gt = depth_gt_ms["stage{}".format(str(idx+1))]
        mask = mask_ms["stage{}".format(str(idx+1))]
        mask = mask > 0.5

        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]
        ref_img, src_imgs = imgs[0], imgs[1:]
        ref_img = F.interpolate(ref_img, size=(depth_est.shape[-2], depth_est.shape[-1]), mode='bilinear', align_corners=False)
        i = 0
        ph_loss = 0
        for src_img, src_proj in zip(src_imgs, src_projs):
            src_img = F.interpolate(src_img, size=(depth_est.shape[-2], depth_est.shape[-1]), mode='bilinear', align_corners=False)
            warped_img, mask_est = inverse_warping(src_img, ref_proj, src_proj, depth_est)
            warped_img_gt, mask_gt = inverse_warping(src_img, ref_proj, src_proj, depth_gt)
            ph_loss += F.smooth_l1_loss(warped_img*mask.unsqueeze(1)*mask_gt*mask_est, warped_img_gt*mask.unsqueeze(1)*mask_gt*mask_est, reduction='mean')
        if depth_loss_weights is not None:
            total_loss += depth_loss_weights[idx] * ph_loss
        else:
            total_loss += 1.0 * ph_loss
    return total_loss*weight, ph_loss*weight

def mvsnet_loss(inputs, depth_gt_ms, mask_ms, proj_matrices, imgs, depth_interval, offsetnet_only, backbone_only, ph_w, **kwargs):
    depth_loss_weights = kwargs.get("dlossw", None)
    depth_interval = depth_interval.to(depth_gt_ms["stage1"].dtype)

    total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)

    for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
        stage_idx = int(stage_key.replace("stage", "")) - 1
        if stage_idx<4:
           idx = stage_idx//2
        else:
           idx = stage_idx - 2 
        depth_gt = depth_gt_ms["stage{}".format(str(idx+1))]
        mask = mask_ms["stage{}".format(str(idx+1))]
        mask = mask > 0.5

        if offsetnet_only:
            if (stage_idx == 2) or (stage_idx == 4):
               depth_est = inputs["stage{}".format(str(stage_idx))]["depth"]
               depth_res = (depth_gt - depth_est).abs()
               depth_res_scale = depth_res / depth_interval.unsqueeze(-1).unsqueeze(-1)
               depth_res = depth_res_scale[mask].mean()
               depth_loss_scale = depth_res_scale
            else:
               depth_res = None
            if depth_loss_weights is not None:
                if depth_res != None:
                   total_loss += depth_loss_weights[idx-1] * depth_res
                   res_loss = depth_res
            else:
                if depth_res != None:
                   total_loss += 1.0 * depth_res
                   res_loss = depth_res
            depth_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)

        elif backbone_only:
           depth_est = stage_inputs["depth"]
           depth_loss = (depth_est - depth_gt).abs()
           depth_loss_scale = depth_loss / depth_interval.unsqueeze(-1).unsqueeze(-1)
           depth_loss = depth_loss_scale[mask].mean()
           res_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
           if depth_loss_weights is not None:
               total_loss += depth_loss_weights[idx] * depth_loss
           else:
               total_loss += 1.0 * depth_loss
        else:

            if (stage_idx == 1) or (stage_idx == 3):
               depth_est = stage_inputs["depth_init_res"]
            else:
               depth_est = stage_inputs["depth"]

            depth_loss = (depth_est - depth_gt).abs()
            depth_loss_scale = depth_loss / depth_interval.unsqueeze(-1).unsqueeze(-1)
            depth_loss = depth_loss_scale[mask].mean()

            if (stage_idx == 2) or (stage_idx == 4):
               depth_est = inputs["stage{}".format(str(stage_idx))]["depth"]
               depth_res = (depth_gt - depth_est).abs()
               depth_res_scale = depth_res / depth_interval.unsqueeze(-1).unsqueeze(-1)
               depth_res = depth_res_scale[mask].mean()
            else:
               depth_res = None

            if depth_loss_weights is not None:
                if depth_res != None:
                   total_loss += depth_loss_weights[idx] * depth_loss
                   total_loss += depth_loss_weights[idx-1] * depth_res
                   res_loss = depth_res
                else:
                   total_loss += depth_loss_weights[idx] * depth_loss
            else:
                if depth_res != None:
                   total_loss += 1.0 * depth_loss
                   total_loss += 1.0 * depth_res
                   res_loss = depth_res
                else:
                   total_loss += 1.0 * depth_loss

    if offsetnet_only:
       ph_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
    else:
       ph_losses, ph_loss = photometric_loss(inputs, depth_gt_ms, mask_ms, proj_matrices, imgs, depth_loss_weights, backbone_only, weight=ph_w)
       total_loss += ph_losses

    less1 = (depth_loss_scale[mask] < 1.).to(depth_gt.dtype).mean()
    less3 = (depth_loss_scale[mask] < 3.).to(depth_gt.dtype).mean()

    return total_loss, depth_loss, ph_loss, res_loss, less1, less3

def mvsnet_loss_dtu(inputs, depth_gt_ms, mask_ms, proj_matrices, imgs, depth_interval, offsetnet_only, backbone_only, ph_w, **kwargs):
    depth_loss_weights = kwargs.get("dlossw", None)
    depth_interval = depth_interval.to(depth_gt_ms["stage1"].dtype)

    total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)

    for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
        stage_idx = int(stage_key.replace("stage", "")) - 1
        if stage_idx<4:
           idx = stage_idx//2
        else:
           idx = stage_idx - 2 
        depth_gt = depth_gt_ms["stage{}".format(str(idx+1))]
        mask = mask_ms["stage{}".format(str(idx+1))]
        mask = mask > 0.5

        if offsetnet_only:
            if (stage_idx == 2) or (stage_idx == 4):
               depth_est = inputs["stage{}".format(str(stage_idx))]["depth"]
               depth_res = F.smooth_l1_loss(depth_est[mask], depth_gt[mask], reduction='mean')
               depth_loss_less = (depth_est - depth_gt).abs()
            else:
               depth_res = None
            if depth_loss_weights is not None:
                if depth_res != None:
                   total_loss += depth_loss_weights[idx-1] * depth_res
                   res_loss = depth_res
            else:
                if depth_res != None:
                   total_loss += 1.0 * depth_res
                   res_loss = depth_res
            depth_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)

        elif backbone_only:
           depth_est = stage_inputs["depth"]
           depth_loss = F.smooth_l1_loss(depth_est[mask], depth_gt[mask], reduction='mean')
           depth_loss_less = (depth_est - depth_gt).abs()
           res_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
           if depth_loss_weights is not None:
               total_loss += depth_loss_weights[idx] * depth_loss
           else:
               total_loss += 1.0 * depth_loss
        else:

            if (stage_idx == 1) or (stage_idx == 3):
               depth_est = stage_inputs["depth_init_res"]
            else:
               depth_est = stage_inputs["depth"]

            depth_loss = F.smooth_l1_loss(depth_est[mask], depth_gt[mask], reduction='mean')
            depth_loss_less = (depth_est - depth_gt).abs()

            if (stage_idx == 2) or (stage_idx == 4):
               depth_est = inputs["stage{}".format(str(stage_idx))]["depth"]
               depth_res = F.smooth_l1_loss(depth_est[mask], depth_gt[mask], reduction='mean')
            else:
               depth_res = None

            if depth_loss_weights is not None:
                if depth_res != None:
                   total_loss += depth_loss_weights[idx] * depth_loss
                   total_loss += depth_loss_weights[idx-1] * depth_res
                   res_loss = depth_res
                else:
                   total_loss += depth_loss_weights[idx] * depth_loss
            else:
                if depth_res != None:
                   total_loss += 1.0 * depth_loss
                   total_loss += 1.0 * depth_res
                   res_loss = depth_res
                else:
                   total_loss += 1.0 * depth_loss

    if offsetnet_only:
       ph_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
    else:
       ph_losses, ph_loss = photometric_loss(inputs, depth_gt_ms, mask_ms, proj_matrices, imgs, depth_loss_weights, backbone_only, weight=ph_w)
       total_loss += ph_losses

    less1 = (depth_loss_less[mask] < 1.).to(depth_gt.dtype).mean()
    less3 = (depth_loss_less[mask] < 3.).to(depth_gt.dtype).mean()

    return total_loss, depth_loss, ph_loss, res_loss, less1, less3
