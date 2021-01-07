# -*- coding: utf-8 -*-
"""


@author: Shirin
"""

def photometric_reconstruction_loss(tgt_img, ref_imgs, intrinsics, intrinsics_inv, depth, explainability_mask, pose, rotation_mode='euler', padding_mode='zeros', lambda_oob=0, qch=0.5, wssim=0.5):
    def one_scale(depth, explainability_mask, occ_masks):
        assert(explainability_mask is None or depth.size()[2:] == explainability_mask.size()[2:])
        assert(pose.size(1) == len(ref_imgs))

        reconstruction_loss = 0
        b, _, h, w = depth.size()
        downscale = tgt_img.size(2)/h

        tgt_img_scaled = nn.functional.adaptive_avg_pool2d(tgt_img, (h, w))
        ref_imgs_scaled = [nn.functional.adaptive_avg_pool2d(ref_img, (h, w)) for ref_img in ref_imgs]
        intrinsics_scaled = torch.cat((intrinsics[:, 0:2]/downscale, intrinsics[:, 2:]), dim=1)
        intrinsics_scaled_inv = torch.cat((intrinsics_inv[:, :, 0:2]*downscale, intrinsics_inv[:, :, 2:]), dim=2)

        weight = 1.

        for i, ref_img in enumerate(ref_imgs_scaled):
            current_pose = pose[:, i]

            ref_img_warped = inverse_warp(ref_img, depth[:,0], current_pose, intrinsics_scaled, intrinsics_scaled_inv, rotation_mode, padding_mode)
            valid_pixels = 1 - (ref_img_warped == 0).prod(1, keepdim=True).type_as(ref_img_warped)
            diff = (tgt_img_scaled - ref_img_warped) * valid_pixels
            ssim_loss = 1 - ssim(tgt_img_scaled, ref_img_warped) * valid_pixels
            oob_normalization_const = valid_pixels.nelement()/valid_pixels.sum()

            assert((oob_normalization_const == oob_normalization_const).item() == 1)

            if explainability_mask is not None:
                diff = diff * (1 - occ_masks[:,i:i+1])* explainability_mask[:,i:i+1].expand_as(diff)
                ssim_loss = ssim_loss * (1 - occ_masks[:,i:i+1])* explainability_mask[:,i:i+1].expand_as(ssim_loss)
            else:
                diff = diff *(1-occ_masks[:,i:i+1]).expand_as(diff)
                ssim_loss = ssim_loss*(1-occ_masks[:,i:i+1]).expand_as(ssim_loss)

            reconstruction_loss +=  (1- wssim)*weight*oob_normalization_const*(robust_l1(diff, q=qch) + wssim*ssim_loss.mean()) + lambda_oob*robust_l1(1 - valid_pixels, q=qch)
            assert((reconstruction_loss == reconstruction_loss).item() == 1)
            #weight /= 2.83
        return reconstruction_loss
    

def compute_errors(gt, pred, crop=True):
    abs_diff, abs_rel, sq_rel, a1, a2, a3 = 0,0,0,0,0,0
    batch_size = gt.size(0)

    '''
    crop used by Garg ECCV16 to reprocude Eigen NIPS14 results
    construct a mask of False values, with the same size as target
    and then set to True values inside the crop
    '''
    if crop:
        crop_mask = gt[0] != gt[0]
        y1,y2 = int(0.40810811 * gt.size(1)), int(0.99189189 * gt.size(1))
        x1,x2 = int(0.03594771 * gt.size(2)), int(0.96405229 * gt.size(2))
        crop_mask[y1:y2,x1:x2] = 1

    for current_gt, current_pred in zip(gt, pred):
        valid = (current_gt > 0) & (current_gt < 80)
        if crop:
            valid = valid & crop_mask

        valid_gt = current_gt[valid]
        valid_pred = current_pred[valid].clamp(1e-3, 80)

        valid_pred = valid_pred * torch.median(valid_gt)/torch.median(valid_pred)

        thresh = torch.max((valid_gt / valid_pred), (valid_pred / valid_gt))
        a1 += (thresh < 1.25).float().mean()
        a2 += (thresh < 1.25 ** 2).float().mean()
        a3 += (thresh < 1.25 ** 3).float().mean()

        abs_diff += torch.mean(torch.abs(valid_gt - valid_pred))
        abs_rel += torch.mean(torch.abs(valid_gt - valid_pred) / valid_gt)

        sq_rel += torch.mean(((valid_gt - valid_pred)**2) / valid_gt)

    return [metric / batch_size for metric in [abs_diff, abs_rel, sq_rel, a1, a2, a3]]