# In cxas/losses/compound_losses.py

import torch
from torch import nn
import torch.nn.functional as F
from typing import Callable, Optional, Any, Tuple, Type, Dict, List

# Ensure torch.distributed is available if DDP is used
# It's good practice to check this or ensure it's initialized before operations.
# For simplicity, we assume it's set up when AllGatherGrad is called.

class AllGatherGrad(torch.autograd.Function):
    """
    Stolen from PyTorch Lightning.
    Performs an all-gather operation with a custom backward pass that
    ensures gradients are correctly propagated across distributed processes.
    """
    @staticmethod
    def forward(
        ctx: Any, # Context object to save information for backward pass
        tensor: torch.Tensor,
        group: Optional["torch.distributed.ProcessGroup"] = None,
    ) -> torch.Tensor:
        """
        Forward pass for AllGatherGrad. Gathers tensors from all processes.

        Args:
            ctx (Any): The context object.
            tensor (torch.Tensor): The tensor to be gathered from the current process.
            group (Optional[torch.distributed.ProcessGroup]): The process group to work on.

        Returns:
            torch.Tensor: A tensor containing the gathered tensors from all processes,
                          stacked along a new dimension.
        """
        ctx.group = group

        world_size: int = torch.distributed.get_world_size(group)
        gathered_tensor: List[torch.Tensor] = [torch.zeros_like(tensor) for _ in range(world_size)]

        torch.distributed.all_gather(gathered_tensor, tensor, group=group)
        
        gathered_tensor = torch.stack(gathered_tensor, dim=0)

        return gathered_tensor

    @staticmethod
    def backward(ctx: Any, *grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """
        Backward pass for AllGatherGrad. Reduces gradients across distributed processes.

        Args:
            ctx (Any): The context object.
            *grad_output (torch.Tensor): Gradients of the output of the forward pass.

        Returns:
            Tuple[torch.Tensor, None]: A tuple containing the gradient for the input tensor
                                      and None for the group (as it doesn't require gradients).
        """
        grad_output_combined: torch.Tensor = torch.cat(grad_output)

        torch.distributed.all_reduce(grad_output_combined, op=torch.distributed.ReduceOp.SUM, async_op=False, group=ctx.group)

        return grad_output_combined[torch.distributed.get_rank()], None


class MemoryEfficientSoftDiceLoss(nn.Module):
    """
    Calculates the Memory-Efficient Soft Dice Loss.
    This implementation handles multi-class segmentation, batch-wise Dice calculation,
    optional background class exclusion, and Distributed Data Parallel (DDP) aggregation.
    It can now accept `class_weights` as either a 1D tensor (per-class) or a 2D tensor
    (batch-specific per-class weights).
    """
    def __init__(self, 
                 apply_nonlin: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                 batch_dice: bool = False, 
                 do_bg: bool = True, 
                 smooth: float = 1.0,
                 ddp: bool = True) -> None:
        """
        Initializes the MemoryEfficientSoftDiceLoss.

        Args:
            apply_nonlin (Optional[Callable]): A callable (e.g., torch.sigmoid, torch.softmax)
                                               to apply to the model output before Dice calculation.
                                               If None, the input `x` is assumed to be probabilities.
            batch_dice (bool): If True, calculate Dice globally across the batch. If False,
                               calculate Dice per sample and then average.
            do_bg (bool): If True, include the background class (class 0) in Dice calculation.
                          If False, exclude it.
            smooth (float): Smoothing factor (epsilon) to prevent division by zero.
            ddp (bool): If True, use AllGatherGrad for DDP-compatible batch Dice calculation.
                        Requires torch.distributed to be initialized.
        """
        super().__init__()

        self.do_bg: bool = do_bg
        self.batch_dice: bool = batch_dice
        self.apply_nonlin: Optional[Callable[[torch.Tensor], torch.Tensor]] = apply_nonlin
        self.smooth: float = smooth
        self.ddp: bool = ddp

    def forward(self, 
                x: torch.Tensor, 
                y: torch.Tensor, 
                loss_mask: Optional[torch.Tensor] = None, 
                class_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculates the Soft Dice Loss.

        Args:
            x (torch.Tensor): Model output (logits or probabilities, depending on apply_nonlin).
                              Expected shape: (N, C, H, W) or (N, C, D, H, W).
            y (torch.Tensor): Ground truth. Can be class indices (N, H, W) or (N, D, H, W),
                              or one-hot encoded (N, C, H, W) or (N, C, D, H, W).
            loss_mask (Optional[torch.Tensor]): Mask to exclude certain regions from loss calculation.
                                                Expected shape: (N, 1, H, W) or (N, H, W) (spatial dims only),
                                                or (N, 1, D, H, W) for 3D. Values should be 0 (ignore) or 1 (include).
            class_weights (Optional[torch.Tensor]): Per-class weights. Can be:
                                                    - (C_effective,) for uniform weights across batch.
                                                    - (N, C_effective) for batch-specific weights.

        Returns:
            torch.Tensor: Scalar Dice Loss.
        """
        if self.apply_nonlin is not None:
            # Apply non-linearity (e.g., sigmoid for binary, softmax for multi-class)
            x = self.apply_nonlin(x) # x is now probabilities after non-linearity

        # Determine spatial axes (e.g., (2, 3) for 2D images, (2, 3, 4) for 3D images)
        # These are the dimensions over which sums are performed for Dice calculation.
        axes: Tuple[int, ...] = tuple(range(2, x.ndim)) 

        # --- Prepare Ground Truth (y_onehot) ---
        with torch.no_grad(): # Operations on y (ground truth) should not track gradients
            # Ensure y (ground truth) is in a consistent shape for one-hot conversion
            # If y.ndim is less than x.ndim, it means y is likely class indices.
            if x.ndim != y.ndim: 
                # Add channel dimension if missing, e.g., (N, H, W) -> (N, 1, H, W)
                y = y.view((y.shape[0], 1, *y.shape[1:])) 

            y_onehot: torch.Tensor
            # Convert ground truth to one-hot encoding if not already.
            # Assumes x.shape[1] is the number of classes.
            if x.shape == y.shape:
                # If shapes match, y is probably already one-hot. Ensure float type.
                y_onehot = y.float() 
            else:
                # y is (N, 1, ...) with class indices. Convert to (N, C, ...) one-hot.
                # Use x.dtype for consistency in the one-hot tensor.
                y_onehot = torch.zeros(x.shape, device=x.device, dtype=x.dtype)
                y_onehot.scatter_(1, y.long(), 1) # Scatter 1s at the class indices

            # Handle exclusion of background class from y_onehot if not desired
            if not self.do_bg:
                # If do_bg is False, we typically assume class 0 is background.
                # Slice y_onehot to exclude the first channel.
                y_onehot = y_onehot[:, 1:] 

            # Calculate sum of ground truth pixels for each class (used in Dice denominator).
            # If loss_mask is provided, apply it element-wise.
            sum_gt: torch.Tensor = y_onehot.sum(axes) 
            if loss_mask is not None:
                # Ensure loss_mask is broadcastable and float for multiplication.
                # It might need unsqueezing if it's (N, H, W) and x is (N, C, H, W).
                # Expand mask to match feature map dimensions if it's only spatial.
                if loss_mask.ndim == y_onehot.ndim - 1: # e.g. (N, H, W) for (N,C,H,W)
                    mask_expanded = loss_mask.unsqueeze(1).float()
                else: # Assume (N,1,H,W) or (N,1,D,H,W)
                    mask_expanded = loss_mask.float()
                sum_gt = (y_onehot * mask_expanded).sum(axes)


        # --- Calculate intersection and sum of predicted pixels ---
        # This one MUST be outside the with torch.no_grad(): context to allow gradient flow.
        # Handle exclusion of background class from x (predictions) if not desired
        x_processed: torch.Tensor = x
        if not self.do_bg:
            x_processed = x[:, 1:] # Exclude background from predictions

        # Calculate intersection (product of probabilities and one-hot ground truth)
        intersect: torch.Tensor = (x_processed * y_onehot).sum(axes)
        # Calculate sum of predicted probabilities
        sum_pred: torch.Tensor = x_processed.sum(axes)

        if loss_mask is not None:
            # Apply loss_mask to intersection and sum_pred if provided.
            # Use the same expanded mask as for sum_gt.
            if loss_mask.ndim == x_processed.ndim - 1:
                mask_expanded_for_x = loss_mask.unsqueeze(1).float()
            else:
                mask_expanded_for_x = loss_mask.float()

            intersect = (x_processed * y_onehot * mask_expanded_for_x).sum(axes)
            sum_pred = (x_processed * mask_expanded_for_x).sum(axes)

        # --- Handle Distributed Data Parallel (DDP) aggregation ---
        if self.batch_dice and torch.distributed.is_initialized(): # Check if DDP is initialized
            intersect = AllGatherGrad.apply(intersect).sum(0) # Sum across batch dimension (first dim after gather)
            sum_pred = AllGatherGrad.apply(sum_pred).sum(0)
            sum_gt = AllGatherGrad.apply(sum_gt).sum(0) # sum_gt comes from `with no_grad` but needs to be gathered

        denominator_val = sum_gt + sum_pred + self.smooth


        # Calculate Dice coefficient (dc) for each class
        # Add smooth factor to numerator and denominator to prevent division by zero.
        denominator: torch.Tensor = sum_gt + sum_pred + self.smooth
        # Clamp denominator to ensure it's never less than smooth, avoiding NaN/inf
        dc: torch.Tensor = (2 * intersect + self.smooth) / torch.clamp_min(denominator, self.smooth) 

        # --- Apply class_weights and compute final Dice Loss (1 - Dice_Coefficient) ---
        # `class_weights` can be (C_effective,) or (N, C_effective).
        final_loss: torch.Tensor
        if class_weights is not None:
            class_weights = class_weights / class_weights.sum()

            # Reshape class_weights for broadcasting with `dc`
            if class_weights.ndim == 1:
                if self.batch_dice: # dc is (C_effective,)
                     weights_reshaped = class_weights # (C_effective,)
                else: # dc is (N, C_effective)
                     weights_reshaped = class_weights.unsqueeze(0) # (1, C_effective)
            elif class_weights.ndim == 2: # (N, C_effective)
                # If batch_dice is True, the individual sample weights in the batch dimension
                # will be lost when dc is summed across the batch. This means batch-specific
                # weights are generally only meaningful when batch_dice is False.
                if self.batch_dice:
                    print("Warning: Batch-specific class_weights (2D) provided but batch_dice is True. "
                          "Weights will be summed/averaged across the batch before application, potentially losing per-sample distinction.")
                    weights_reshaped = class_weights.sum(dim=0) 
                else: # dc is (N, C_effective), weights are (N, C_effective)
                    weights_reshaped = class_weights 
            else:
                raise ValueError(f"class_weights for Dice Loss expected to be 1D (C_effective,) or 2D (N, C_effective), got {class_weights.shape}")
            
            # Weighted loss per class (or per sample per class if batch_dice=False)
            loss_per_class: torch.Tensor = (1.0 - dc) * weights_reshaped 
            final_loss = loss_per_class.sum() 
        else:
            # Calculate mean of (1 - Dice_coefficient) across all classes (unweighted)
            final_loss = (1.0 - dc).mean()

        return final_loss


class DC_and_BCE_loss(nn.Module):
    """
    Combined Dice and Binary Cross-Entropy Loss.
    This loss function expects logits from the network's final layer.
    It combines a BCEWithLogitsLoss with a MemoryEfficientSoftDiceLoss.
    It can now accept `class_weights` as either a 1D tensor (per-class) or a 2D tensor
    (batch-specific per-class weights).
    """
    def __init__(self,
                 bce_kwargs: Dict[str, Any],
                 soft_dice_kwargs: Dict[str, Any],
                 weight_ce: float = 1.0,
                 weight_dice: float = 1.0,
                 use_ignore_label: bool = False,
                 dice_class: Type[MemoryEfficientSoftDiceLoss] = MemoryEfficientSoftDiceLoss) -> None:
        super().__init__()

        if 'reduction' not in bce_kwargs or bce_kwargs['reduction'] != 'none':
            print("Warning: BCEWithLogitsLoss reduction is not 'none'. Forcing to 'none' for manual weighting.")
            bce_kwargs['reduction'] = 'none'

        self.weight_dice: float = weight_dice
        self.weight_ce: float = weight_ce
        self.use_ignore_label: bool = use_ignore_label

        self.dice_do_bg: bool = soft_dice_kwargs.get('do_bg', True)

        self.bce: nn.BCEWithLogitsLoss = nn.BCEWithLogitsLoss(**bce_kwargs)

        # Instantiate Dice loss. It will apply sigmoid internally as specified by apply_nonlin.
        self.dc: MemoryEfficientSoftDiceLoss = dice_class(apply_nonlin=torch.sigmoid, **soft_dice_kwargs)

    def forward(self,
                net_output: torch.Tensor,
                target: torch.Tensor,
                class_weights: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for the combined loss.

        Args:
            net_output (torch.Tensor): Model output (logits). Shape (N, C_total, H, W) or (N, C_total, D, H, W).
            target (torch.Tensor): Ground truth.
                                   If `use_ignore_label` is False:
                                       Shape (N, H, W) (class indices)
                                       OR (N, C_total, H, W) (one-hot, float).
                                   If `use_ignore_label` is True:
                                       Shape (N, C_total_model_output + 1, H, W) (one-hot, float),
                                       where the last channel is the ignore mask.
            class_weights (Optional[torch.Tensor]): Per-class weights. Expected shape (C_total_effective,),
                                                    where C_total_effective is the number of classes after
                                                    potential ignore label removal. These weights are applied
                                                    to both BCE and Dice components.

        Returns:
            torch.Tensor: Scalar total loss.
        """
        num_classes_model_output: int = net_output.shape[1]

        # --- 1. Prepare targets and masks based on `use_ignore_label` ---
        mask_for_loss: Optional[torch.Tensor] = None
        target_regions_for_bce: torch.Tensor = target
        net_output_for_bce: torch.Tensor = net_output

        if self.use_ignore_label:
            if target.ndim != net_output.ndim or target.shape[1] != num_classes_model_output + 1:
                raise ValueError(
                    f"When use_ignore_label=True, target must be one-hot and have an extra channel "
                    f"for the ignore mask. Expected target shape to be (N, {num_classes_model_output + 1}, H, W), "
                    f"but got {target.shape} with net_output {net_output.shape}."
                )
            mask_channel: torch.Tensor = target[:, -1:]
            mask_for_loss = (1.0 - mask_channel).bool()
            target_regions_for_bce = target[:, :-1]
            net_output_for_bce = net_output

        # --- 2. Convert `target_regions_for_bce` to one-hot float for BCEWithLogitsLoss if necessary ---
        if target_regions_for_bce.ndim == net_output_for_bce.ndim - 1:
            num_classes_for_onehot: int = net_output_for_bce.shape[1]
            permute_dims: List[int] = [0, target_regions_for_bce.ndim] + list(range(1, target_regions_for_bce.ndim))
            target_regions_for_bce = F.one_hot(target_regions_for_bce.long(), num_classes_for_onehot).permute(permute_dims).float()
        elif target_regions_for_bce.ndim == net_output_for_bce.ndim and target_regions_for_bce.dtype != torch.float:
            target_regions_for_bce = target_regions_for_bce.float()

        # --- 3. Prepare `class_weights` for BCE & Dice ---
        class_weights_for_bce: Optional[torch.Tensor] = class_weights
        class_weights_for_dice: Optional[torch.Tensor] = class_weights

        if class_weights is not None:
            if self.use_ignore_label:
                if class_weights.ndim == 1:
                    class_weights_for_bce = class_weights[:-1]
                    class_weights_for_dice = class_weights[:-1]
                elif class_weights.ndim == 2:
                    class_weights_for_bce = class_weights[:, :-1]
                    class_weights_for_dice = class_weights[:, :-1]
                else:
                    raise ValueError(f"class_weights has unexpected ndim: {class_weights.ndim} for ignore label processing. Expected 1 or 2.")

            if not self.dice_do_bg and class_weights_for_dice is not None:
                if class_weights_for_dice.ndim == 1:
                    class_weights_for_dice = class_weights_for_dice[1:]
                elif class_weights_for_dice.ndim == 2:
                    class_weights_for_dice = class_weights_for_dice[:, 1:]
                else:
                    raise ValueError(f"class_weights_for_dice has unexpected ndim: {class_weights_for_dice.ndim} for do_bg=False processing. Expected 1 or 2.")

        bce_class_weights_reshaped: Optional[torch.Tensor] = None
        if class_weights_for_bce is not None:
            if class_weights_for_bce.ndim == 1:
                bce_class_weights_reshaped = class_weights_for_bce.view(1, -1, *([1] * (net_output_for_bce.ndim - 2)))
            elif class_weights_for_bce.ndim == 2:
                bce_class_weights_reshaped = class_weights_for_bce.unsqueeze(-1).unsqueeze(-1)
                if net_output_for_bce.ndim == 5:
                    bce_class_weights_reshaped = bce_class_weights_reshaped.unsqueeze(-1)
            else:
                raise ValueError(f"class_weights_for_bce has unexpected ndim: {class_weights_for_bce.ndim}. Expected 1 or 2.")

        # --- 4. Calculate BCE Loss ---
        bce_loss_unreduced: torch.Tensor = self.bce(net_output_for_bce, target_regions_for_bce)

        bce_loss_weighted: torch.Tensor
        if bce_class_weights_reshaped is not None:
            bce_loss_weighted = bce_loss_unreduced * bce_class_weights_reshaped
        else:
            bce_loss_weighted = bce_loss_unreduced

        ce_loss: torch.Tensor
        if mask_for_loss is not None:
            num_active_elements: torch.Tensor = mask_for_loss.float().sum()
            ce_loss = (bce_loss_weighted * mask_for_loss.float()).sum() / torch.clamp_min(num_active_elements, 1e-8)
        else:
            ce_loss = bce_loss_weighted.mean()

        # >>> START OF MODIFICATION FOR BCE SCALING <<<
        # Get the number of classes actually used in the BCE calculation (after potential ignore label slicing)
        num_classes_bce_effective = net_output_for_bce.shape[1]

        
        # Scale BCE loss. For multi-class BCE, dividing by log(num_classes) can normalize it
        # so that it's often closer to 1 when the model is randomly guessing.
        # Check for num_classes_bce_effective > 1 to avoid log(1) issues.
        if num_classes_bce_effective > 1:
            k = torch.tensor(3.0, device=ce_loss.device)
            # Using the natural logarithm (base e)
            original_scaling_factor = torch.log(torch.tensor(float(num_classes_bce_effective/2), device=ce_loss.device))
            original_scaling_factor = torch.clamp_min(original_scaling_factor, 1e-8) 
            # Ensure scaling factor is not too small (e.g., if num_classes_bce_effective is 1 which shouldn't happen here)
            transition = torch.sigmoid(k * (ce_loss - 1))
            bce_scaling_factor = transition * original_scaling_factor + (1 - transition) * 1.0
            
            ce_loss_scaled = ce_loss / bce_scaling_factor
        else: # If for some reason only 1 effective class (unlikely for multi-class segmentation)
            ce_loss_scaled = ce_loss # No scaling needed or possible meaningfully

        # --- 5. Calculate Dice Loss ---
        dc_loss: torch.Tensor = self.dc(net_output, target, loss_mask=mask_for_loss, class_weights=class_weights_for_dice)


        # --- 6. Combine and Return ---
        # Use the scaled BCE loss component
        result: torch.Tensor = self.weight_ce * ce_loss_scaled + self.weight_dice * dc_loss
        # >>> END OF MODIFICATION FOR BCE SCALING <<<
        return result, ce_loss_scaled, dc_loss