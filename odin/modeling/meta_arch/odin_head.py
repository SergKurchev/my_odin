# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from typing import Dict
import torch

from torch import nn
from torch_scatter import scatter_mean

from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from odin.modeling.transformer_decoder.odin_transformer_decoder import build_transformer_decoder
from odin.modeling.pixel_decoder.msdeformattn import build_pixel_decoder


import ipdb
st = ipdb.set_trace

@SEM_SEG_HEADS_REGISTRY.register()
class ODINHead(nn.Module):

    _version = 2

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "sem_seg_head" in k and not k.startswith(prefix + "predictor"):
                    newk = k.replace(prefix, prefix + "pixel_decoder.")
                    # logger.debug(f"{k} ==> {newk}")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        num_classes: int,
        pixel_decoder: nn.Module,
        loss_weight: float = 1.0,
        ignore_value: int = -1,
        # extra parameters
        transformer_in_feature: str,
        decoder_3d=False,
        cross_view=False,
        hidden_dim=None,
        cfg=None,
        transformer_predictor_in_channels=None,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            num_classes: number of classes to predict
            pixel_decoder: the pixel decoder module
            loss_weight: loss weight
            ignore_value: category id to be ignored during training.
            transformer_predictor: the transformer decoder that makes prediction
            transformer_in_feature: input feature name to the transformer_predictor
        """
        super().__init__()
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]

        self.ignore_value = ignore_value
        self.common_stride = 4
        self.loss_weight = loss_weight
        self.hidden_dim = hidden_dim
        self.cfg = cfg

        self.pixel_decoder = pixel_decoder
        
        self.transformer_in_feature = transformer_in_feature

        self.num_classes = num_classes
        self.decoder_3d = decoder_3d
        self.cross_view = cross_view

        self.predictor = build_transformer_decoder(
                cfg,
                transformer_predictor_in_channels,
                mask_classification=True,
            )
        
    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        # figure out in_channels to transformer predictor
        if cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "transformer_encoder":
            transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        elif cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "pixel_embedding":
            transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        elif cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "multi_scale_pixel_decoder":  # for maskformer2
            transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        else:
            transformer_predictor_in_channels = input_shape[cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE].channels
        return {
            "input_shape": {
                k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
            },
            "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "pixel_decoder": build_pixel_decoder(cfg, input_shape),
            "loss_weight": cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT,
            "transformer_in_feature": cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE,
            "decoder_3d": cfg.MODEL.DECODER_3D,
            "cross_view": cfg.MODEL.CROSS_VIEW_CONTEXTUALIZE,
            "hidden_dim": cfg.MODEL.MASK_FORMER.HIDDEN_DIM,
            "cfg": cfg,
            "transformer_predictor_in_channels": transformer_predictor_in_channels,
        }

    def forward(
        self, features, shape=None, mask=None,
        multiview_data=None, scannet_pc=None, scannet_p2v=None,
        segments=None, decoder_3d=False,
        captions=None, positive_map_od=None, num_classes=None,
        scene_names=None
    ):
        """
        Args:
            - features: dict,
                keys are ['res2', 'res3', 'res4', 'res5']
                    if self.cfg.MODEL.UPSAMPLE_FMAP 'res1' will also be there
                values are [(B*num_views, F_i, H_i, W_i)]
            - shape: the shape of original image, e.g. (B, 3, 256, 320)
            - mask: padding mask, usually None
            - multiview_data: {
                'multi_scale_xyz': [(B, num_views, H_i, W_i, 3)],  # xyz
                'multi_scale_p2v': [(B, num_views * H_i * W_i)]  # point2voxel
            }  # order is from 'res5' to 'res2'
            - scannet_pc: original scannet points
            - segments: tensor (B, num_points), scannet segemnts

        Returns:
            - predictions: {
                'pred_logits': tensor (B, num_queries, num_classes),
                'pred_masks': tensor (B, num_queries, num_views, H_m, W_m),
                    m is the index of the largest feature map that we score
                'pred_scannet_masks':,
                'aux_outputs': [{'pred_logits', 'pred_masks'}]
                    intermediate outputs, shapes same to corr above fields
                    length is equal to the number of layers
            }
        """
        multi_scale_xyz = None  # [features to attend to, diff scales]
        mask_features_xyz = None  # features to score
        mask_features_p2v = None  # point2voxel of mask_features_xyz

        # Select feature scales to attend to
        if decoder_3d:
            multi_scale_xyz = multiview_data['multi_scale_xyz']
            mask_features_xyz = multi_scale_xyz[3]  # xyz of res2
            multi_scale_xyz = multi_scale_xyz[:3]  # xyz of [res5:3]
            mask_features_p2v = multiview_data['multi_scale_p2v'][3] if self.cfg.INPUT.VOXELIZE else None  # p2v of res2


        if decoder_3d and self.cfg.USE_GHOST_POINTS:
            scannet_pc = scatter_mean(
                scannet_pc, scannet_p2v, dim=1
            )
            scannet_p2v = torch.arange(scannet_pc.shape[1], device=scannet_pc.device).unsqueeze(0).repeat(scannet_pc.shape[0], 1)

        # Decoder-part (upsampler) of ResUNet
        mask_features, _, multi_scale_features = self.pixel_decoder.forward_features(
            features, shape, multi_scale_xyz, multiview_data=multiview_data,
            mask_features_xyz=mask_features_xyz, 
            mask_features_p2v=mask_features_p2v, 
            scannet_pc=scannet_pc, scannet_p2v=scannet_p2v,
            decoder_3d=decoder_3d
        )

        # mask_features (B*num_views, F_m, H_m, W_m), m is largest f_map (res2)
        # multi_scale_features: feats of small scales [res5, res4, res3]
        if decoder_3d:
            if self.cfg.USE_GHOST_POINTS:
                mask_features_xyz = scannet_pc
            elif self.cfg.INPUT.VOXELIZE:
                mask_features_xyz = scatter_mean(
                    mask_features_xyz.flatten(1, 3), mask_features_p2v, dim=1
                )

        # Feed to Transformer decoder
        if shape is None:
            shape = [multi_scale_features[0].shape[0], 1]

        # Bayesian Inference: Select inference mode
        bayesian_type = getattr(self.cfg.MODEL, "BAYESIAN_TYPE", "none")
        num_samples = getattr(self.cfg.MODEL, "BAYESIAN_SAMPLES", 1)

        # Prepare forward pass arguments
        forward_kwargs = {
            'x': multi_scale_features,
            'mask_features': mask_features,
            'shape': shape[:2],
            'x_xyz': multi_scale_xyz,
            'mask': mask,
            'mask_features_xyz': mask_features_xyz,
            'multiview_data': multiview_data,
            'segments': segments,
            'scannet_p2v': scannet_p2v,
            'decoder_3d': decoder_3d,
            'captions': captions,
            'positive_map_od': positive_map_od,
            'num_classes': num_classes
        }

        # Determine if we should use Bayesian inference
        bayesian_during_training = getattr(self.cfg.MODEL, "BAYESIAN_INFERENCE_DURING_TRAINING", False)

        use_bayesian_inference = (
            bayesian_type != "none" and
            num_samples > 1 and
            (not self.training or bayesian_during_training)
        )

        # DEBUG: Print inference mode
        if not hasattr(self, '_inference_mode_printed'):
            print(f"\n=== BAYESIAN INFERENCE CONFIG ===")
            print(f"self.training: {self.training}")
            print(f"bayesian_type: {bayesian_type}")
            print(f"num_samples: {num_samples}")
            print(f"bayesian_during_training: {bayesian_during_training}")
            print(f"use_bayesian_inference: {use_bayesian_inference}")
            print(f"=== END CONFIG ===\n")
            self._inference_mode_printed = True

        # Select inference method
        if use_bayesian_inference:
            if bayesian_type == "mc_dropout":
                predictions = self._mc_dropout_inference(num_samples, forward_kwargs)
            elif bayesian_type == "swag":
                predictions = self._swag_inference(num_samples, forward_kwargs)
            else:
                # Unknown type, fall back to deterministic
                predictions = self._deterministic_inference(forward_kwargs)
        else:
            # Training or deterministic inference
            predictions = self._deterministic_inference(forward_kwargs)

        return predictions

    def _deterministic_inference(self, forward_kwargs):
        """
        Deterministic inference: single forward pass without sampling.

        Args:
            forward_kwargs: Dictionary of arguments for predictor forward pass

        Returns:
            Model predictions
        """
        return self.predictor(**forward_kwargs)

    def _mc_dropout_inference(self, num_samples, forward_kwargs):
        """
        MC Dropout inference: multiple forward passes with dropout enabled.

        This method fixes the original implementation by actually enabling dropout
        during inference, rather than just repeating deterministic passes.

        Args:
            num_samples: Number of MC samples to draw
            forward_kwargs: Dictionary of arguments for predictor forward pass

        Returns:
            Averaged predictions over MC samples
        """
        # Enable dropout layers while keeping BatchNorm in eval mode
        def enable_dropout(m):
            if isinstance(m, nn.Dropout):
                m.train()

        self.predictor.apply(enable_dropout)

        all_outputs = []
        for _ in range(num_samples):
            out = self.predictor(**forward_kwargs)
            all_outputs.append(out)

        # Restore eval mode
        self.predictor.eval()

        # Average predictions
        predictions = self._average_predictions(all_outputs)

        return predictions

    def _swag_inference(self, num_samples, forward_kwargs):
        """
        SWAG inference: sample weights from SWAG posterior and run forward passes.

        Requires SWAG wrapper to be attached to the model during training.

        Args:
            num_samples: Number of weight samples to draw
            forward_kwargs: Dictionary of arguments for predictor forward pass

        Returns:
            Averaged predictions over SWAG samples
        """
        # Check if SWAG wrapper exists
        if not hasattr(self, 'swag_model') or self.swag_model is None:
            print("Warning: SWAG model not found, falling back to deterministic inference")
            return self._deterministic_inference(forward_kwargs)

        scale = getattr(self.cfg.MODEL.SWAG, "SCALE", 1.0)

        all_outputs = []
        for _ in range(num_samples):
            # Sample weights from SWAG posterior
            self.swag_model.sample(scale=scale, cov=True)

            # Forward pass with sampled weights
            out = self.predictor(**forward_kwargs)
            all_outputs.append(out)

        # Restore SWA mean weights
        self.swag_model.set_swa()

        # Average predictions
        predictions = self._average_predictions(all_outputs)

        return predictions

    def _average_predictions(self, all_outputs):
        """
        Average predictions from multiple forward passes.

        Averages softmax probabilities and converts back to log-probabilities
        for compatibility with the rest of the codebase.

        Also computes uncertainty metrics from the variance across samples.

        Bayesian Uncertainty Formulas (from "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?" Kendall & Gal, 2017):

        1. Predictive Entropy (Total Uncertainty):
           H[y|x,D] = -∑_c p̄(y=c|x,D) log p̄(y=c|x,D)
           where p̄(y=c|x,D) = (1/T)∑_t p(y=c|x,θ_t) is the averaged prediction

        2. Expected Entropy (Aleatoric/Data Uncertainty):
           E_θ[H[y|x,θ]] = (1/T)∑_t [-∑_c p(y=c|x,θ_t) log p(y=c|x,θ_t)]
           Average of individual prediction entropies

        3. Mutual Information (Epistemic/Model Uncertainty):
           I[y,θ|x,D] = H[y|x,D] - E_θ[H[y|x,θ]]
           Difference between total and aleatoric uncertainty

        Args:
            all_outputs: List of prediction dictionaries from multiple forward passes

        Returns:
            Averaged predictions with uncertainty metrics
        """
        # Use first output as template
        predictions = all_outputs[0]

        # Stack and average logits
        logits_stack = torch.stack([o['pred_logits'] for o in all_outputs])  # [S, B, Q, C]

        # DEBUG: Print shapes and statistics
        print(f"\n=== UNCERTAINTY DEBUG ===")
        print(f"Number of samples: {len(all_outputs)}")
        print(f"Logits stack shape: {logits_stack.shape}")  # [S, B, Q, C]
        print(f"Logits stack stats: min={logits_stack.min():.4f}, max={logits_stack.max():.4f}, mean={logits_stack.mean():.4f}")

        # Average probabilities (not logits)
        probs_stack = torch.softmax(logits_stack, dim=-1)  # [S, B, Q, C]
        print(f"Probs stack shape: {probs_stack.shape}")
        print(f"Probs stack stats: min={probs_stack.min():.4f}, max={probs_stack.max():.4f}, mean={probs_stack.mean():.4f}")

        # Check variance across samples
        probs_variance = probs_stack.var(dim=0)  # [B, Q, C]
        print(f"Probs variance shape: {probs_variance.shape}")
        print(f"Probs variance stats: min={probs_variance.min():.6f}, max={probs_variance.max():.6f}, mean={probs_variance.mean():.6f}")

        avg_probs = probs_stack.mean(dim=0)  # [B, Q, C]
        print(f"Avg probs shape: {avg_probs.shape}")
        print(f"Avg probs stats: min={avg_probs.min():.4f}, max={avg_probs.max():.4f}, mean={avg_probs.mean():.4f}")

        # Convert back to log-probabilities
        predictions['pred_logits'] = torch.log(avg_probs + 1e-8)

        # Compute uncertainty metrics from variance across samples
        # 1. Predictive entropy (uncertainty in averaged prediction)
        # H[y|x,D] = -∑_c p̄_c log p̄_c
        predictive_entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-8), dim=-1)  # [B, Q]
        print(f"\nPredictive entropy shape: {predictive_entropy.shape}")
        print(f"Predictive entropy stats: min={predictive_entropy.min():.6f}, max={predictive_entropy.max():.6f}, mean={predictive_entropy.mean():.6f}")

        # 2. Expected entropy (average uncertainty of individual predictions)
        # E_θ[H[y|x,θ]] = (1/T)∑_t H[y|x,θ_t]
        sample_entropies = -torch.sum(probs_stack * torch.log(probs_stack + 1e-8), dim=-1)  # [S, B, Q]
        print(f"Sample entropies shape: {sample_entropies.shape}")
        print(f"Sample entropies stats: min={sample_entropies.min():.6f}, max={sample_entropies.max():.6f}, mean={sample_entropies.mean():.6f}")

        expected_entropy = sample_entropies.mean(dim=0)  # [B, Q]
        print(f"Expected entropy shape: {expected_entropy.shape}")
        print(f"Expected entropy stats: min={expected_entropy.min():.6f}, max={expected_entropy.max():.6f}, mean={expected_entropy.mean():.6f}")

        # 3. Mutual information (epistemic uncertainty)
        # I[y,θ|x,D] = H[y|x,D] - E_θ[H[y|x,θ]]
        mutual_info = predictive_entropy - expected_entropy  # [B, Q]
        print(f"Mutual information shape: {mutual_info.shape}")
        print(f"Mutual information stats: min={mutual_info.min():.6f}, max={mutual_info.max():.6f}, mean={mutual_info.mean():.6f}")
        print(f"=== END DEBUG ===\n")

        # Store uncertainty metrics in predictions
        predictions['uncertainty'] = {
            'predictive_entropy': predictive_entropy,  # Total uncertainty
            'expected_entropy': expected_entropy,      # Aleatoric (data) uncertainty
            'mutual_information': mutual_info          # Epistemic (model) uncertainty
        }

        # Average auxiliary outputs if present
        if "aux_outputs" in predictions:
            for i in range(len(predictions["aux_outputs"])):
                aux_logits_stack = torch.stack([o["aux_outputs"][i]["pred_logits"] for o in all_outputs])
                avg_aux_probs = torch.softmax(aux_logits_stack, dim=-1).mean(dim=0)
                predictions["aux_outputs"][i]["pred_logits"] = torch.log(avg_aux_probs + 1e-8)

        return predictions
