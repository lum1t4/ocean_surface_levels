from osl.model.registry import ModelRegistry, load_model
from osl.model.segformer import SegformerConfig, SegformerForSemanticSegmentation
from osl.model.resunet import Unet, UnetConfig
from osl.model.convlstm import AEConvLSTM, AEConvLSTMConfig
from osl.model.forevit import ForeViT, ForeViTConfig
from osl.model.dyffusion import DiffysionTinyInterpolator, DyffusionTinyForecaster, DyffusionTinyConfig
from osl.model.autoencoderkl import AutoencoderKLFlux2, AutoencoderKLFlux2Config
from osl.model.latte import Latte, LatteConfig
from osl.model.dit import DiT, DiTConfig
from osl.model.lavit import LaViT, LaViTConfig
from osl.model.afno import AFNO, AFNOConfig

# ------------------------------------
# Default
# ------------------------------------

ModelRegistry.register_model('unet', Unet, UnetConfig())  # 118_743_571
ModelRegistry.register_model('unet/S', Unet, UnetConfig(dim=32))
# ModelRegistry.register_model('unet-time', Unet, UnetConfig(with_time_emb=True))  # 123_012_115
ModelRegistry.register_model('unet-time', Unet, UnetConfig(dim=32, with_time_emb=True))

ModelRegistry.register_model('nvidia/segformer-b0', SegformerForSemanticSegmentation, SegformerConfig(depths=[2, 2, 2, 2], hidden_sizes=[32, 64, 160, 256], decoder_hidden_size=256, num_labels=1), "https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512/resolve/main/model.safetensors")  # 3_714_401
ModelRegistry.register_model('nvidia/segformer-b1', SegformerForSemanticSegmentation, SegformerConfig(depths=[2, 2, 2, 2], hidden_sizes=[64, 128, 320, 512], decoder_hidden_size=256, num_labels=1), "https://huggingface.co/nvidia/segformer-b1-finetuned-ade-512-512/resolve/main/pytorch_model.bin")  # 13_677_505
ModelRegistry.register_model('nvidia/segformer-b2', SegformerForSemanticSegmentation, SegformerConfig(depths=[3, 4, 6, 3], hidden_sizes=[64, 128, 320, 512], decoder_hidden_size=768, num_labels=1))  # 27_347_393
ModelRegistry.register_model('nvidia/segformer-b3', SegformerForSemanticSegmentation, SegformerConfig(depths=[3, 4, 18, 3], hidden_sizes=[64, 128, 320, 512], decoder_hidden_size=768, num_labels=1))  # 47_223_233
ModelRegistry.register_model('nvidia/segformer-b4', SegformerForSemanticSegmentation, SegformerConfig(depths=[3, 8, 27, 3], hidden_sizes=[64, 128, 320, 512], decoder_hidden_size=768, num_labels=1))  # 63_993_793
ModelRegistry.register_model('nvidia/segformer-b5', SegformerForSemanticSegmentation, SegformerConfig(depths=[3, 6, 40, 3], hidden_sizes=[64, 128, 320, 512], decoder_hidden_size=768, num_labels=1))  # 84_594_113

ModelRegistry.register_model('facebook/DiT-B-2', DiT, DiTConfig(image_size=32, patch_size=2, in_channels=4, hidden_size=768, depth=12, num_heads=12, mlp_ratio=4.0, num_classes=1000, class_dropout_prob=0.1, learn_sigma=True))  # 130_512_416
ModelRegistry.register_model('facebook/DiT-L-2', DiT, DiTConfig(image_size=32, patch_size=2, in_channels=4, hidden_size=1024, depth=24, num_heads=16, mlp_ratio=4.0, num_classes=1000, class_dropout_prob=0.1, learn_sigma=True))  # 458_102_816
ModelRegistry.register_model('facebook/DiT-XL-2', DiT, DiTConfig(image_size=32, patch_size=2, in_channels=4, hidden_size=1152, depth=28, num_heads=16, mlp_ratio=4.0, num_classes=1000, class_dropout_prob=0.1, learn_sigma=True))  # 675_129_632
ModelRegistry.register_model('DiT/T', DiT, DiTConfig(image_size=224, patch_size=8, in_channels=1, out_channels=1, hidden_size=256, depth=6, num_heads=8))  # 7_597_888

ModelRegistry.register_model("Latte/H", Latte, LatteConfig(num_channels=32, out_channels=8, image_size=32, num_layers=28, hidden_size=1152, patch_size=2, num_attention_heads=16, timestep_emb_dim=256, max_temporal_positions=16), "https://huggingface.co/maxin-cn/Latte/resolve/main/ffs.pt")  # 674_123_936
ModelRegistry.register_model("Latte/L", Latte, LatteConfig(num_channels=32, out_channels=8, image_size=32, num_layers=24, hidden_size=1024, patch_size=2, num_attention_heads=16, timestep_emb_dim=256, max_temporal_positions=16), "https://huggingface.co/maxin-cn/Latte/resolve/main/ffs_l_2.pt")  # 457_208_864
ModelRegistry.register_model("Latte/B", Latte, LatteConfig(num_channels=32, out_channels=8, image_size=32, num_layers=12, hidden_size=768, patch_size=2, num_attention_heads=12, timestep_emb_dim=256, max_temporal_positions=16), "https://huggingface.co/maxin-cn/Latte/resolve/main/ffs_b_2.pt")  # 130_428_704
ModelRegistry.register_model("Latte/S", Latte, LatteConfig(num_channels=32, out_channels=8, image_size=32, num_layers=12, hidden_size=384, patch_size=2, num_attention_heads=6, timestep_emb_dim=256, max_temporal_positions=16), "https://huggingface.co/maxin-cn/Latte/resolve/main/ffs_s_2.pt")  # 32_628_128

# ------------------------------------
# Variational Autoendoder Models
# ------------------------------------
ModelRegistry.register_model('black-forest-labs/FLUX.2-klein-4B-VAE', AutoencoderKLFlux2, AutoencoderKLFlux2Config(), "https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/resolve/main/vae/diffusion_pytorch_model.safetensors")  # 84_046_115
ModelRegistry.register_model('VAE/S', AutoencoderKLFlux2, AutoencoderKLFlux2Config(in_channels=1, out_channels=1, block_out_channels=(32, 64, 128, 128)))  # 5_353_281
ModelRegistry.register_model('VAE/M', AutoencoderKLFlux2, AutoencoderKLFlux2Config(in_channels=1, out_channels=1, block_out_channels=(64, 128, 256, 256)))
ModelRegistry.register_model('VAE/L', AutoencoderKLFlux2, AutoencoderKLFlux2Config(in_channels=1, out_channels=1, block_out_channels=(128, 256, 512, 512)))
ModelRegistry.register_model('VAEP/L', AutoencoderKLFlux2, AutoencoderKLFlux2Config(in_channels=1, out_channels=1, block_out_channels=(128, 256, 512, 512)), "https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/resolve/main/vae/diffusion_pytorch_model.safetensors")  # 84_046_115

# ------------------------------------
# Auto Regressive Models
# ------------------------------------
ModelRegistry.register_model('osl/convlstm-s', AEConvLSTM, AEConvLSTMConfig(hidden_dims=[32, 64, 32], kernel_sizes=[3, 3, 3], num_layers=3))  # 778_723
ModelRegistry.register_model('osl/convlstm-m', AEConvLSTM, AEConvLSTMConfig(hidden_dims=[64, 128, 128, 64], kernel_sizes=[3, 3, 3, 3], num_layers=4))  # 5_466_051

ModelRegistry.register_model('ForeViT/S',  ForeViT, ForeViTConfig(hidden_size=256, num_hidden_layers=6, num_attention_heads=8))  # 9_334_819
ModelRegistry.register_model('ForeViT/M',  ForeViT, ForeViTConfig(hidden_size=512, num_hidden_layers=6, num_attention_heads=8))  # 36_754_499
ModelRegistry.register_model('ForeViT/B',  ForeViT, ForeViTConfig(hidden_size=768, num_hidden_layers=6, num_attention_heads=8))  # 82_259_043
ModelRegistry.register_model('ForeViT/L',  ForeViT, ForeViTConfig(hidden_size=768, num_hidden_layers=12, num_attention_heads=16))  # 160_226_403
ModelRegistry.register_model('ForeViT/X',  ForeViT, ForeViTConfig(hidden_size=1024, num_hidden_layers=14, num_attention_heads=16))  # 330_594_435

# Latent Auto-Regressive Models
ModelRegistry.register_model("LaViT/S", LaViT, LaViTConfig())  # 9_569_568
ModelRegistry.register_model("LaViT/M", LaViT, LaViTConfig(hidden_size=192, num_hidden_layers=4, num_attention_heads=6))  # 3_813_344
ModelRegistry.register_model("LaViT/L", LaViT, LaViTConfig(hidden_size=320, num_hidden_layers=8, num_attention_heads=8))  # 19_370_592

# Latent Progressive Models (32ch, 28x28 latent space)
ModelRegistry.register_model('DiT/T-latent', DiT, DiTConfig(image_size=28, patch_size=2, in_channels=32, out_channels=32, hidden_size=256, depth=6, num_heads=8))  # 7_639_552
ModelRegistry.register_model('unet-time-latent', Unet, UnetConfig(dim=32, in_channels=32, out_channels=32, with_time_emb=True))  # 31_328_650

# Latent Dyffusion Models (cat([x0, xT], 1) = 64ch input, 32ch output)
ModelRegistry.register_model('dyffusion/interpolator/unet-latent', Unet, UnetConfig(dim=32, in_channels=64, out_channels=32, with_time_emb=True))  # 31_394_506
ModelRegistry.register_model('dyffusion/interpolator/dit-latent', DiT, DiTConfig(image_size=28, patch_size=2, in_channels=64, out_channels=32, hidden_size=256, depth=6, num_heads=8))  # 7_770_624
ModelRegistry.register_model('dyffusion/forecaster/unet-latent', Unet, UnetConfig(dim=64, in_channels=64, out_channels=32, with_time_emb=True))  # 31_394_506
ModelRegistry.register_model('dyffusion/forecaster/dit-latent', DiT, DiTConfig(image_size=28, patch_size=2, in_channels=64, out_channels=32, hidden_size=256, depth=6, num_heads=8))  # 7_770_624


# ------------------------------------
# Next Frame prediction models
# ------------------------------------

ModelRegistry.register_model('osl/segformer-b0-1x1', SegformerForSemanticSegmentation, SegformerConfig(num_channels=1, num_labels=1, depths=[2, 2, 2, 2], hidden_sizes=[32, 64, 160, 256], decoder_hidden_size=256), "https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512/resolve/main/model.safetensors")  # 3_711_265
ModelRegistry.register_model('osl/segformer-b1-1x1', SegformerForSemanticSegmentation, SegformerConfig(num_channels=1, num_labels=1, depths=[2, 2, 2, 2], hidden_sizes=[64, 128, 320, 512], decoder_hidden_size=256), "https://huggingface.co/nvidia/segformer-b1-finetuned-ade-512-512/resolve/main/pytorch_model.bin")  # 13_671_233
ModelRegistry.register_model('osl/segformer-b2-1x1', SegformerForSemanticSegmentation, SegformerConfig(num_channels=1, num_labels=1, depths=[3, 4, 6, 3], hidden_sizes=[64, 128, 320, 512], decoder_hidden_size=768))  # 27_341_121
ModelRegistry.register_model('osl/segformer-b3-1x1', SegformerForSemanticSegmentation, SegformerConfig(num_channels=1, num_labels=1, depths=[3, 4, 18, 3], hidden_sizes=[64, 128, 320, 512], decoder_hidden_size=768))  # 47_216_961
ModelRegistry.register_model('osl/segformer-b4-1x1', SegformerForSemanticSegmentation, SegformerConfig(num_channels=1, num_labels=1, depths=[3, 8, 27, 3], hidden_sizes=[64, 128, 320, 512], decoder_hidden_size=768))  # 63_987_521
ModelRegistry.register_model('osl/segformer-b5-1x1', SegformerForSemanticSegmentation, SegformerConfig(num_channels=1, num_labels=1, depths=[3, 6, 40, 3], hidden_sizes=[64, 128, 320, 512], decoder_hidden_size=768))  # 84_587_841


ModelRegistry.register_model('osl/segformer-b0-3x3', SegformerForSemanticSegmentation, SegformerConfig(num_channels=3, num_labels=3, depths=[2, 2, 2, 2], hidden_sizes=[32, 64, 160, 256], decoder_hidden_size=256), "https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512/resolve/main/model.safetensors")  # 3_714_915
ModelRegistry.register_model('osl/segformer-b1-3x3', SegformerForSemanticSegmentation, SegformerConfig(num_channels=3, num_labels=3, depths=[2, 2, 2, 2], hidden_sizes=[64, 128, 320, 512], decoder_hidden_size=256), "https://huggingface.co/nvidia/segformer-b1-finetuned-ade-512-512/resolve/main/pytorch_model.bin")  # 13_678_019
ModelRegistry.register_model('osl/segformer-b2-3x3', SegformerForSemanticSegmentation, SegformerConfig(num_channels=3, num_labels=3, depths=[3, 4, 6, 3], hidden_sizes=[64, 128, 320, 512], decoder_hidden_size=768))  # 27_348_931
ModelRegistry.register_model('osl/segformer-b3-3x3', SegformerForSemanticSegmentation, SegformerConfig(num_channels=3, num_labels=3, depths=[3, 4, 18, 3], hidden_sizes=[64, 128, 320, 512], decoder_hidden_size=768))  # 47_224_771
ModelRegistry.register_model('osl/segformer-b4-3x3', SegformerForSemanticSegmentation, SegformerConfig(num_channels=3, num_labels=3, depths=[3, 8, 27, 3], hidden_sizes=[64, 128, 320, 512], decoder_hidden_size=768))  # 63_995_331
ModelRegistry.register_model('osl/segformer-b5-3x3', SegformerForSemanticSegmentation, SegformerConfig(num_channels=3, num_labels=3, depths=[3, 6, 40, 3], hidden_sizes=[64, 128, 320, 512], decoder_hidden_size=768))  # 84_595_651


# ------------------------------------
# AFNO models (FourCastNet-inspired)
# ------------------------------------
ModelRegistry.register_model('AFNO/S', AFNO, AFNOConfig(hidden_size=256, num_layers=6))  # 11_206_976
ModelRegistry.register_model('AFNO/M', AFNO, AFNOConfig(hidden_size=384, num_layers=8))  # 32_257_216
ModelRegistry.register_model('AFNO/B', AFNO, AFNOConfig(hidden_size=512, num_layers=10))  # 70_027_840


# ------------------------------------
# Dyffusion models
# ------------------------------------
ModelRegistry.register_model('dyffusion/interpolator/tiny', DiffysionTinyInterpolator, DyffusionTinyConfig(horizon=8))  # 13_953
ModelRegistry.register_model('dyffusion/interpolator/unet', Unet, UnetConfig(dim=32, in_channels=2, out_channels=1, with_time_emb=True))  # 8_072_285
ModelRegistry.register_model('dyffusion/interpolator/dit', DiT, DiTConfig(image_size=224, patch_size=8, in_channels=2, out_channels=1, hidden_size=256, depth=6, num_heads=8))  # 7_614_272

ModelRegistry.register_model('dyffusion/forecaster/tiny', DyffusionTinyForecaster, DyffusionTinyConfig(horizon=8))  # 51_009
ModelRegistry.register_model('dyffusion/forecaster/unet', Unet, UnetConfig(dim=32, in_channels=2, out_channels=1, with_time_emb=True))  # 8_072_285
ModelRegistry.register_model('dyffusion/forecaster/dit', DiT, DiTConfig(image_size=224, patch_size=8, in_channels=2, out_channels=1, hidden_size=256, depth=6, num_heads=8))  # 7_614_272



__all__ = ['load_model']
