from osl.model.registry import ModelRegistry, load_model
from osl.model.segformer import SegformerConfig, SegformerForSemanticSegmentation
from osl.model.convlstm import OSPConfig, OceanSurfacePredictorConvLSTM
from osl.model.vivit import VivitConfig, VivitDecoder
from osl.model.resunet import Unet, UnetConfig
from osl.model.simvp import SimVP, SimVPConfig
from osl.model.afno import AFNONet, AFNOConfig

__all__ = ['load_model']


ModelRegistry.register_model('nvidia/segformer-b0', SegformerForSemanticSegmentation, SegformerConfig(depths=[2, 2, 2, 2], hidden_sizes=[32, 64, 160, 256], decoder_hidden_size=256, num_labels=1), "https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512/resolve/main/model.safetensors")
ModelRegistry.register_model('nvidia/segformer-b1', SegformerForSemanticSegmentation, SegformerConfig(depths=[2, 2, 2, 2], hidden_sizes=[64, 128, 320, 512], decoder_hidden_size=256, num_labels=1), "https://huggingface.co/nvidia/segformer-b1-finetuned-ade-512-512/resolve/main/pytorch_model.bin")
ModelRegistry.register_model('nvidia/segformer-b2', SegformerForSemanticSegmentation, SegformerConfig(depths=[3, 4, 6, 3], hidden_sizes=[64, 128, 320, 512], decoder_hidden_size=768, num_labels=1))
ModelRegistry.register_model('nvidia/segformer-b3', SegformerForSemanticSegmentation, SegformerConfig(depths=[3, 4, 18, 3], hidden_sizes=[64, 128, 320, 512], decoder_hidden_size=768, num_labels=1))
ModelRegistry.register_model('nvidia/segformer-b4', SegformerForSemanticSegmentation, SegformerConfig(depths=[3, 8, 27, 3], hidden_sizes=[64, 128, 320, 512], decoder_hidden_size=768, num_labels=1))
ModelRegistry.register_model('nvidia/segformer-b5', SegformerForSemanticSegmentation, SegformerConfig(depths=[3, 6, 40, 3], hidden_sizes=[64, 128, 320, 512], decoder_hidden_size=768, num_labels=1))

ModelRegistry.register_model('osl/convlstm-s', OceanSurfacePredictorConvLSTM, OSPConfig(hidden_dims=[32, 64, 32], kernel_sizes=[3, 3, 3], num_layers=3))
ModelRegistry.register_model('osl/convlstm-m', OceanSurfacePredictorConvLSTM, OSPConfig(hidden_dims=[64, 128, 128, 64], kernel_sizes=[3, 3, 3, 3], num_layers=4))

ModelRegistry.register_model('osl/vivit-decoder-l', VivitDecoder, VivitConfig(hidden_size=768)) # 89_822_979 params
ModelRegistry.register_model('osl/vivit-decoder-s', VivitDecoder, VivitConfig(hidden_size=144)) # 12_558_675 params

# ModelRegistry.register_model('osl/unet', Unet, UnetConfig(num_channels=3, init_dim=32, dim=32))  #   7_809_475 params
# ModelRegistry.register_model('osl/unet', Unet, UnetConfig(num_channels=3, init_dim=32, dim=64))  #  30_189_891 params
# ModelRegistry.register_model('osl/unet', Unet, UnetConfig(num_channels=3, init_dim=32, dim=128)) # 118_669_315 params

# SimVP: Fully convolutional video prediction (no patch artifacts)
# Note: temporal_module='conv' supports variable seq_length; 'inception'/'tau' require fixed seq_length
ModelRegistry.register_model('osl/simvp-s', SimVP, SimVPConfig(hidden_dim=64, num_layers=4, temporal_module='conv', in_frames=16))
ModelRegistry.register_model('osl/simvp-m', SimVP, SimVPConfig(hidden_dim=128, num_layers=4, temporal_module='conv', in_frames=16))
ModelRegistry.register_model('osl/simvp-tau-s', SimVP, SimVPConfig(hidden_dim=64, num_layers=4, temporal_module='tau', in_frames=16))
ModelRegistry.register_model('osl/simvp-inception-s', SimVP, SimVPConfig(hidden_dim=64, num_layers=4, temporal_module='inception', in_frames=16))

# AFNO: Adaptive Fourier Neural Operator (spectral mixing, no patch artifacts)
# Based on FourCastNet - excellent for periodic boundaries and multi-scale patterns
ModelRegistry.register_model('osl/afno-s', AFNONet, AFNOConfig(hidden_dim=128, num_blocks=4, num_layers=4, in_frames=16))
ModelRegistry.register_model('osl/afno-m', AFNONet, AFNOConfig(hidden_dim=256, num_blocks=8, num_layers=6, in_frames=16))
