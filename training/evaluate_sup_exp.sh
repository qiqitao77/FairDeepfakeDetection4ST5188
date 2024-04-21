# reweigh-prior gender
#python evaluate.py --model  ResNet-18 \
#--ckpt ./ckpt/20240412_074538_ResNet-18_lr0.0005_100.pth \
#--output_path ./outputs/reweigh/gender-group \
#-bs 512

# reweigh-prior race
#python evaluate.py --model  ResNet-18 \
#--ckpt ./ckpt/20240411_233203_ResNet-18_lr0.0005_100.pth \
#--output_path ./outputs/reweigh/race-group \
#-bs 512

# reweigh-prior intersec
#python evaluate.py --model  ResNet-18 \
#--ckpt ./ckpt/20240411_161048_ResNet-18_lr0.0005_100.pth \
#--output_path ./outputs/reweigh/intersec-group \
#-bs 512

# reweigh-subgroup gender
#python evaluate.py --model  ResNet-18 \
#--ckpt ./ckpt/20240412_074412_ResNet-18_lr0.0005_100.pth \
#--output_path ./outputs/reweigh/gender-group \
#-bs 512

# reweigh-subgroup race
#python evaluate.py --model  ResNet-18 \
#--ckpt ./ckpt/20240412_074157_ResNet-18_lr0.0005_100.pth \
#--output_path ./outputs/reweigh/race-group \
#-bs 512

# reweigh-subgroup intersec
#python evaluate.py --model  ResNet-18 \
#--ckpt ./ckpt/20240411_232729_ResNet-18_lr0.0005_100.pth \
#--output_path ./outputs/reweigh/gender-group \
#-bs 512

# resample-all-augmentations-balanced-batch
#python evaluate.py --model  ResNet-18 \
#--ckpt ./ckpt/20240411_185311_ResNet-18_lr0.0005_100.pth \
#--output_path ./outputs/resample/all-augmentations \
#-bs 512
# resample-all-augmentations-balanced-batch
#python evaluate.py --model  ResNet-18 \
#--ckpt ./ckpt/20240411_144706_ResNet-18_lr0.0005_100.pth \
#--output_path ./outputs/resample/sub-augmentations \
#-bs 512

# adv-pred-prob-EfficientNet-B3
#

# adv-feature-EfficientNet-B3
#python evaluate.py --model EfficientNet-B3 \
#--ckpt ./ckpt/[adv]20240409_192405_EfficientNet-B3_lr0.0005_60.pth \
#--output_path ./outputs/adversarial-learning/feature/alpha0.1 \
#--gpu 1 \
#-bs 256python evaluate.py --model EfficientNet-B3 \
#--ckpt ./ckpt/[adv]20240409_173413_EfficientNet-B3_lr0.0005_80.pth \
#--output_path ./outputs/adversarial-learning/pred_prob/alpha0.1 \
#--gpu 1 \
#-bs 256

# adv-pred-prob-XceptionNet
python evaluate.py --model XceptionNet-hongguliu-ImageNet-pretrained \
--ckpt ./ckpt/[adv]20240409_193142_XceptionNet-hongguliu-ImageNet-pretrained_lr0.0005_100.pth \
--output_path ./outputs/adversarial-learning/pred_prob/alpha0.1 \
--gpu 1 \
-bs 256

# adv-feature-XceptionNet
python evaluate.py --model XceptionNet-hongguliu-ImageNet-pretrained \
--ckpt ./ckpt/[adv]20240409_192417_XceptionNet-hongguliu-ImageNet-pretrained_lr0.0005_100.pth \
--output_path ./outputs/adversarial-learning/feature/alpha0.1 \
--gpu 1 \
-bs 128
