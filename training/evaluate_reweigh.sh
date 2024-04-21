# reweigh gender
#python evaluate.py --model ResNet-18 \
#--ckpt ./ckpt/useful_ckpt/reweigh/gender-group/20240331_213432_ResNet-18_lr0.0005_100.pth \
#--output_path ./outputs/reweigh/gender-group \
#-bs 256

# reweigh race
#python evaluate.py --model ResNet-18 \
#--ckpt ./ckpt/useful_ckpt/reweigh/race-group/20240331_142856_ResNet-18_lr0.0005_100.pth \
#--output_path ./outputs/reweigh/race-group \
#-bs 256

# reweigh intersec
#python evaluate.py --model  ResNet-18 \
#--ckpt ./ckpt/useful_ckpt/reweigh/intersec-group/20240331_142235_ResNet-18_lr0.0005_100.pth \
#--output_path ./outputs/reweigh/intersec-group \
#-bs 512
#
#python evaluate.py --model  ResNet-50 \
#--ckpt ./ckpt/useful_ckpt/reweigh/intersec-group/20240331_213616_ResNet-50_lr0.0005_100.pth \
#--output_path ./outputs/reweigh/intersec-group \
#-bs 256
#
#python evaluate.py --model  EfficientNet-B3 \
#--ckpt ./ckpt/useful_ckpt/reweigh/intersec-group/20240401_180840_EfficientNet-B3_lr0.0005_40.pth \
#--output_path ./outputs/reweigh/intersec-group \
#-bs 256
#
#python evaluate.py --model XceptionNet-hongguliu-ImageNet-pretrained \
#--ckpt ./ckpt/useful_ckpt/reweigh/intersec-group/20240331_214015_XceptionNet-hongguliu-ImageNet-pretrained_lr0.0005_100.pth \
#--output_path ./outputs/reweigh/intersec-group \
#-bs 256