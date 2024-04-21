# pred_prob input

# alpha 0.1
python evaluate.py --model ResNet-18 \
--ckpt ./ckpt/useful_ckpt/adversarial-learning/pred_prob/alpha0.1/[adv]20240402_184545_ResNet-18_lr0.0005_100.pth \
--output_path ./outputs/adversarial-learning/pred_prob/alpha0.1 \
--gpu 1 \
-bs 512

# alpha 0.5
python evaluate.py --model ResNet-18 \
--ckpt ./ckpt/useful_ckpt/adversarial-learning/pred_prob/alpha0.5/[adv]20240402_184552_ResNet-18_lr0.0005_100.pth \
--output_path ./outputs/adversarial-learning/pred_prob/alpha0.5 \
--gpu 1 \
-bs 512


# feature input
# alpha 0.1
python evaluate.py --model ResNet-18 \
--ckpt ./ckpt/useful_ckpt/adversarial-learning/feature/alpha0.1/[adv]20240402_184258_ResNet-18_lr0.0005_100.pth \
--output_path ./outputs/adversarial-learning/feature/alpha0.1 \
--gpu 1 \
-bs 512

# alpha 0.5
python evaluate.py --model ResNet-18 \
--ckpt ./ckpt/useful_ckpt/adversarial-learning/feature/alpha0.5/[adv]20240402_184311_ResNet-18_lr0.0005_100.pth \
--output_path ./outputs/adversarial-learning/feature/alpha0.5 \
--gpu 1 \
-bs 512
