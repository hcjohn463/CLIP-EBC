import os
import torch
from torchvision import transforms
from torchvision.transforms.functional import normalize, to_pil_image
import matplotlib.pyplot as plt
from PIL import Image
import sys
import json

# 加入專案路徑
sys.path.append(os.path.abspath("."))

from models import get_model
from utils import resize_density_map, sliding_window_predict

# 參數設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "clip_vit_b_16"
input_size = 224
reduction = 8
truncation = 2
granularity = "fine"
anchor_points_type = "average"
prompt_type = "word"
num_vpt = 32
vpt_drop = 0.0
deep_vpt = True
window_size = 224
stride = 224
alpha = 0.8

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

# 你的圖片路徑
img_path = "1367.jpg"  # 改成你的圖片檔名
# 權重路徑
ckpt_path = "best_mae.pth"  # 改成你的權重檔案

# 讀取 config
with open(f"configs/reduction_{reduction}.json", "r") as f:
    config = json.load(f)[str(truncation)]["qnrf"]  # 使用 qnrf 設定
bins = config["bins"][granularity]
anchor_points = config["anchor_points"][granularity][anchor_points_type]
bins = [(float(b[0]), float(b[1])) for b in bins]
anchor_points = [float(p) for p in anchor_points]

# 載入模型
model = get_model(
    backbone=model_name,
    input_size=input_size,
    reduction=reduction,
    bins=bins,
    anchor_points=anchor_points,
    prompt_type=prompt_type,
    num_vpt=num_vpt,
    vpt_drop=vpt_drop,
    deep_vpt=deep_vpt
)
ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt)
model = model.to(device)
model.eval()

# 圖片前處理
transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
image = Image.open(img_path).convert("RGB")
image = transform(image).unsqueeze(0).to(device)
image_height, image_width = image.shape[-2:]
image_name = os.path.basename(img_path)

# 推論
with torch.no_grad():
    pred_density = sliding_window_predict(model, image, window_size, stride)
    pred_count = pred_density.sum().item()
    resized_pred_density = resize_density_map(pred_density, (image_height, image_width)).cpu()

# 還原圖片顯示
image_disp = image.clone()
image_disp = normalize(image_disp, mean=(0., 0., 0.), std=(1. / std[0], 1. / std[1], 1. / std[2]))
image_disp = normalize(image_disp, mean=(-mean[0], -mean[1], -mean[2]), std=(1., 1., 1.))
image_disp = to_pil_image(image_disp.squeeze(0))

resized_pred_density = resized_pred_density.squeeze().numpy()

fig, axes = plt.subplots(1, 2, dpi=200, tight_layout=True, frameon=False)
axes[0].imshow(image_disp)
axes[0].axis("off")
axes[0].set_title(f"{image_name}")

axes[1].imshow(image_disp)
axes[1].imshow(resized_pred_density, cmap="jet", alpha=alpha)
axes[1].axis("off")
axes[1].set_title(f"Pred count: {pred_count:.2f}")

plt.show()