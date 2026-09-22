import torch, os, sys
files = [
    'checkpoints/unet_best.pth',
    'checkpoints/unet_last.pth',
]
for f in files:
    if not os.path.exists(f):
        print(f'{f}: NOT FOUND')
        continue
    c = torch.load(f, map_location='cpu', weights_only=False)
    if isinstance(c, dict) and 'epoch' in c:
        ep   = c.get('epoch', '?')
        viou = c.get('val_iou', '?')
        tiou = c.get('train_iou', '?')
        print(f'{f}: epoch={ep}, val_iou={viou}, train_iou={tiou}')
    else:
        print(f'{f}: raw state_dict (no metadata)')
