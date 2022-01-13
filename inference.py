import os

from datasets import SRDataset
from utils import *

output_dir = 'output'
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
srresnet_checkpoint = './models/best_checkpoint_srresnet.pth.tar'

srresnet = torch.load(srresnet_checkpoint)['model'].to(device)
srresnet.eval()
model = srresnet

test_dataset = SRDataset(split='test', crop_size=0, scaling_factor=3, lr_img_type='imagenet-norm',
                         hr_img_type='[-1, 1]')

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4,
                                          pin_memory=True)

with torch.no_grad():
    for i, (lr_imgs, hr_imgs, image_path) in enumerate(test_loader):
        lr_imgs = lr_imgs.to(device)
        sr_imgs = convert_image(model(lr_imgs).squeeze(0).cpu().detach(), '[-1, 1]', 'pil')

        sr_imgs.save(os.path.join(output_dir, image_path[0].split('/')[-1].replace('.png','_pred.png')))

