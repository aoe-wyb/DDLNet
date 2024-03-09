import torch
from data import valid_dataloader, nighttime_valid_dataloader
from utils import Adder
import os
from skimage.metrics import peak_signal_noise_ratio
import torch.nn.functional as f


def _valid(model, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.valid_data == 'NH-Haze':
        use_transform=True
    else:
        use_transform=False
        
    data_set = valid_dataloader(path=args.data_dir, batch_size=1, num_workers=0, valid_data=args.valid_data)  # , use_transform=True
    #data_set = nighttime_valid_dataloader(path=args.data_dir, batch_size=1, num_workers=0, valid_data=args.valid_data)


    model.eval()
    psnr_adder = Adder()

    with torch.no_grad():
        print('Start Dehazing Evaluation')
        factor = 8
        for idx, data in enumerate(data_set):
            input_img, label_img = data
            input_img = input_img.to(device)

            h, w = input_img.shape[2], input_img.shape[3]
            H, W = ((h + factor) // factor) * factor, ((w + factor) // factor * factor)
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            input_img = f.pad(input_img, (0, padw, 0, padh), 'reflect')

            # if not os.path.exists(os.path.join(args.result_dir, '%d' % (ep))):
            #     os.mkdir(os.path.join(args.result_dir, '%d' % (ep)))

            pred = model(input_img)[2]
            pred = pred[:, :, :h, :w]

            pred_clip = torch.clamp(pred, 0, 1)
            p_numpy = pred_clip.squeeze(0).cpu().numpy()
            label_numpy = label_img.squeeze(0).cpu().numpy()
            # print(p_numpy.size(), label_numpy.size())
            psnr = peak_signal_noise_ratio(p_numpy, label_numpy, data_range=1)

            psnr_adder(psnr)
            # print('\r%03d'%idx, end=' ')

    # print('\n')
    model.train()
    return psnr_adder.average()
