import argparse
import os
import random
import torch
from pipal_data import NTIRE2022
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from utils import Normalize, ToTensor, crop_image


def parse_args():
    parser = argparse.ArgumentParser(description='Inference script of RealBasicVSR')
    parser.add_argument('--model_path', help='checkpoint file', required=True)
    parser.add_argument('--input_dir', help='directory of the input video', required=True)
    parser.add_argument(
        '--output_dir',
        help='directory of the output results',
        default='output/ensemble_attentionIQA2_finetune_e2/AnimeSR')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # configuration
    batch_size = 10
    num_workers = 8
    average_iters = 20
    crop_size = 224
    os.makedirs(args.output_dir, exist_ok=True)

    model = torch.load(args.model_path)

    # map to cuda, if available
    cuda_flag = False
    if torch.cuda.is_available():
        model = model.cuda()
        cuda_flag = True
    model.eval()
    total_avg_score = []
    subfolder_namelist = []
    for subfolder_name in sorted(os.listdir(args.input_dir)):
        avg_score = 0.0
        subfolder_root = os.path.join(args.input_dir, subfolder_name)

        if os.path.isdir(subfolder_root) and subfolder_name != 'assemble-folder':
            # data load
            val_dataset = NTIRE2022(
                ref_path=subfolder_root,
                dis_path=subfolder_root,
                transform=transforms.Compose([Normalize(0.5, 0.5), ToTensor()]),
            )
            val_loader = DataLoader(
                dataset=val_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True, shuffle=False)

            name_list, pred_list = [], []
            with open(os.path.join(args.output_dir, f'{subfolder_name}.txt'), 'w') as f:

                for data in tqdm(val_loader):
                    pred = 0

                    for i in range(average_iters):
                        if cuda_flag:
                            x_d = data['d_img_org'].cuda()
                        b, c, h, w = x_d.shape
                        top = random.randint(0, h - crop_size)
                        left = random.randint(0, w - crop_size)
                        img = crop_image(top, left, crop_size, img=x_d)
                        with torch.no_grad():
                            pred += model(img)
                    pred /= average_iters
                    d_name = data['d_name']
                    pred = pred.cpu().numpy()
                    name_list.extend(d_name)
                    pred_list.extend(pred)

                for i in range(len(name_list)):
                    f.write(f'{name_list[i]}, {float(pred_list[i][0]): .6f}\n')
                    avg_score += float(pred_list[i][0])

                avg_score /= len(name_list)
                f.write(f'The average score of {subfolder_name} is {avg_score:.6f}')
                f.close()
                subfolder_namelist.append(subfolder_name)
                total_avg_score.append(avg_score)

    with open(os.path.join(args.output_dir, 'average.txt'), 'w') as f:
        for idx, averge_score in enumerate(total_avg_score):
            string = f'Folder {subfolder_namelist[idx]}, Average Score: {averge_score:.6f}\n'
            f.write(string)
            print(f'Folder {subfolder_namelist[idx]}, Average Score: {averge_score:.6f}')

        print(f'Average Score of {len(subfolder_namelist)} Folders: {sum(total_avg_score) / len(total_avg_score):.6f}')
        string = f'Average Score of {len(subfolder_namelist)} Folders: {sum(total_avg_score) / len(total_avg_score):.6f}'  # noqa E501
        f.write(string)
        f.close()


if __name__ == '__main__':
    main()
