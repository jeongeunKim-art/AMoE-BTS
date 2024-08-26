
import argparse
from utils.data_utils import get_loader
from medical.trainer import Trainer, Validator
from monai.inferers import SlidingWindowInferer
import nibabel as nib
import torch
import os
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.parallel
from tqdm import tqdm
from functools import partial
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
import numpy as np
from torchmetrics import Metric
from medpy import metric
from medpy.metric.binary import hd95 
from monai.metrics import DiceMetric
from monai.metrics.utils import do_metric_reduction
from monai.utils.enums import MetricReduction
from medical.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from monai.losses.dice import DiceLoss
from monai.networks.nets import SwinUNETR

parser = argparse.ArgumentParser(description='Swin UNETR segmentation pipeline for BRATS Challenge')
parser.add_argument('--model_name', default="swinunetr", help='the model will be trained')
parser.add_argument('--checkpoint', default=None, help='start training from saved checkpoint')
parser.add_argument('--logdir', default='test', type=str, help='directory to save the tensorboard logs')
parser.add_argument('--fold', default=0, type=int, help='data fold')
parser.add_argument('--pretrained_model_name', default='swinunetrmodel_final.pt', type=str, help='pretrained model name')
parser.add_argument('--load_pretrain', action="store_true", help='pretrained model name')
parser.add_argument('--data_dir', default='MICCAI_BraTS_2019_Data_Training', type=str, help='dataset directory')
parser.add_argument('--json_list', default='./train19_data.json', type=str, help='dataset json file')
parser.add_argument('--testjson_list', default='./test_data19.json', type=str, help='dataset json file')
parser.add_argument('--max_epochs', default=300, type=int, help='max number of training epochs')
parser.add_argument('--batch_size', default=1, type=int, help='number of batch size')
parser.add_argument('--sw_batch_size', default=4, type=int, help='number of sliding window batch size')
parser.add_argument('--optim_lr', default=1e-4, type=float, help='optimization learning rate')
parser.add_argument('--optim_name', default='adamw', type=str, help='optimization algorithm')
parser.add_argument('--reg_weight', default=1e-5, type=float, help='regularization weight')
parser.add_argument('--momentum', default=0.99, type=float, help='momentum')
parser.add_argument('--val_every', default=10, type=int, help='validation frequency')
parser.add_argument('--distributed', action='store_true', help='start distributed training')
parser.add_argument('--world_size', default=1, type=int, help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int, help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:23456', type=str, help='distributed url')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--norm_name', default='instance', type=str, help='normalization name')
parser.add_argument('--workers', default=8, type=int, help='number of workers')
parser.add_argument('--feature_size', default=24, type=int, help='feature size')
parser.add_argument('--in_channels', default=4, type=int, help='number of input channels')
parser.add_argument('--out_channels', default=3, type=int, help='number of output channels')
parser.add_argument('--cache_dataset', action='store_true', help='use monai Dataset class')
parser.add_argument('--a_min', default=-175.0, type=float, help='a_min in ScaleIntensityRanged')
parser.add_argument('--a_max', default=250.0, type=float, help='a_max in ScaleIntensityRanged')
parser.add_argument('--b_min', default=0.0, type=float, help='b_min in ScaleIntensityRanged')
parser.add_argument('--b_max', default=1.0, type=float, help='b_max in ScaleIntensityRanged')
parser.add_argument('--space_x', default=1.0, type=float, help='spacing in x direction')
parser.add_argument('--space_y', default=1.0, type=float, help='spacing in y direction')
parser.add_argument('--space_z', default=1.0, type=float, help='spacing in z direction')
parser.add_argument('--roi_x', default=128, type=int, help='roi size in x direction')
parser.add_argument('--roi_y', default=128, type=int, help='roi size in y direction')
parser.add_argument('--roi_z', default=128, type=int, help='roi size in z direction')
parser.add_argument('--dropout_rate', default=0.0, type=float, help='dropout rate')
parser.add_argument('--dropout_path_rate', default=0.0, type=float, help='drop path rate')
parser.add_argument('--RandFlipd_prob', default=0.2, type=float, help='RandFlipd aug probability')
parser.add_argument('--RandRotate90d_prob', default=0.2, type=float, help='RandRotate90d aug probability')
parser.add_argument('--RandScaleIntensityd_prob', default=0.1, type=float, help='RandScaleIntensityd aug probability')
parser.add_argument('--RandShiftIntensityd_prob', default=0.1, type=float, help='RandShiftIntensityd aug probability')
parser.add_argument('--infer_overlap', default=0.25, type=float, help='sliding window inference overlap')
parser.add_argument('--lrschedule', default='warmup_cosine', type=str, help='type of learning rate scheduler')
parser.add_argument('--warmup_epochs', default=50, type=int, help='number of warmup epochs')
parser.add_argument('--resume_ckpt', action='store_true', help='resume training from pretrained checkpoint')
parser.add_argument(
    "--pretrained_dir",
    default="runs/log_train_nestedformer/",
    type=str,
    help="pretrained checkpoint directory",
)


def hd(pred, gt):
    if pred.sum() > 0 and gt.sum() > 0:
        hd95 = metric.binary.hd95(pred, gt)
        return hd95
    else:
        return 0

def post_pred_func(pred):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    return pred

def main():
    args = parser.parse_args()
    args.logdir = './runs/' + args.logdir
    if args.distributed:
        args.ngpus_per_node = torch.cuda.device_count()
        print('Found total gpus', args.ngpus_per_node)
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker,
                 nprocs=args.ngpus_per_node,
                 args=(args,))
    else:
        main_worker(gpu=0, args=args)

def main_worker(gpu, args):

    if args.distributed:
        torch.multiprocessing.set_start_method('fork', force=True)
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, suppress=True)
    args.gpu = gpu
    if args.distributed:
        args.rank = args.rank * args.ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend,
                                init_method=args.dist_url,
                                world_size=args.world_size,
                                rank=args.rank)
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    args.test_mode = True

    test_loader = get_loader(args)
    pretrained_dir = args.pretrained_dir
    model_name = args.pretrained_model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_pth = os.path.join(pretrained_dir, model_name)
    inf_size = [args.roi_x, args.roi_y, args.roi_z]

    model = SwinUNETR(
        img_size=(128, 128, 128),
        in_channels=4,
        out_channels=3,
        feature_size=24,
        use_checkpoint=True,)

    window_infer = SlidingWindowInferer(roi_size=inf_size,
                                        sw_batch_size=args.sw_batch_size,
                                        overlap=args.infer_overlap)
    
    model_dict = torch.load(pretrained_pth)["state_dict"]
    model.load_state_dict(model_dict)
    model.eval()
    model.to(device)


    class Validator:
        def __init__(self, args, model, test_loader, class_list, metric_functions, sliding_window_infer=None, post_label=None, post_pred=None) -> None:
            self.val_loader = test_loader
            self.sliding_window_infer = sliding_window_infer
            self.model = model
            self.args = args
            self.post_label = post_label
            self.post_pred = post_pred
            self.metric_functions = metric_functions
            self.class_list = class_list

        def metric_dice_avg(self, metric):
            metric_sum = 0.0
            c_nums = 0
            for m, v in metric.items():
                if "dice" in m.lower():
                    metric_sum += v
                    c_nums += 1
            return metric_sum / c_nums

        def is_best_metric(self, cur_metric, best_metric):
            best_metric_sum = self.metric_dice_avg(best_metric)
            metric_sum = self.metric_dice_avg(cur_metric)
            if best_metric_sum < metric_sum:
                return True
            return False

        def run(self):
            self.model.eval()
            args = self.args

            assert len(self.metric_functions[0]) == 2

            accs = [None for _ in range(len(self.metric_functions))]
            not_nans = [None for _ in range(len(self.metric_functions))]
            class_metric = []
            hausdorff_results = {clas: [] for clas in self.class_list}  # add dictionary
            
            for m in self.metric_functions:
                for clas in self.class_list:
                    class_metric.append(f"{clas}_{m[0]}")

            first_batch_saved = False
        

            for idx, batch in tqdm(enumerate(self.val_loader), total=len(self.val_loader)):
                batch = {x: batch[x].to(torch.device('cuda', args.rank)) for x in batch if x not in ['fold', 'image_meta_dict', 'label_meta_dict', 'foreground_start_coord', 'foreground_end_coord', 'image_transforms', 'label_transforms']}
                label = batch["label"]

                with torch.no_grad():
                    if self.sliding_window_infer is not None:
                        logits = self.sliding_window_infer(batch["image"], self.model)
                    else:
                        logits = self.model(batch["image"])

                    if self.post_label is not None:
                        label = self.post_label(label)

                    if self.post_pred is not None:
                        logits = self.post_pred(logits)

                    if not first_batch_saved:
                        self.save_images(batch["image"], logits, label)
                        first_batch_saved = True

                    for i in range(len(self.metric_functions)):
                        acc = self.metric_functions[i][1](y_pred=logits, y=label)
                        acc, not_nan = do_metric_reduction(acc, MetricReduction.MEAN_BATCH)
                        acc = acc.cuda(args.rank)
                        not_nan = not_nan.cuda(args.rank)
                        if accs[i] is None:
                            accs[i] = acc
                            not_nans[i] = not_nan
                        else:
                            accs[i] += acc
                            not_nans[i] += not_nan
                    
                    # HD95 계산 및 저장
                    for clas in self.class_list:
                        logits_np = logits.cpu().numpy()[:, self.class_list.index(clas)]  
                        label_np = label.cpu().numpy()[:, self.class_list.index(clas)] 

                        # 확률을 이진 마스크로 변환 (임계값 설정)
                        threshold = 0.5 
                        logits_binary = (logits_np > threshold).astype(int)
                        label_binary = (label_np > 0).astype(int)  # 레이블이 0이 아닌 경우를 1로 설정

                        # HD95 점수 계산
                        hd95_score = hd(logits_binary, label_binary)  
                        hausdorff_results[clas].append(hd95_score)
                        
            hd95_avg_results = {}
            for clas in self.class_list:
                valid_scores = [score for score in hausdorff_results[clas] if score > 0]  # 0이 아닌 HD95 점수만 필터링
                if valid_scores:
                    hd95_avg_results[clas] = sum(valid_scores) / len(valid_scores)
                else:
                    hd95_avg_results[clas] = 0  # 모든 점수가 0인 경우 0으로 설정

            if args.distributed:
                accs = torch.stack(accs).cuda(args.rank).flatten()
                not_nans = torch.stack(not_nans).cuda(args.rank).flatten()
                torch.distributed.barrier()
                gather_list_accs = [torch.zeros_like(accs) for _ in range(torch.distributed.get_world_size())]
                torch.distributed.all_gather(gather_list_accs, accs)
                gather_list_not_nans = [torch.zeros_like(not_nans) for _ in range(torch.distributed.get_world_size())]
                torch.distributed.all_gather(gather_list_not_nans, not_nans)

                accs_sum = torch.stack(gather_list_accs).sum(dim=0).flatten()
                not_nans_sum = torch.stack(gather_list_not_nans).sum(dim=0).flatten()

                not_nans_sum[not_nans_sum == 0] = 1
                accs_sum = accs_sum / not_nans_sum
                all_metric_list = {k: v for k, v in zip(class_metric, accs_sum.tolist())}

            else:
                accs = torch.stack(accs, dim=0).flatten()
                not_nans = torch.stack(not_nans, dim=0).flatten()
                not_nans[not_nans == 0] = 1
                accs = accs / not_nans
                all_metric_list = {k: v for k, v in zip(class_metric, accs.tolist())}

            return all_metric_list, hd95_avg_results

        def save_images(self, images, logits, labels):
            for i, image in enumerate(images):
                image_3d = image.cpu().numpy().astype(np.float32)  # 데이터 타입 변환
                image_nii = nib.Nifti1Image(image_3d, np.eye(4))
                nib.save(image_nii, f"predictions/swinunetr_image_{i}.nii.gz")

            for i, logit in enumerate(logits):
                logit_3d = logit.cpu().numpy().astype(np.float32)  # 데이터 타입 변환
                logit_nii = nib.Nifti1Image(logit_3d, np.eye(4))
                nib.save(logit_nii, f"predictions/swinunetr_logit_{i}.nii.gz")

            for i, label in enumerate(labels):
                label_3d = label.cpu().numpy().astype(np.float32)  # 데이터 타입 변환
                label_nii = nib.Nifti1Image(label_3d, np.eye(4))
                nib.save(label_nii, f"predictions/swinunetr_label_{i}.nii.gz")

    dice_metric = DiceMetric(include_background=True,
                             reduction=MetricReduction.MEAN_BATCH,
                             get_not_nans=True)
    # Validator 생성 및 실행
    validator = Validator(
        args=args,
        model=model,
        test_loader=test_loader,
        class_list=["TC", "WT", "ET"],
        metric_functions=[["dice", dice_metric]],
        sliding_window_infer=window_infer,
        post_label=None,
        post_pred=post_pred_func)

    metrics, hausdorff_metrics= validator.run()
    for class_name in ["TC", "WT", "ET"]:
        print(f"{class_name} Dice Score: {metrics[f'{class_name}_dice']}")
        hd95_score = hausdorff_metrics[class_name]
        print(f"{class_name} HD95 Score: {hd95_score:.2f}")  # 평균 HD95 점수 출력

if __name__ == "__main__":
    main()
