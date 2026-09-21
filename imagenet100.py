import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import argparse
import tqdm
import os
import random
import shutil
import math
from datetime import datetime
from PIL import Image, ImageFilter
import typing as t
from einops import rearrange


class SCSA(nn.Module):

    def __init__(
        self,
        dim: int,
        head_num: int,
        window_size: int = 7,
        group_kernel_sizes: t.List[int] = [3, 5, 7, 9],
        qkv_bias: bool = False,
        fuse_bn: bool = False,
        down_sample_mode: str = "avg_pool",
        attn_drop_ratio: float = 0.0,
        gate_layer: str = "sigmoid",
    ):
        super(SCSA, self).__init__()
        self.dim = dim
        self.head_num = head_num
        self.head_dim = dim // head_num
        self.scaler = self.head_dim ** (-0.5)
        self.group_kernel_sizes = group_kernel_sizes
        self.window_size = window_size
        self.qkv_bias = qkv_bias
        self.fuse_bn = fuse_bn
        self.down_sample_mode = down_sample_mode
        assert (
            self.dim % 4 == 0
        ), "The dimension of input feature should be divisible by 4."
        self.group_chans = group_chans = self.dim // 4
        self.local_dwc = nn.Conv1d(
            group_chans,
            group_chans,
            kernel_size=group_kernel_sizes[0],
            padding=group_kernel_sizes[0] // 2,
            groups=group_chans,
        )
        self.global_dwc_s = nn.Conv1d(
            group_chans,
            group_chans,
            kernel_size=group_kernel_sizes[1],
            padding=group_kernel_sizes[1] // 2,
            groups=group_chans,
        )
        self.global_dwc_m = nn.Conv1d(
            group_chans,
            group_chans,
            kernel_size=group_kernel_sizes[2],
            padding=group_kernel_sizes[2] // 2,
            groups=group_chans,
        )
        self.global_dwc_l = nn.Conv1d(
            group_chans,
            group_chans,
            kernel_size=group_kernel_sizes[3],
            padding=group_kernel_sizes[3] // 2,
            groups=group_chans,
        )
        self.sa_gate = nn.Softmax(dim=2) if gate_layer == "softmax" else nn.Sigmoid()
        self.norm_h = nn.GroupNorm(4, dim)
        self.norm_w = nn.GroupNorm(4, dim)
        self.conv_d = nn.Identity()
        self.norm = nn.GroupNorm(1, dim)
        self.q = nn.Conv2d(
            in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim
        )
        self.k = nn.Conv2d(
            in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim
        )
        self.v = nn.Conv2d(
            in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim
        )
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.ca_gate = nn.Softmax(dim=1) if gate_layer == "softmax" else nn.Sigmoid()
        if window_size == -1:
            self.down_func = nn.AdaptiveAvgPool2d((1, 1))
        elif down_sample_mode == "recombination":
            self.down_func = self.space_to_chans
            self.conv_d = nn.Conv2d(
                in_channels=dim * window_size**2,
                out_channels=dim,
                kernel_size=1,
                bias=False,
            )
        elif down_sample_mode == "avg_pool":
            self.down_func = nn.AvgPool2d(
                kernel_size=(window_size, window_size), stride=window_size
            )
        elif down_sample_mode == "max_pool":
            self.down_func = nn.MaxPool2d(
                kernel_size=(window_size, window_size), stride=window_size
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h_, w_ = x.size()
        x_h = x.mean(dim=3)
        l_x_h, g_x_h_s, g_x_h_m, g_x_h_l = torch.split(x_h, self.group_chans, dim=1)
        x_w = x.mean(dim=2)
        l_x_w, g_x_w_s, g_x_w_m, g_x_w_l = torch.split(x_w, self.group_chans, dim=1)
        x_h_attn = self.sa_gate(
            self.norm_h(
                torch.cat(
                    (
                        self.local_dwc(l_x_h),
                        self.global_dwc_s(g_x_h_s),
                        self.global_dwc_m(g_x_h_m),
                        self.global_dwc_l(g_x_h_l),
                    ),
                    dim=1,
                )
            )
        )
        x_h_attn = x_h_attn.view(b, c, h_, 1)
        x_w_attn = self.sa_gate(
            self.norm_w(
                torch.cat(
                    (
                        self.local_dwc(l_x_w),
                        self.global_dwc_s(g_x_w_s),
                        self.global_dwc_m(g_x_w_m),
                        self.global_dwc_l(g_x_w_l),
                    ),
                    dim=1,
                )
            )
        )
        x_w_attn = x_w_attn.view(b, c, 1, w_)
        x = x * x_h_attn * x_w_attn
        y = self.down_func(x)
        y = self.conv_d(y)
        _, _, h_, w_ = y.size()
        y = self.norm(y)
        q = self.q(y)
        k = self.k(y)
        v = self.v(y)
        q = rearrange(
            q,
            "b (head_num head_dim) h w -> b head_num head_dim (h w)",
            head_num=int(self.head_num),
            head_dim=int(self.head_dim),
        )
        k = rearrange(
            k,
            "b (head_num head_dim) h w -> b head_num head_dim (h w)",
            head_num=int(self.head_num),
            head_dim=int(self.head_dim),
        )
        v = rearrange(
            v,
            "b (head_num head_dim) h w -> b head_num head_dim (h w)",
            head_num=int(self.head_num),
            head_dim=int(self.head_dim),
        )
        attn = q @ k.transpose(-2, -1) * self.scaler
        attn = self.attn_drop(attn.softmax(dim=-1))
        attn = attn @ v
        attn = rearrange(
            attn,
            "b head_num head_dim (h w) -> b (head_num head_dim) h w",
            h=int(h_),
            w=int(w_),
        )
        attn = attn.mean((2, 3), keepdim=True)
        attn = self.ca_gate(attn)
        return attn * x


def get_args():
    parser = argparse.ArgumentParser(description="GenMix on ImageNet-100")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/imagenet100_ready",
        help="Path to ImageNet-100 (must contain train/ and val/)",
    )
    parser.add_argument(
        "--exp_dir", type=str, default="logs_imagenet100_full", help="Log directory"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--epochs_uns", type=int, default=400, help="Pre-training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--lr_uns", type=float, default=0.5, help="Main-model learning rate"
    )
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-05)
    parser.add_argument(
        "--lr_online", type=float, default=0.1, help="Online classifier LR"
    )
    parser.add_argument(
        "--epochs_sup", type=int, default=100, help="Linear eval epochs"
    )
    parser.add_argument("--lr_sup", type=float, default=30.0, help="Linear eval LR")
    parser.add_argument("--step_size", type=int, default=30)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--use_clsp", action="store_true", help="Enable GLA generator")
    parser.add_argument("--lambda_syn", type=float, default=0.4)
    parser.add_argument("--gen_lr", type=float, default=0.0001)
    return parser.parse_args()


def get_path(args):
    path = args.exp_dir + datetime.now().strftime("_%Y%m%d_%H%M%S/")
    os.makedirs(path, exist_ok=True)
    args.path = path


def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class GaussianBlur(object):

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.0):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        if random.random() < self.prob:
            radius = random.uniform(self.radius_min, self.radius_max)
            return img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img


class TwoCropsTransform:

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


def get_data(args):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    aug_uns = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    aug_sup = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    aug_test = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )
    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")
    train_dataset_uns = datasets.ImageFolder(
        train_dir, transform=TwoCropsTransform(aug_uns)
    )
    train_dataset_sup = datasets.ImageFolder(train_dir, transform=aug_sup)
    val_dataset = datasets.ImageFolder(val_dir, transform=aug_test)
    print(f"Dataset: {args.data_dir}")
    print(f"Classes: {len(train_dataset_uns.classes)}")
    loader_uns = torch.utils.data.DataLoader(
        train_dataset_uns,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )
    loader_sup = torch.utils.data.DataLoader(
        train_dataset_sup,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )
    loader_val = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )
    return (loader_uns, loader_sup, loader_val, len(train_dataset_uns.classes))


class ProjectionMLP(nn.Module):

    def __init__(self, input_dim, hidden_dim=2048, output_dim=2048):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, output_dim), nn.BatchNorm1d(output_dim)
        )

    def forward(self, x):
        return self.layer2(self.layer1(x))


class PredictionMLP(nn.Module):

    def __init__(self, input_dim=2048, hidden_dim=512, output_dim=2048):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.layer2(self.layer1(x))


class AdaptiveFeatureMixer(nn.Module):

    def __init__(self, input_dim=2048, hidden_dim=512):
        super().__init__()
        self.gate_mlp = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, z1, z2):
        m = self.gate_mlp(torch.cat([z1, z2], dim=1))
        return m * z1 + (1 - m) * z2


class GenerativeLatentAugmentation(nn.Module):

    def __init__(self, input_dim=2048, hidden_dim=1024, output_dim=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z1, z2):
        return self.net(torch.cat([z1, z2], dim=1))


class GenMix(nn.Module):

    def __init__(self, backbone, input_dim, lambda_syn=0.4, use_clsp=True):
        super().__init__()
        self.backbone = backbone
        self.projector = ProjectionMLP(input_dim)
        self.predictor = PredictionMLP()
        self.mixer = AdaptiveFeatureMixer()
        self.lambda_syn = lambda_syn
        self.use_clsp = use_clsp
        if use_clsp:
            self.generator = GenerativeLatentAugmentation()

    def forward(self, x1, x2):
        x0 = (x1 + x2) / 2
        f_x1 = self.backbone(x1)
        f_x2 = self.backbone(x2)
        f_x0 = self.backbone(x0)
        z1, z2, z0 = (self.projector(f_x1), self.projector(f_x2), self.projector(f_x0))
        z_mix = self.mixer(z1, z2)
        p1, p2, p0 = (self.predictor(z1), self.predictor(z2), self.predictor(z0))
        loss_sim = self.d(p1, z2) / 4 + self.d(p2, z1) / 4 + self.d(p0, z_mix) / 2
        loss_gen = torch.tensor(0.0, device=x1.device)
        loss_syn_main = torch.tensor(0.0, device=x1.device)
        total_loss = loss_sim
        if self.use_clsp:
            z_syn = self.generator(z1.detach(), z2.detach())
            loss_syn_main = self.d(p1, z_syn) / 2 + self.d(p2, z_syn) / 2
            loss_gen = (
                -F.cosine_similarity(p1.detach(), z_syn, dim=-1).mean()
                + -F.cosine_similarity(p2.detach(), z_syn, dim=-1).mean()
            ) / 2.0
            total_loss = (
                1 - self.lambda_syn
            ) * loss_sim + self.lambda_syn * loss_syn_main
        return (
            total_loss,
            loss_sim.detach(),
            loss_gen,
            loss_syn_main.detach(),
            f_x1.detach(),
        )

    @staticmethod
    def d(p, z):
        return -F.cosine_similarity(p, z.detach(), dim=-1).mean()


class ResNetWithSCSA(nn.Module):

    def __init__(self, original_resnet, scsa_module):
        super().__init__()
        self.conv1 = original_resnet.conv1
        self.bn1 = original_resnet.bn1
        self.relu = original_resnet.relu
        self.maxpool = original_resnet.maxpool
        self.layer1 = original_resnet.layer1
        self.layer2 = original_resnet.layer2
        self.layer3 = original_resnet.layer3
        self.layer4 = original_resnet.layer4
        self.scsa = scsa_module
        self.avgpool = original_resnet.avgpool
        self.fc = original_resnet.fc

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x_layer4 = self.layer4(x)
        x = x_layer4 + self.scsa(x_layer4)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def get_backbone(dim_scsa=512):
    backbone = models.resnet18(pretrained=False)
    dim_features = backbone.fc.in_features
    backbone.fc = nn.Identity()
    scsa = SCSA(dim=dim_features, head_num=8, window_size=7)
    return (ResNetWithSCSA(backbone, scsa), dim_features)


def validate(backbone, classifier, loader, args):
    backbone.eval()
    classifier.eval()
    top1 = 0
    top5 = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = (x.to(args.device), y.to(args.device))
            out = classifier(backbone(x))
            acc1, acc5 = accuracy(out, y, topk=(1, 5))
            bs = x.size(0)
            top1 += acc1.item() * bs
            top5 += acc5.item() * bs
            total += bs
    return (top1 / total, top5 / total)


def train_phase1_pretraining(
    model,
    online_clf,
    loader_train,
    loader_val,
    opt_main,
    opt_gen,
    opt_online,
    sched_main,
    args,
):
    log_file = os.path.join(args.path, "online_acc_log.txt")
    with open(log_file, "w") as f:
        f.write("Epoch,Loss,Top1,Top5\n")
    print(f"\n>>> Phase 1: Pre-training started. Logging to {log_file}")
    pbar = tqdm.trange(args.epochs_uns, desc="Pre-training")
    for epoch in pbar:
        model.train()
        online_clf.train()
        loss_m = 0
        loss_o = 0
        for images, target in loader_train:
            x1, x2 = (images[0].to(args.device), images[1].to(args.device))
            target = target.to(args.device)
            loss_main, _, loss_gen, _, f_x1 = model(x1, x2)
            opt_main.zero_grad()
            loss_main.backward()
            opt_main.step()
            if args.use_clsp and opt_gen is not None:
                opt_gen.zero_grad()
                loss_gen.backward()
                opt_gen.step()
            opt_online.zero_grad()
            logits = online_clf(f_x1)
            loss_online = F.cross_entropy(logits, target)
            loss_online.backward()
            opt_online.step()
            loss_m += loss_main.item()
            loss_o += loss_online.item()
        sched_main.step()
        acc1, acc5 = validate(model.backbone, online_clf, loader_val, args)
        avg_loss = loss_m / len(loader_train)
        with open(log_file, "a") as f:
            f.write(f"{epoch},{avg_loss:.4f},{acc1:.2f},{acc5:.2f}\n")
        pbar.set_description(
            f"Ep {epoch} | Loss:{avg_loss:.3f} | Online Acc@1:{acc1:.2f}%"
        )
    torch.save(model.state_dict(), os.path.join(args.path, "model_pretrained.pt"))
    print(">>> Phase 1 Finished.")


def train_phase2_lineareval(backbone, dim, num_classes, loader_train, loader_val, args):
    print(f"\n>>> Phase 2: Standard Linear Evaluation started.")
    for param in backbone.parameters():
        param.requires_grad = False
    backbone.eval()
    classifier = nn.Linear(dim, num_classes).to(args.device)
    optimizer = torch.optim.SGD(
        classifier.parameters(), lr=args.lr_sup, momentum=args.momentum, weight_decay=0
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs_sup
    )
    log_file = os.path.join(args.path, "final_linear_log.txt")
    with open(log_file, "w") as f:
        f.write("Epoch,Top1,Top5\n")
    pbar = tqdm.trange(args.epochs_sup, desc="Linear Eval")
    for epoch in pbar:
        classifier.train()
        total_loss = 0
        for x, y in loader_train:
            x, y = (x.to(args.device), y.to(args.device))
            with torch.no_grad():
                feat = backbone(x)
            optimizer.zero_grad()
            out = classifier(feat)
            loss = F.cross_entropy(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        acc1, acc5 = validate(backbone, classifier, loader_val, args)
        with open(log_file, "a") as f:
            f.write(f"{epoch},{acc1:.2f},{acc5:.2f}\n")
        pbar.set_description(f"Eval Ep {epoch} | Acc@1:{acc1:.2f}%")
    print(f">>> Phase 2 Finished. Final Acc: {acc1:.2f}%")
    torch.save(classifier.state_dict(), os.path.join(args.path, "final_classifier.pt"))


def main():
    args = get_args()
    get_path(args)
    set_seeds(args.seed)
    print(f"--- GenMix: ImageNet-100 ---")
    print(f"Output Dir: {args.path}")
    loader_uns, loader_sup, loader_val, num_classes = get_data(args)
    backbone, dim = get_backbone()
    model = GenMix(backbone, dim, args.lambda_syn, args.use_clsp).to(args.device)
    online_clf = nn.Linear(dim, num_classes).to(args.device)
    opt_online = torch.optim.SGD(
        online_clf.parameters(), lr=args.lr_online, momentum=0.9, weight_decay=0
    )
    if args.use_clsp:
        main_params = (
            list(model.backbone.parameters())
            + list(model.projector.parameters())
            + list(model.predictor.parameters())
            + list(model.mixer.parameters())
        )
        opt_main = torch.optim.SGD(
            main_params,
            lr=args.lr_uns,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        opt_gen = torch.optim.Adam(model.generator.parameters(), lr=args.gen_lr)
    else:
        opt_main = torch.optim.SGD(
            model.parameters(),
            lr=args.lr_uns,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        opt_gen = None
    sched_main = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_main, T_max=args.epochs_uns
    )
    train_phase1_pretraining(
        model,
        online_clf,
        loader_uns,
        loader_val,
        opt_main,
        opt_gen,
        opt_online,
        sched_main,
        args,
    )
    train_phase2_lineareval(
        model.backbone, dim, num_classes, loader_sup, loader_val, args
    )
    if os.path.exists(__file__):
        shutil.copyfile(__file__, os.path.join(args.path, "code_backup.py"))
    print("All Done.")


if __name__ == "__main__":
    main()
