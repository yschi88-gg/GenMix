import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import argparse
import tqdm
import matplotlib.pyplot as plt
import os
import shutil
from PIL import Image
from datetime import datetime
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
    parser = argparse.ArgumentParser(description="GenMix on CIFAR-10")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--exp_dir", type=str, default="logs_c10_clsp_scsa_gate")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs_uns", type=int, default=800)
    parser.add_argument("--epochs_sup", type=int, default=800)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument(
        "--lr_uns", type=float, default=0.06, help="Main model LR (SGD)"
    )
    parser.add_argument("--lr_sup", type=float, default=30)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument(
        "--use_clsp",
        action="store_true",
        default=True,
        help="Enable GLA synthetic positives",
    )
    parser.add_argument(
        "--lambda_syn", type=float, default=0.4, help="Weight for synthetic loss"
    )
    parser.add_argument(
        "--gen_lr", type=float, default=0.0001, help="Generator LR (Adam)"
    )
    parser.add_argument("--save_exp", type=bool, default=True)
    return parser.parse_args()


def get_path(args):
    if args.save_exp:
        path = args.exp_dir
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        path = os.path.join(path, datetime.now().strftime("%Y-%m-%d_%H-%M-%S/"))
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
    else:
        path = "./"
    args.path = path


def set_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_data(args):

    class SiameseTransform:

        def __init__(self, transform):
            self.transform = transform

        def __call__(self, img):
            return (self.transform(img), self.transform(img))

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.201)
    transform_train_uns = SiameseTransform(
        transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    32, scale=(0.2, 1.0), interpolation=Image.BICUBIC
                ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    )
    transform_train_sup = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                32, scale=(0.2, 1.0), interpolation=Image.BICUBIC
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize(int(32 * (8 / 7)), interpolation=Image.BICUBIC),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    dataset_train_uns = datasets.CIFAR10(
        args.data_dir, train=True, download=False, transform=transform_train_uns
    )
    dataset_train_sup = datasets.CIFAR10(
        args.data_dir, train=True, download=False, transform=transform_train_sup
    )
    dataset_train = datasets.CIFAR10(
        args.data_dir, train=True, download=False, transform=transform_test
    )
    dataset_test = datasets.CIFAR10(
        args.data_dir, train=False, download=False, transform=transform_test
    )
    loader_train_uns = torch.utils.data.DataLoader(
        dataset_train_uns,
        batch_size=args.batch_size,
        num_workers=5,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    loader_train_sup = torch.utils.data.DataLoader(
        dataset_train_sup,
        batch_size=args.batch_size,
        num_workers=5,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=5,
        shuffle=False,
        pin_memory=True,
    )
    loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        num_workers=5,
        shuffle=False,
        pin_memory=True,
    )
    return (loader_train_uns, loader_train_sup, loader_train, loader_test)


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
        x_attended = self.scsa(x_layer4)
        x = x_layer4 + x_attended
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def get_backbone():
    backbone = models.resnet18()
    backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    backbone.maxpool = nn.Identity()
    dim = backbone.fc.in_features
    backbone.fc = nn.Identity()
    scsa_module = SCSA(dim=dim, head_num=8, window_size=4)
    wrapped_backbone = ResNetWithSCSA(backbone, scsa_module)
    return (wrapped_backbone, dim)


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

    def d(self, p, z):
        return -F.cosine_similarity(p, z.detach(), dim=-1).mean()

    def forward(self, x1, x2):
        x0 = (x1 + x2) / 2
        z1, z2, z0 = (
            self.projector(self.backbone(x1)),
            self.projector(self.backbone(x2)),
            self.projector(self.backbone(x0)),
        )
        z_mix = self.mixer(z1, z2)
        p1, p2, p0 = (self.predictor(z1), self.predictor(z2), self.predictor(z0))
        loss_sim = self.d(p1, z2) / 4 + self.d(p2, z1) / 4 + self.d(p0, z_mix) / 2
        if self.use_clsp:
            z_syn = self.generator(z1.detach(), z2.detach())
            loss_syn_for_main = self.d(p1, z_syn) / 2 + self.d(p2, z_syn) / 2
            loss_s1_gen = -F.cosine_similarity(p1.detach(), z_syn, dim=-1).mean()
            loss_s2_gen = -F.cosine_similarity(p2.detach(), z_syn, dim=-1).mean()
            loss_gen = (loss_s1_gen + loss_s2_gen) / 2.0
            total_loss = (
                1 - self.lambda_syn
            ) * loss_sim + self.lambda_syn * loss_syn_for_main
        else:
            total_loss = loss_sim
            loss_gen = torch.tensor(0.0, device=z1.device)
            loss_syn_for_main = torch.tensor(0.0, device=z1.device)
        return (total_loss, loss_sim.detach(), loss_gen, loss_syn_for_main.detach())


def train_uns(model, loader, optimizer_main, optimizer_gen, scheduler_main, args):
    model.train()
    losses = []
    losses_gen = []
    lrs = []
    global_progress = tqdm.tqdm(range(args.epochs_uns), desc="Training_uns")
    for epoch in global_progress:
        loss_main_accum = 0
        loss_gen_accum = 0
        current_lr = scheduler_main.get_last_lr()[0] if epoch > 0 else args.lr_uns
        local_progress = tqdm.tqdm(
            loader, desc=f"Epoch {epoch}/{args.epochs_uns}", leave=False
        )
        for (img1, img2), _ in local_progress:
            img1, img2 = (img1.to(args.device), img2.to(args.device))
            loss_main, loss_sim, loss_gen, loss_syn_main = model(img1, img2)
            optimizer_main.zero_grad()
            loss_main.backward()
            optimizer_main.step()
            if args.use_clsp and optimizer_gen is not None:
                optimizer_gen.zero_grad()
                loss_gen.backward()
                optimizer_gen.step()
            loss_main_accum += loss_main.item() * len(img1)
            loss_gen_accum += loss_gen.item() * len(img1)
            local_progress.set_postfix(
                {"Loss": loss_main.item(), "GenLoss": loss_gen.item(), "lr": current_lr}
            )
        avg_main_loss = loss_main_accum / len(loader.dataset)
        avg_gen_loss = loss_gen_accum / len(loader.dataset)
        losses.append(avg_main_loss)
        losses_gen.append(avg_gen_loss)
        lrs.append(current_lr)
        scheduler_main.step()
    x = range(1, args.epochs_uns + 1)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plot1 = ax1.plot(x, losses, "-", label="Main Loss")
    plot2 = ax1.plot(x, losses_gen, "--g", label="Gen Loss")
    ax2 = ax1.twinx()
    plot3 = ax2.plot(x, lrs, "-r", label="Learning Rate")
    plots = plot1 + plot2 + plot3
    labs = [p.get_label() for p in plots]
    ax1.legend(plots, labs)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(axis="y")
    ax2.set_ylim(0, 1.1 * args.lr_uns)
    plt.title("Training_uns (Dual Opt)")
    plt.savefig(f"{args.path}Training_uns_{args.epochs_uns}.png")


def train_sup(
    backbone,
    classifier,
    loader,
    optimizer,
    scheduler,
    args,
    loader_train=None,
    loader_test=None,
):
    backbone.eval()
    classifier.train()
    losses = []
    accuracies = []
    accuracies_train = []
    accuracies_test = []
    lrs = []
    total = len(loader.dataset)
    global_progress = tqdm.tqdm(range(args.epochs_sup), desc="Training_sup")
    for epoch in global_progress:
        loss_ = 0
        correct = 0
        accuracy = 0
        lr = optimizer.state_dict()["param_groups"][0]["lr"]
        local_progress = tqdm.tqdm(
            loader, desc=f"Epoch {epoch}/{args.epochs_sup}", leave=False
        )
        for idx, (data, target) in enumerate(local_progress):
            classifier.zero_grad()
            with torch.no_grad():
                feature = backbone(data.to(args.device))
            output = classifier(feature)
            loss = F.cross_entropy(output, target.to(args.device))
            loss.backward()
            optimizer.step()
            pred = F.log_softmax(output, dim=1).argmax(dim=1)
            correct += (pred == target.to(args.device)).sum().item()
            accuracy = 100.0 * correct / total
            loss_ += loss.item() * len(target)
            local_progress.set_postfix(
                {"loss": loss.item(), "lr": lr, "Accuracy": "{:.2f}%".format(accuracy)}
            )
            accuracy_train = 0
            if idx == len(loader) - 1 and loader_train:
                correct_train = 0
                with torch.no_grad():
                    for data_train, target_train in loader_train:
                        pred = F.log_softmax(
                            classifier(backbone(data_train.to(args.device))), dim=1
                        ).argmax(dim=1)
                        correct_train += (
                            (pred == target_train.to(args.device)).sum().item()
                        )
                        accuracy_train = (
                            100.0 * correct_train / len(loader_train.dataset)
                        )
                accuracies_train.append(accuracy_train)
            accuracy_test = 0
            if idx == len(loader) - 1 and loader_test:
                correct_test = 0
                with torch.no_grad():
                    for data_test, target_test in loader_test:
                        pred = F.log_softmax(
                            classifier(backbone(data_test.to(args.device))), dim=1
                        ).argmax(dim=1)
                        correct_test += (
                            (pred == target_test.to(args.device)).sum().item()
                        )
                        accuracy_test = 100.0 * correct_test / len(loader_test.dataset)
                accuracies_test.append(accuracy_test)
        losses.append(loss_ / total)
        accuracies.append(accuracy)
        lrs.append(lr)
        scheduler.step()
    x = range(1, args.epochs_sup + 1)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plot1 = ax1.plot(x, accuracies, "-", label="Train Accuracy")
    plots = plot1
    if loader_train:
        plot2 = ax1.plot(x, accuracies_train, "-", label="Test Accuracy of Train")
        plots = plots + plot2
    if loader_test:
        plot3 = ax1.plot(x, accuracies_test, "-", label="Test Accuracy of Test")
        plots = plots + plot3
    ax2 = ax1.twinx()
    plot4 = ax2.plot(x, lrs, "-r", label="Learning Rate")
    plots = plots + plot4
    labs = [p.get_label() for p in plots]
    ax1.legend(plots, labs)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy(%)")
    ax1.grid(axis="y")
    ax2.set_ylim(-0.1 * args.lr_sup, 1.1 * args.lr_sup)
    plt.title("Training_sup")
    plt.savefig(f"{args.path}Training_sup_{args.epochs_sup}.png")


def test(backbone, classifier, loader_train, loader_test, args):
    backbone.eval()
    classifier.eval()
    correct_train = 0.0
    correct_test = 0.0
    correct_test_top5 = 0
    result = torch.zeros(10, 10)
    progress_train = tqdm.tqdm(loader_train, desc="Testing_train")
    with torch.no_grad():
        for data, target in progress_train:
            pred = F.log_softmax(
                classifier(backbone(data.to(args.device))), dim=1
            ).argmax(dim=1)
            correct_train += (pred == target.to(args.device)).sum().item()
            progress_train.set_postfix(
                {
                    "Accuracy": "{:.2f}%".format(
                        100.0 * correct_train / len(loader_train.dataset)
                    )
                }
            )
    progress_test = tqdm.tqdm(loader_test, desc="Testing_test")
    with torch.no_grad():
        for data, target in progress_test:
            logits = classifier(backbone(data.to(args.device)))
            pred = logits.argmax(dim=1)
            top5 = logits.topk(5, dim=1).indices
            correct_test_top5 += (
                top5.eq(target.to(args.device).unsqueeze(1)).any(dim=1).sum().item()
            )
            for i in range(len(target)):
                result[target[i]][pred[i]] += 1
            correct_test += (pred == target.to(args.device)).sum().item()
            progress_test.set_postfix(
                {
                    "Accuracy": "{:.2f}%".format(
                        100.0 * correct_test / len(loader_test.dataset)
                    )
                }
            )
    test_top1 = 100.0 * correct_test / len(loader_test.dataset)
    test_top5 = 100.0 * correct_test_top5 / len(loader_test.dataset)
    print(f"Final Test Top-1: {test_top1:.2f}%, Top-5: {test_top5:.2f}%")
    labels = [
        "Plane",
        "Car ",
        "Bird",
        "Cat ",
        "Deer",
        "Dog ",
        "Frog",
        "Horse",
        "Ship",
        "Truck",
    ]
    print("\nTest Results Matrix:")
    print("Tgt\\Prd", *labels, "RECALL(%)", sep="\t")
    for i, t in enumerate(result / 1000):
        print(labels[i], end="\t")
        for j, p in enumerate(t):
            print("{:.3f}".format(p), "*" if i == j else " ", end="\t", sep="")
        print("{:.1f}%".format(t[i] * 100))
    print("SUM ", end="\t")
    for i in result.sum(dim=0):
        print("{:.3f}".format(i / 1000), end="\t")
    print("\nPRE(%)", end="\t")
    for i in range(10):
        r = 0.1 * result[i][i]
        p = 1000 * r / result.sum(dim=0)[i] if result.sum(dim=0)[i] else 100
        print("{:.2f}%".format(p), end="\t")
    print()
    if args.save_exp:
        with open(
            "{}result_{:.2f}%_{:.2f}%.txt".format(
                args.path,
                100.0 * correct_train / len(loader_train.dataset),
                100.0 * correct_test / len(loader_test.dataset),
            ),
            "w",
        ) as f:
            print(f"Test Top-1: {test_top1:.2f}%", file=f)
            print(f"Test Top-5: {test_top5:.2f}%", file=f)
            print("Tgt\\Prd", *labels, "RECALL(%)", sep="\t", file=f)
            for i, t in enumerate(result / 1000):
                print(labels[i], end="\t", file=f)
                for j, p in enumerate(t):
                    print(
                        "{:.3f}".format(p),
                        "*" if i == j else " ",
                        end="\t",
                        sep="",
                        file=f,
                    )
                print("{:.1f}%".format(t[i] * 100), file=f)
            print("SUM ", end="\t", file=f)
            for i in result.sum(dim=0):
                print("{:.3f}".format(i / 1000), end="\t", file=f)
            print("\nPRE(%)", end="\t", file=f)
            for i in range(10):
                r = 0.1 * result[i][i]
                p = 1000 * r / result.sum(dim=0)[i] if result.sum(dim=0)[i] else 100
                print("{:.2f}%".format(p), end="\t", file=f)


def main():
    args = get_args()
    get_path(args)
    set_seeds(args.seed)
    loader_train_uns, loader_train_sup, loader_train, loader_test = get_data(args)
    backbone, dim = get_backbone()
    model = GenMix(
        backbone, dim, lambda_syn=args.lambda_syn, use_clsp=args.use_clsp
    ).to(args.device)
    classifier = nn.Linear(in_features=dim, out_features=10, bias=True).to(args.device)
    if args.use_clsp:
        main_params = (
            list(model.backbone.parameters())
            + list(model.projector.parameters())
            + list(model.predictor.parameters())
            + list(model.mixer.parameters())
        )
        gen_params = model.generator.parameters()
        optimizer_main = torch.optim.SGD(
            main_params,
            lr=args.lr_uns,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        optimizer_gen = torch.optim.Adam(gen_params, lr=args.gen_lr)
    else:
        optimizer_main = torch.optim.SGD(
            model.parameters(),
            lr=args.lr_uns,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        optimizer_gen = None
    scheduler_main = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_main, T_max=args.epochs_uns, eta_min=0
    )
    optimizer_sup = torch.optim.SGD(
        classifier.parameters(), lr=args.lr_sup, momentum=args.momentum
    )
    scheduler_sup = torch.optim.lr_scheduler.StepLR(
        optimizer_sup, step_size=40, gamma=0.6
    )
    flag_uns, flag_sup = (True, True)
    if flag_uns:
        set_seeds(args.seed)
        print("--- Starting Dual Optimizer Training ---")
        train_uns(
            model, loader_train_uns, optimizer_main, optimizer_gen, scheduler_main, args
        )
        torch.save(model.state_dict(), f"{args.path}model.pt")
    elif os.path.exists(f"{args.path}model.pt"):
        model.load_state_dict(torch.load(f"{args.path}model.pt"))
    else:
        print("Warning: Pretrained model not found!")
    if flag_sup:
        set_seeds(args.seed)
        print("--- Starting Linear Evaluation ---")
        train_sup(
            backbone,
            classifier,
            loader_train_sup,
            optimizer_sup,
            scheduler_sup,
            args,
            loader_train=loader_train,
            loader_test=loader_test,
        )
        torch.save(classifier.state_dict(), f"{args.path}classifier.pt")
    elif os.path.exists(f"{args.path}classifier.pt"):
        classifier.load_state_dict(torch.load(f"{args.path}classifier.pt"))
    test(backbone, classifier, loader_train, loader_test, args)
    if args.save_exp:
        shutil.copyfile(__file__, f"{args.path}main.py")


if __name__ == "__main__":
    main()
