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
    parser = argparse.ArgumentParser(description="GenMix on CIFAR-100")
    parser.add_argument(
        "--data_dir", type=str, default="data_cifar100", help="Dataset root directory"
    )
    parser.add_argument(
        "--exp_dir",
        type=str,
        default="logs_clsp_dual_optimizer",
        help="Directory for logs/models",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs_uns", type=int, default=800)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument(
        "--lr_uns", type=float, default=0.06, help="Learning rate for main model (SGD)"
    )
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--epochs_sup", type=int, default=800)
    parser.add_argument("--lr_sup", type=float, default=30)
    parser.add_argument("--step_size", type=int, default=40)
    parser.add_argument("--gamma", type=float, default=0.6)
    parser.add_argument(
        "--use_clsp", action="store_true", help="Enable GLA synthetic positives"
    )
    parser.add_argument(
        "--lambda_syn",
        type=float,
        default=0.4,
        help="Weight for synthetic positive loss (for main model)",
    )
    parser.add_argument(
        "--gen_lr",
        type=float,
        default=0.0001,
        help="Learning rate for generator (Adam)",
    )
    parser.add_argument("--save_exp", type=bool, default=True)
    return parser.parse_args()


def get_path(args):
    path = args.exp_dir + datetime.now().strftime("_%Y%m%d_%H%M%S/")
    os.makedirs(path, exist_ok=True)
    args.path = path


def set_seeds(seed):
    torch.manual_seed(seed)
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

    mean, std = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    transform_common = transforms.Compose(
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
    transform_train_uns = SiameseTransform(transform_common)
    transform_train_sup = transform_common
    transform_test = transforms.Compose(
        [
            transforms.Resize(int(32 * (8 / 7)), interpolation=Image.BICUBIC),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    train_uns = datasets.CIFAR100(
        args.data_dir, train=True, download=True, transform=transform_train_uns
    )
    train_sup = datasets.CIFAR100(
        args.data_dir, train=True, download=True, transform=transform_train_sup
    )
    test = datasets.CIFAR100(
        args.data_dir, train=False, download=True, transform=transform_test
    )
    loader_train_uns = torch.utils.data.DataLoader(
        train_uns,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    loader_train_sup = torch.utils.data.DataLoader(
        train_sup,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    loader_test = torch.utils.data.DataLoader(
        test, batch_size=args.batch_size, num_workers=4, shuffle=False, pin_memory=True
    )
    return (loader_train_uns, loader_train_sup, loader_test)


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
        x_attended = self.scsa(x_layer4)
        x = x_layer4 + x_attended
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def get_backbone(dim_scsa=512):
    backbone = models.resnet18()
    backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    backbone.maxpool = nn.Identity()
    dim_features = backbone.fc.in_features
    backbone.fc = nn.Identity()
    scsa_module = SCSA(dim=dim_features, head_num=8, window_size=4)
    wrapped_backbone = ResNetWithSCSA(backbone, scsa_module)
    return (wrapped_backbone, dim_features)


def train_uns(model, loader, optimizer_main, optimizer_gen, scheduler_main, args):
    model.train()
    epoch_pbar = tqdm.trange(args.epochs_uns, desc="Unsupervised Training")
    for epoch in epoch_pbar:
        total_main_loss, total_sim_loss, total_gen_loss, total_syn_main_loss = (
            0,
            0,
            0,
            0,
        )
        batch_pbar = tqdm.tqdm(
            loader,
            desc=f"Epoch {epoch:03d}/{args.epochs_uns}",
            leave=False,
            total=len(loader),
        )
        for (x1, x2), _ in batch_pbar:
            x1, x2 = (x1.to(args.device), x2.to(args.device))
            loss_main, loss_sim, loss_gen, loss_syn_main = model(x1, x2)
            optimizer_main.zero_grad()
            loss_main.backward()
            optimizer_main.step()
            if args.use_clsp and optimizer_gen is not None:
                optimizer_gen.zero_grad()
                loss_gen.backward()
                optimizer_gen.step()
            current_lr = scheduler_main.get_last_lr()[0]
            batch_pbar.set_postfix(
                {
                    "loss": loss_main.item(),
                    "lr": current_lr,
                    "gen_loss": loss_gen.item(),
                }
            )
            total_main_loss += loss_main.item() * len(x1)
            total_sim_loss += loss_sim.item() * len(x1)
            total_gen_loss += loss_gen.item() * len(x1)
            total_syn_main_loss += loss_syn_main.item() * len(x1)
        scheduler_main.step()
        n_samples = len(loader.dataset)
        avg_main = total_main_loss / n_samples
        avg_sim = total_sim_loss / n_samples
        avg_gen = total_gen_loss / n_samples
        avg_syn_main = total_syn_main_loss / n_samples
        epoch_summary = f"MainLoss={avg_main:.4f} (Sim={avg_sim:.4f}, SynMain={avg_syn_main:.4f}), GenLoss={avg_gen:.4f}"
        print(f"Epoch {epoch:03d} Summary: {epoch_summary}")
        epoch_pbar.set_description(f"Unsupervised Training (Epoch {epoch:03d})")
    if args.save_exp:
        torch.save(model.state_dict(), f"{args.path}/model_uns.pt")


def train_sup(backbone, classifier, loader, optimizer, scheduler, args, loader_test):
    backbone.eval()
    classifier.train()
    for epoch in tqdm.trange(args.epochs_sup, desc="Linear Eval"):
        total_loss, correct = (0, 0)
        for x, y in loader:
            x, y = (x.to(args.device), y.to(args.device))
            optimizer.zero_grad()
            with torch.no_grad():
                feat = backbone(x)
            out = classifier(feat)
            loss = F.cross_entropy(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (out.argmax(dim=1) == y).sum().item()
        scheduler.step()
        acc = 100.0 * correct / len(loader.dataset)
        print(
            f"Epoch {epoch:03d}: loss={total_loss / len(loader.dataset):.4f}, acc={acc:.2f}%"
        )
    test(backbone, classifier, loader_test, args)


def test(backbone, classifier, loader_test, args):
    backbone.eval()
    classifier.eval()
    correct = 0
    correct_top5 = 0
    with torch.no_grad():
        for x, y in tqdm.tqdm(loader_test, desc="Testing"):
            x, y = (x.to(args.device), y.to(args.device))
            logits = classifier(backbone(x))
            pred = logits.argmax(dim=1)
            correct_top5 += (
                logits.topk(5, dim=1).indices.eq(y.unsqueeze(1)).any(dim=1).sum().item()
            )
            correct += (pred == y).sum().item()
    acc = 100.0 * correct / len(loader_test.dataset)
    acc5 = 100.0 * correct_top5 / len(loader_test.dataset)
    print(f"Final Test Top-1: {acc:.2f}%, Top-5: {acc5:.2f}%")
    if args.save_exp:
        with open(f"{args.path}/result.txt", "w") as f:
            f.write(f"Test Top-1: {acc:.2f}%\nTest Top-5: {acc5:.2f}%\n")


def main():
    args = get_args()
    get_path(args)
    set_seeds(args.seed)
    loader_train_uns, loader_train_sup, loader_test = get_data(args)
    backbone, dim = get_backbone()
    model = GenMix(
        backbone, dim, lambda_syn=args.lambda_syn, use_clsp=args.use_clsp
    ).to(args.device)
    if args.use_clsp:
        print("GLA enabled: separate main-model and generator optimizers")
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
        print("GLA disabled: main-model optimizer only")
        optimizer_main = torch.optim.SGD(
            model.parameters(),
            lr=args.lr_uns,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        optimizer_gen = None
    scheduler_main = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_main, T_max=args.epochs_uns
    )
    print(f"\n--- Unsupervised Training ---")
    print(
        f"Main LR (SGD): {args.lr_uns}, GLA generator LR (Adam): {(args.gen_lr if args.use_clsp else 'N/A')}"
    )
    train_uns(
        model, loader_train_uns, optimizer_main, optimizer_gen, scheduler_main, args
    )
    print(f"\n--- Linear Evaluation ---")
    classifier = nn.Linear(dim, 100).to(args.device)
    optimizer_sup = torch.optim.SGD(
        classifier.parameters(), lr=args.lr_sup, momentum=args.momentum
    )
    scheduler_sup = torch.optim.lr_scheduler.StepLR(
        optimizer_sup, step_size=args.step_size, gamma=args.gamma
    )
    train_sup(
        backbone,
        classifier,
        loader_train_sup,
        optimizer_sup,
        scheduler_sup,
        args,
        loader_test,
    )
    if args.save_exp:
        shutil.copyfile(__file__, f"{args.path}/{os.path.basename(__file__)}")
        print(f"Code and results saved to: {args.path}")


if __name__ == "__main__":
    main()
