import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset

from torch.optim.swa_utils import AveragedModel, update_bn

import numpy as np

from torchvision import transforms

from torchvision.models import resnet18, resnet34, resnet50

from collections import OrderedDict

import random

import copy



# Reproducibility

SEED = 42

torch.manual_seed(SEED)

np.random.seed(SEED)

random.seed(SEED)

if torch.cuda.is_available():

    torch.cuda.manual_seed(SEED)

    torch.cuda.manual_seed_all(SEED)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = True



# Hyperparameters

MODEL_ARCH    = 'resnet50'

NUM_CLASSES   = 9



EPOCHS        = 200

BATCH_SIZE    = 128

LR            = 0.1

WEIGHT_DECAY  = 5e-4

MOMENTUM      = 0.9



EPS           = 8 / 255

STEP_SIZE     = 2 / 255

NUM_STEPS     = 10

EVAL_STEPS    = 20

BETA          = 6.0            # TRADES regularization (used in Phase 2+)



AWP_GAMMA     = 0.005          # AWP perturbation strength

AWP_WARMUP    = 10             # Phase 2 starts: switch from PGD-AT to TRADES+AWP

SWA_START     = 160            # Phase 3 starts: add SWA

SWA_LR        = 0.001



CUTOUT_SIZE   = 16

LABEL_SMOOTH  = 0.1

GRAD_CLIP     = 1.0            # conservative clipping for early stability

MAX_LOSS      = 50.0



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

print(f"Model: {MODEL_ARCH}")





# Augmentations

class Cutout:

    def __init__(self, size):

        self.size = size



    def __call__(self, img):

        _, h, w = img.shape

        y = random.randint(0, h - 1)

        x = random.randint(0, w - 1)

        y1, y2 = max(0, y - self.size // 2), min(h, y + self.size // 2)

        x1, x2 = max(0, x - self.size // 2), min(w, x + self.size // 2)

        img = img.clone()

        img[:, y1:y2, x1:x2] = 0.0

        return img





class AugmentedDataset(Dataset):

    def __init__(self, subset, transform=None):

        self.subset = subset

        self.transform = transform



    def __len__(self):

        return len(self.subset)



    def __getitem__(self, idx):

        img, label = self.subset[idx]

        if self.transform:

            img = self.transform(img)

        return img, label





# AWP (Adversarial Weight Perturbation)

AWP_EPS = 1e-20





def diff_in_weights(model, proxy):

    diff_dict = OrderedDict()

    model_state = model.state_dict()

    proxy_state = proxy.state_dict()

    for (k1, w1), (k2, w2) in zip(model_state.items(), proxy_state.items()):

        if len(w1.size()) <= 1:

            continue

        if 'weight' in k1:

            diff_w = w2 - w1

            diff_dict[k1] = w1.norm() / (diff_w.norm() + AWP_EPS) * diff_w

    return diff_dict





def add_into_weights(model, diff, coeff=1.0):

    names_in_diff = diff.keys()

    with torch.no_grad():

        for name, param in model.named_parameters():

            if name in names_in_diff:

                param.add_(coeff * diff[name])





class TradesAWP:

    """Adversarial Weight Perturbation for TRADES loss."""

    def __init__(self, model, proxy, proxy_optim, gamma):

        self.model = model

        self.proxy = proxy

        self.proxy_optim = proxy_optim

        self.gamma = gamma



    def calc_awp(self, inputs_adv, inputs_clean, targets, beta):

        self.proxy.load_state_dict(self.model.state_dict())

        self.proxy.train()



        loss_natural = F.cross_entropy(self.proxy(inputs_clean), targets)

        loss_robust = F.kl_div(

            F.log_softmax(self.proxy(inputs_adv), dim=1),

            F.softmax(self.proxy(inputs_clean), dim=1),

            reduction='batchmean'

        )

        loss = -1.0 * (loss_natural + beta * loss_robust)



        self.proxy_optim.zero_grad()

        loss.backward()

        self.proxy_optim.step()



        diff = diff_in_weights(self.model, self.proxy)

        return diff



    def perturb(self, diff):

        add_into_weights(self.model, diff, coeff=1.0 * self.gamma)



    def restore(self, diff):

        add_into_weights(self.model, diff, coeff=-1.0 * self.gamma)





# Model

def build_model(arch, num_classes):

    if arch == 'resnet18':

        model = resnet18(weights=None)

    elif arch == 'resnet34':

        model = resnet34(weights=None)

    elif arch == 'resnet50':

        model = resnet50(weights=None)

    else:

        raise ValueError(f"Unknown architecture: {arch}")

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model





# PGD Attack

def pgd_attack(model, x, y, eps=EPS, step_size=STEP_SIZE, num_steps=NUM_STEPS):

    model.eval()

    x_adv = x.detach() + torch.empty_like(x).uniform_(-eps, eps)

    x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()



    for _ in range(num_steps):

        x_adv.requires_grad_(True)

        with torch.enable_grad():

            loss = F.cross_entropy(model(x_adv), y)

        grad = torch.autograd.grad(loss, x_adv)[0]

        x_adv = x_adv.detach() + step_size * grad.sign()

        delta = torch.clamp(x_adv - x, -eps, eps)

        x_adv = torch.clamp(x + delta, 0.0, 1.0).detach()



    return x_adv





# TRADES Inner Loop

def trades_perturb(model, x_natural, eps=EPS, step_size=STEP_SIZE, num_steps=NUM_STEPS):


    model.eval()

    with torch.no_grad():

        p_natural = F.softmax(model(x_natural), dim=1)



    x_adv = x_natural.detach() + 0.001 * torch.randn_like(x_natural)

    x_adv = torch.clamp(x_adv, 0.0, 1.0)



    for _ in range(num_steps):

        x_adv.requires_grad_(True)

        with torch.enable_grad():

            loss_kl = F.kl_div(

                F.log_softmax(model(x_adv), dim=1),

                p_natural,

                reduction='sum'

            )

        grad = torch.autograd.grad(loss_kl, x_adv)[0]

        x_adv = x_adv.detach() + step_size * grad.sign()

        delta = torch.clamp(x_adv - x_natural, -eps, eps)

        x_adv = torch.clamp(x_natural + delta, 0.0, 1.0).detach()



    return x_adv





# Phase 1: Standard PGD-AT Loss

def pgd_at_loss(model, x, y, eps=EPS, step_size=STEP_SIZE, num_steps=NUM_STEPS):

    x_adv = pgd_attack(model, x, y, eps, step_size, num_steps)



    model.train()

    logits_clean = model(x)

    logits_adv = model(x_adv)



    loss_clean = F.cross_entropy(logits_clean, y, label_smoothing=LABEL_SMOOTH)

    loss_adv = F.cross_entropy(logits_adv, y, label_smoothing=LABEL_SMOOTH)



    loss = 0.5 * loss_clean + 0.5 * loss_adv

    return loss, x_adv





# Phase 2+: TRADES Loss

def trades_loss(model, x, y, eps=EPS, step_size=STEP_SIZE, num_steps=NUM_STEPS, beta=BETA):

    x_adv = trades_perturb(model, x, eps, step_size, num_steps)



    model.train()

    logits_nat = model(x)

    logits_adv = model(x_adv)



    loss_natural = F.cross_entropy(logits_nat, y, label_smoothing=LABEL_SMOOTH)

    loss_robust = F.kl_div(

        F.log_softmax(logits_adv, dim=1),

        F.softmax(logits_nat, dim=1).detach(),

        reduction='batchmean'

    )



    loss = loss_natural + beta * loss_robust

    return loss, x_adv





# Evaluation helpers

@torch.no_grad()

def evaluate_clean(model, loader):

    model.eval()

    correct, total = 0, 0

    for imgs, lbls in loader:

        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)

        preds = model(imgs).argmax(1)

        correct += preds.eq(lbls).sum().item()

        total += lbls.size(0)

    return 100.0 * correct / total





def evaluate_robust(model, loader, eps=EPS, alpha=STEP_SIZE, iters=EVAL_STEPS):

    model.eval()

    correct, total = 0, 0

    for imgs, lbls in loader:

        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)

        x_adv = pgd_attack(model, imgs, lbls, eps, alpha, iters)

        with torch.no_grad():

            preds = model(x_adv).argmax(1)

        correct += preds.eq(lbls).sum().item()

        total += lbls.size(0)

    return 100.0 * correct / total





# Learning Rate Schedule

def get_lr(epoch):

    if epoch >= SWA_START:

        return SWA_LR

    elif epoch >= 150:

        return LR * 0.01

    elif epoch >= 100:

        return LR * 0.1

    else:

        return LR





# Main training script

def main():

    print("=" * 70)

    print("  Phased Adversarial Training")

    print("  Phase 1 (ep 0-9):    PGD-AT warmup       [STABLE]")

    print("  Phase 2 (ep 10-159): TRADES + AWP         [HIGH PERF]")

    print("  Phase 3 (ep 160-199): TRADES + AWP + SWA  [GENERALIZE]")

    print("=" * 70)



    # ---- Load Data ----

    print("\nLoading data...")

    data = np.load("train.npz")

    images = torch.from_numpy(data["images"]).float() / 255.0

    labels = torch.from_numpy(data["labels"]).long()

    print(f"Dataset: {len(images)} images, shape={images.shape[1:]}, "

          f"classes={labels.unique().numel()}")



    full_ds = TensorDataset(images, labels)

    train_size = int(0.9 * len(full_ds))

    val_size = len(full_ds) - train_size

    train_sub, val_sub = random_split(

        full_ds, [train_size, val_size],

        generator=torch.Generator().manual_seed(SEED)

    )



    train_transform = transforms.Compose([

        transforms.RandomCrop(32, padding=4),

        transforms.RandomHorizontalFlip(),

        Cutout(CUTOUT_SIZE),

    ])



    train_ds = AugmentedDataset(train_sub, transform=train_transform)

    val_ds = AugmentedDataset(val_sub, transform=None)



    train_loader = DataLoader(

        train_ds, batch_size=BATCH_SIZE, shuffle=True,

        num_workers=4, pin_memory=True, drop_last=True

    )

    val_loader = DataLoader(

        val_ds, batch_size=256, shuffle=False,

        num_workers=4, pin_memory=True

    )



    # ---- Models ----

    model = build_model(MODEL_ARCH, NUM_CLASSES).to(DEVICE)

    proxy = build_model(MODEL_ARCH, NUM_CLASSES).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model: {MODEL_ARCH} ({n_params / 1e6:.1f}M parameters)")



    # ---- Optimizers ----

    optimizer = optim.SGD(

        model.parameters(), lr=LR,

        momentum=MOMENTUM, weight_decay=WEIGHT_DECAY

    )

    proxy_optim = optim.SGD(proxy.parameters(), lr=0.01)

    awp_adversary = TradesAWP(model, proxy, proxy_optim, AWP_GAMMA)



    # ---- SWA ----

    swa_model = AveragedModel(model)

    swa_n = 0



    # ---- Training State ----

    best_score = 0.0

    best_epoch = 0

    nan_count = 0



    print(f"\nStarting training for {EPOCHS} epochs...")

    print("-" * 70)



    for epoch in range(EPOCHS):

        # ---- Determine Phase ----

        use_trades = (epoch >= AWP_WARMUP)

        use_awp = (epoch >= AWP_WARMUP)

        use_swa = (epoch >= SWA_START)



        phase_str = "Phase1:PGD-AT" if not use_trades else (

            "Phase3:TRADES+AWP+SWA" if use_swa else "Phase2:TRADES+AWP"

        )



        # ---- Set Learning Rate ----

        lr = get_lr(epoch)

        for pg in optimizer.param_groups:

            pg['lr'] = lr



        # ---- Train ----

        model.train()

        running_loss = 0.0

        correct, total = 0, 0

        batch_count = 0

        skipped = 0



        for imgs, lbls in train_loader:

            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)



            # === Compute loss + adversarial examples ===

            if use_trades:

                loss, x_adv = trades_loss(model, imgs, lbls)

            else:

                loss, x_adv = pgd_at_loss(model, imgs, lbls)



            # === NaN safety ===

            if torch.isnan(loss) or torch.isinf(loss) or loss.item() > MAX_LOSS:

                optimizer.zero_grad()

                skipped += 1

                nan_count += 1

                if nan_count > 50:

                    print(f"\n[FATAL] Too many NaN batches ({nan_count}). Stopping.")

                    return

                continue



            # === AWP: perturb weights, recompute loss ===

            if use_awp:

                awp = awp_adversary.calc_awp(

                    inputs_adv=x_adv, inputs_clean=imgs,

                    targets=lbls, beta=BETA

                )

                awp_adversary.perturb(awp)



                # Recompute loss with perturbed weights

                model.train()

                logits_nat = model(imgs)

                logits_adv = model(x_adv)

                loss_natural = F.cross_entropy(

                    logits_nat, lbls, label_smoothing=LABEL_SMOOTH

                )

                loss_robust = F.kl_div(

                    F.log_softmax(logits_adv, dim=1),

                    F.softmax(logits_nat, dim=1).detach(),

                    reduction='batchmean'

                )

                loss = loss_natural + BETA * loss_robust



            # === Backward + step ===

            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)

            optimizer.step()



            # === Restore AWP ===

            if use_awp:

                awp_adversary.restore(awp)



            running_loss += loss.item()

            batch_count += 1

            with torch.no_grad():

                preds = model(imgs).argmax(1)

                correct += preds.eq(lbls).sum().item()

                total += lbls.size(0)



        # ---- SWA Update ----

        if use_swa:

            swa_model.update_parameters(model)

            swa_n += 1



        # ---- Logging ----

        train_acc = 100.0 * correct / max(total, 1)

        avg_loss = running_loss / max(batch_count, 1)

        skip_str = f" skip={skipped}" if skipped > 0 else ""

        print(f"Epoch [{epoch + 1:03d}/{EPOCHS}] {phase_str} lr={lr:.4f} "

              f"loss={avg_loss:.4f} acc={train_acc:.2f}%{skip_str}")



        # ---- Evaluation (every 5 epochs + last epoch) ----

        if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:

            clean_acc = evaluate_clean(model, val_loader)

            robust_acc = evaluate_robust(model, val_loader)

            score = 0.5 * clean_acc + 0.5 * robust_acc



            flag = ""

            if score > best_score:

                best_score = score

                best_epoch = epoch + 1

                torch.save(model.state_dict(), "model_best.pt")

                flag = " <-- BEST"



            print(f"  -> Val: clean={clean_acc:.2f}% robust={robust_acc:.2f}% "

                  f"score={score:.2f}%{flag}")



        # ---- SWA evaluation (every 10 epochs after SWA starts) ----

        if use_swa and (epoch + 1) % 10 == 0:

            swa_copy = copy.deepcopy(swa_model)

            try:

                update_bn(train_loader, swa_copy, device=DEVICE)

                swa_clean = evaluate_clean(swa_copy, val_loader)

                swa_robust = evaluate_robust(swa_copy, val_loader)

                swa_score = 0.5 * swa_clean + 0.5 * swa_robust

                print(f"  -> SWA: clean={swa_clean:.2f}% robust={swa_robust:.2f}% "

                      f"score={swa_score:.2f}%")

                if swa_score > best_score:

                    best_score = swa_score

                    best_epoch = epoch + 1

                    torch.save(swa_copy.module.state_dict(), "model_best.pt")

                    print(f"  -> SWA saved as BEST!")

            except Exception as e:

                print(f"  -> SWA BN update failed: {e}")

            del swa_copy



    # ---- Finalize SWA ----

    print("\n" + "=" * 70)

    if swa_n > 0:

        print(f"Finalizing SWA model (averaged {swa_n} checkpoints)...")

        try:

            update_bn(train_loader, swa_model, device=DEVICE)

            swa_clean = evaluate_clean(swa_model, val_loader)

            swa_robust = evaluate_robust(swa_model, val_loader)

            swa_score = 0.5 * swa_clean + 0.5 * swa_robust

            print(f"Final SWA: clean={swa_clean:.2f}% robust={swa_robust:.2f}% "

                  f"score={swa_score:.2f}%")

            if swa_score > best_score:

                best_score = swa_score

                best_epoch = EPOCHS

                torch.save(swa_model.module.state_dict(), "model_best.pt")

                print("SWA is the BEST! Saved.")

        except Exception as e:

            print(f"SWA finalization failed: {e}")



    # ---- Save ----

    print(f"\nBest val score: {best_score:.2f}% (epoch {best_epoch})")

    model.load_state_dict(torch.load("model_best.pt", map_location=DEVICE))

    model.eval()



    with torch.no_grad():

        dummy = torch.randn(1, 3, 32, 32).to(DEVICE)

        out = model(dummy)

        assert out.shape == (1, NUM_CLASSES), f"Bad shape: {out.shape}"



    torch.save(model.state_dict(), "model.pt")

    print(f"model.pt saved. Ready for submission with MODEL_NAME='{MODEL_ARCH}'")





if __name__ == "__main__":

    main()
