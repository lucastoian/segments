#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sign_contrastive.py — Addestramento & ricerca per segni incisi (agnostico alla forma)
- TRAIN: dual encoder (reperto/segno) con loss contrastiva e "local-max" per localizzazione.
- HEATMAP: visualizza dove un dato segno è rilevato in una foto.
- EMBED: produce indici (npz) per segni e per immagini/ritagli.
- SEARCH: restituisce i top-k reperti per un segno o viceversa.

CSV atteso (impostabile):
    --csv-order sign_photo  →  path_segno,path_reperto   (TUO CASO)
    --csv-order photo_sign  →  path_reperto,path_segno

Le path possono essere file locali o URL (http/https). Gli URL vengono scaricati in cache.

Debug:
  --debug-dir DIR          salva immagini/heatmap/matrice logits durante il training
  --log-every N            salva debug ogni N batch
  --max-debug-samples K    quante coppie (i) del batch salvare per volta
"""

import os, csv, argparse, hashlib
from typing import List, Tuple

import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T

# opzionale per la ricerca FAISS
try:
    import faiss
    _FAISS=True
except Exception:
    _FAISS=False

# download URL -> cache
try:
    import requests
    _HAS_REQUESTS = True
except Exception:
    _HAS_REQUESTS = False


# ------------------------ Utils ------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def ensure_dir(p): 
    if p and p.strip():
        os.makedirs(p, exist_ok=True)

def _is_url(p: str) -> bool:
    p = (p or "").lower()
    return p.startswith("http://") or p.startswith("https://")

def _url_to_cache_path(url: str, cache_dir: str) -> str:
    ensure_dir(cache_dir)
    h = hashlib.md5(url.encode("utf-8")).hexdigest()[:10]
    base = os.path.basename(url.split("?")[0]) or "file"
    return os.path.join(cache_dir, f"{h}_{base}")

def _download_to_cache(url: str, cache_dir: str, timeout: int = 12) -> str:
    """
    Scarica se serve e ritorna il path locale. Lancia eccezione se fallisce.
    """
    if not _HAS_REQUESTS:
        raise RuntimeError("Per scaricare URL remoti serve 'requests' (pip install requests).")
    dst = _url_to_cache_path(url, cache_dir)
    if os.path.isfile(dst) and os.path.getsize(dst) > 0:
        return dst
    r = requests.get(url, stream=True, timeout=timeout)
    r.raise_for_status()
    with open(dst, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk: f.write(chunk)
    return dst

def path_or_url_to_local(p: str, cache_dir: str, timeout: int = 12) -> str:
    """
    Ritorna un path locale (identico se è file, altrimenti scarica l'URL in cache).
    Lancia eccezione se non è accessibile.
    """
    if _is_url(p):
        return _download_to_cache(p, cache_dir, timeout)
    if not os.path.isfile(p):
        raise FileNotFoundError(p)
    return p

def load_rgb_any(path_or_url: str, cache_dir: str, timeout: int = 12, force_3ch=True) -> Image.Image:
    p = path_or_url_to_local(path_or_url, cache_dir, timeout)
    img = Image.open(p)
    return img.convert("RGB") if force_3ch else img


# ---------- DEBUG UTILS (salvataggi immagine/heatmap/logits) ----------

def _denorm_batch(x: torch.Tensor) -> torch.Tensor:
    """x: [B,3,H,W], float tensor normalizzato → uint8 denormalizzato."""
    mean = torch.tensor(IMAGENET_MEAN, device=x.device)[None,:,None,None]
    std  = torch.tensor(IMAGENET_STD,  device=x.device)[None,:,None,None]
    y = x * std + mean
    y = (y.clamp(0,1) * 255).byte()
    return y  # [B,3,H,W] uint8

def _to_np_img(t: torch.Tensor) -> np.ndarray:
    """t: [3,H,W] uint8 torch → HWC RGB uint8 np."""
    return t.permute(1,2,0).detach().cpu().numpy()

def _save_heatmap_overlay(rgb_img_u8: np.ndarray, sim_map_01: np.ndarray, out_path: str):
    """rgb_img_u8: HWC RGB uint8, sim_map_01: [h,w] float 0..1"""
    ensure_dir(os.path.dirname(out_path))
    H,W = rgb_img_u8.shape[:2]
    sim01 = (sim_map_01 - sim_map_01.min()) / (sim_map_01.max()-sim_map_01.min() + 1e-8)
    heat = cv2.applyColorMap((sim01*255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.resize(heat, (W, H), interpolation=cv2.INTER_CUBIC)
    bgr = rgb_img_u8[..., ::-1]  # to BGR
    overlay = cv2.addWeighted(bgr, 0.6, heat, 0.4, 0)
    cv2.imwrite(out_path, overlay)

def _save_logits_image(logits_np: np.ndarray, out_path: str):
    ensure_dir(os.path.dirname(out_path))
    L = logits_np.copy()
    if np.isfinite(L).all():
        L = (L - L.min()) / (L.max() - L.min() + 1e-8)
    else:
        L = np.nan_to_num(L, nan=0.0)
        L = (L - L.min()) / (L.max() - L.min() + 1e-8)
    img = cv2.applyColorMap((L*255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    img = cv2.resize(img, (img.shape[1]*20, img.shape[0]*20), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(out_path, img)


# ---------------------- Dataset ------------------------

class PairDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, img_size=256, center_crop=224,
                 csv_order="sign_photo",  # <-- il tuo CSV
                 cache_dir=os.path.expanduser("~/.cache/sign_dl"),
                 aug=True, timeout=12):
        """
        csv_order:
          - 'sign_photo'  : prima colonna = segno, seconda = reperto (TUO CSV)
          - 'photo_sign'  : prima colonna = reperto, seconda = segno
        """
        self.cache_dir = cache_dir
        self.timeout = timeout
        self.csv_order = csv_order

        raw_pairs=[]
        with open(csv_path, newline="") as f:
            r=csv.reader(f)
            for row in r:
                if not row: continue
                a=row[0].strip(); b=row[1].strip()
                if not a or not b: continue
                if csv_order=="sign_photo":
                    sign, photo = a, b
                else:
                    photo, sign = a, b
                raw_pairs.append((photo, sign))

        # deduplica e mantieni l'ordine
        seen=set(); pairs=[]
        for p in raw_pairs:
            if p not in seen:
                seen.add(p); pairs.append(p)
        self.pairs = pairs
        if len(self.pairs)==0:
            raise RuntimeError(f"Nessuna riga valida in {csv_path}")

        # Trasformazioni
        self.t_photo = T.Compose([
            T.Resize(img_size, antialias=True),
            T.RandomResizedCrop(center_crop, scale=(0.6,1.0), ratio=(0.8,1.25), antialias=True) if aug else T.CenterCrop(center_crop),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02) if aug else T.Lambda(lambda x:x),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        self.t_sign = T.Compose([
            T.Resize((center_crop, center_crop), antialias=True),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self): return len(self.pairs)

    def __getitem__(self, i):
        p_img, p_sign = self.pairs[i]
        try:
            img  = load_rgb_any(p_img,  self.cache_dir, self.timeout)   # foto del reperto
            sign = load_rgb_any(p_sign, self.cache_dir, self.timeout)   # immagine del segno
        except Exception:
            # se fallisce (404, file corrotto ecc.) segnalo al collate di scartare
            return None
        return self.t_photo(img), self.t_sign(sign), p_img, p_sign


def collate_drop_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch)==0:
        return None
    imgs, signs, p_img, p_sign = zip(*batch)
    return torch.stack(imgs), torch.stack(signs), list(p_img), list(p_sign)


# ---------------------- Modello ------------------------

class ConvEncoder(nn.Module):
    """ResNet18 backbone; può restituire mappa o vettore globale"""
    def __init__(self, out_dim=2048):
        super().__init__()
        base = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V2)
        #base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = base.layer1, base.layer2, base.layer3, base.layer4
        self.out_dim = 2048

    def forward(self, x, return_map=False):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)   # [B,512,H,W]
        if return_map:
            fmap = F.normalize(x, dim=1)
            return fmap
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        x = F.normalize(x, dim=1)
        return x

class DualEncoder(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.photo = ConvEncoder(2048)
        self.sign  = ConvEncoder(2048)
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1/temperature)))

    def compute_logits(self, img, sign, return_maps: bool = False):
        # feature map della foto
        fmap = self.photo(img, return_map=True)       # [B,512,H,W] L2-normalized
        B, C, H, W = fmap.shape
        fvecs = fmap.view(B, C, H*W).permute(0,2,1).contiguous()  # [B,HW,C]

        # embedding globale del segno
        svec = self.sign(sign, return_map=False)      # [B,512]
        svec = svec / (svec.norm(dim=1, keepdim=True) + 1e-8)

        # score_maps[i, h, b] = <fvecs[i,h,:], svec[b,:]>
        score_maps = torch.einsum("ihc,bc->ihb", fvecs, svec)  # [B,HW,B]
        logits = score_maps.max(dim=1).values                  # [B,B]

        scale = self.logit_scale.exp().clamp(1/100.0, 100.0)
        logits = logits * scale

        if return_maps:
            return logits, score_maps, (H, W)
        return logits

    def forward(self, img, sign):
        logits = self.compute_logits(img, sign)
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.t(), labels)
        return (loss_i + loss_t) * 0.5, logits


# ---------------------- Training ------------------------

def train(args):
    device = "cuda" if (args.cuda and torch.cuda.is_available()) else "cpu"

    ds = PairDataset(args.pairs, img_size=args.img_size, center_crop=args.crop,
                     csv_order=args.csv_order, cache_dir=args.cache_dir,
                     aug=not args.no_aug)
    # validation split (es. 10%)
    n_train = int(len(ds)*(1.0-args.val_split))

    # split deterministico
    tr, va = torch.utils.data.random_split(
        ds, [n_train, len(ds)-n_train], generator=torch.Generator().manual_seed(42)
    )

    # DataLoader
    dl_tr = torch.utils.data.DataLoader(
        tr, batch_size=args.bs, shuffle=True, num_workers=args.workers,
        pin_memory=True, drop_last=True, collate_fn=collate_drop_none
    )
    dl_va = torch.utils.data.DataLoader(
        va, batch_size=args.bs, shuffle=False, num_workers=args.workers,
        pin_memory=True, drop_last=False, collate_fn=collate_drop_none
    )

    model = DualEncoder(temperature=args.temperature).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # amp compat (nuove API se disponibili)
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=(device=="cuda"))
        autocast_ctx = lambda: torch.amp.autocast('cuda', enabled=(device=="cuda"))
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=(device=="cuda"))
        autocast_ctx = lambda: torch.cuda.amp.autocast(enabled=(device=="cuda"))

    best = {"epoch":-1, "val_acc":0.0}

    ensure_dir(args.out)
    for epoch in range(1, args.epochs+1):
        model.train()
        tot=0.0; nb=0
        for batch in tqdm(dl_tr, desc=f"Epoch {epoch}/{args.epochs}"):
            if batch is None:
                continue
            img, sign, *_ = batch
            img=img.to(device, non_blocking=True); sign=sign.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast_ctx():
                loss, _ = model(img, sign)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tot += loss.item(); nb += 1

            # ---- DEBUG SAVE ----
            if args.debug_dir and (nb % max(1, args.log_every) == 0):
                try:
                    ensure_dir(args.debug_dir)
                    model.eval()
                    with torch.no_grad():
                        logits_dbg, score_maps, (h, w) = model.compute_logits(img, sign, return_maps=True)
                    # denorm
                    img_dn  = _denorm_batch(img)   # [B,3,H,W] uint8
                    sign_dn = _denorm_batch(sign)  # [B,3,H,W] uint8

                    logits_np = logits_dbg.detach().cpu().numpy()
                    _save_logits_image(logits_np, os.path.join(args.debug_dir, f"epoch{epoch:03d}_step{nb:06d}_logits.png"))

                    K = min(args.max_debug_samples, img_dn.shape[0])
                    for i in range(K):
                        cv2.imwrite(
                            os.path.join(args.debug_dir, f"epoch{epoch:03d}_step{nb:06d}_img_{i}.png"),
                            _to_np_img(img_dn[i])[..., ::-1]
                        )
                        cv2.imwrite(
                            os.path.join(args.debug_dir, f"epoch{epoch:03d}_step{nb:06d}_sign_{i}.png"),
                            _to_np_img(sign_dn[i])[..., ::-1]
                        )
                        # POS: (i,i)
                        sim_pos = score_maps[i, :, i].view(h, w).detach().cpu().numpy()
                        _save_heatmap_overlay(_to_np_img(img_dn[i]), sim_pos,
                            os.path.join(args.debug_dir, f"epoch{epoch:03d}_step{nb:06d}_i{i}_POS.png"))

                        # NEG più duro: j = argmax logits[i,j], j != i
                        row = logits_np[i]
                        j_hard = int(np.argmax(np.where(np.arange(row.size)==i, -1e9, row)))
                        sim_neg = score_maps[i, :, j_hard].view(h, w).detach().cpu().numpy()
                        _save_heatmap_overlay(_to_np_img(img_dn[i]), sim_neg,
                            os.path.join(args.debug_dir, f"epoch{epoch:03d}_step{nb:06d}_i{i}_NEG_j{j_hard}.png"))

                    # salva valori grezzi
                    with open(os.path.join(args.debug_dir, f"epoch{epoch:03d}_step{nb:06d}_logits.txt"), "w") as f:
                        for r in logits_np:
                            f.write(" ".join(f"{v:7.3f}" for v in r) + "\n")

                except Exception as e:
                    print(f"[debug-save] warning: {e}")
                finally:
                    model.train()

        avg_tr = tot/max(1,nb)

        # Validazione
        model.eval(); corr=0; totv=0
        with torch.no_grad():
            for batch in dl_va:
                if batch is None: continue
                img, sign, *_ = batch
                img=img.to(device); sign=sign.to(device)
                logits = model.compute_logits(img, sign)
                pred = logits.argmax(dim=1)
                corr += (pred == torch.arange(pred.shape[0], device=device)).sum().item()
                totv += pred.shape[0]
        acc = corr/max(1,totv)

        print(f"Epoch {epoch}: train_loss={avg_tr:.4f}  val_acc={acc:.3f}")
        torch.save({"epoch":epoch, "model":model.state_dict()}, os.path.join(args.out, "last.pth"))
        if acc > best["val_acc"]:
            best={"epoch":epoch, "val_acc":acc}
            torch.save({"epoch":epoch, "model":model.state_dict()}, os.path.join(args.out, "best.pth"))
    print(f"✓ Best val_acc={best['val_acc']:.3f} @ epoch {best['epoch']}  → {os.path.join(args.out,'best.pth')}")


# ---------------------- Heatmap ------------------------

def heatmap(args):
    device = "cuda" if (args.cuda and torch.cuda.is_available()) else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device)
    model = DualEncoder(); model.load_state_dict(ckpt["model"], strict=True); model.to(device).eval()

    t_img  = T.Compose([T.Resize(args.img_size, antialias=True), T.CenterCrop(args.crop),
                        T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    t_sign = T.Compose([T.Resize((args.crop,args.crop), antialias=True), T.ToTensor(),
                        T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    pil_img  = load_rgb_any(args.photo, args.cache_dir)
    pil_sign = load_rgb_any(args.sign,  args.cache_dir)
    img  = t_img(pil_img).unsqueeze(0).to(device)
    sign = t_sign(pil_sign).unsqueeze(0).to(device)

    with torch.no_grad():
        fmap = model.photo(img, return_map=True)[0]           # [512,H,W]
        svec = model.sign(sign, return_map=False)[0]          # [512]
        H,W = fmap.shape[-2], fmap.shape[-1]
        fmap_flat = fmap.view(2048, H*W).t()                   # [HW,512]
        sim = (fmap_flat @ svec).view(H,W).detach().cpu().numpy()
        sim = (sim - sim.min()) / (sim.max()-sim.min() + 1e-8)

    overlay = np.array(T.Compose([T.Resize(args.img_size, antialias=True), T.CenterCrop(args.crop)])(pil_img))
    heat = cv2.applyColorMap((sim*255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.resize(heat, (overlay.shape[1], overlay.shape[0]), interpolation=cv2.INTER_CUBIC)
    out = cv2.addWeighted(overlay[...,::-1], 0.6, heat, 0.4, 0)
    ensure_dir(os.path.dirname(args.out) or ".")
    cv2.imwrite(args.out, out)
    print(f"✓ Heatmap salvata in {args.out}")


# ---------------------- Embed & Search ------------------------

def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = DualEncoder(); model.load_state_dict(ckpt["model"], strict=True); model.to(device).eval()
    return model

@torch.no_grad()
def embed_folder_signs(model, folder, device, out_npz, size=224):
    tr = T.Compose([T.Resize((size,size), antialias=True), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    feats=[]; names=[]
    for fn in sorted(os.listdir(folder)):
        if not fn.lower().endswith((".jpg",".jpeg",".png",".tif",".tiff")): continue
        img=tr(Image.open(os.path.join(folder,fn)).convert("RGB")).unsqueeze(0).to(device)
        vec = model.sign(img).cpu().numpy()[0].astype("float32")
        feats.append(vec); names.append(fn)
    feats=np.vstack(feats).astype("float32")
    np.savez(out_npz, names=np.array(names), feats=feats)
    print(f"✓ Sign embeddings: {feats.shape} → {out_npz}")

@torch.no_grad()
def embed_folder_photos(model, folder, device, out_npz, size=256, crop=224):
    tr = T.Compose([T.Resize(size, antialias=True), T.CenterCrop(crop),
                    T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    feats=[]; names=[]
    for fn in sorted(os.listdir(folder)):
        if not fn.lower().endswith((".jpg",".jpeg",".png",".tif",".tiff")): continue
        img=tr(Image.open(os.path.join(folder,fn)).convert("RGB")).unsqueeze(0).to(device)
        vec = model.photo(img) .cpu().numpy()[0].astype("float32")
        feats.append(vec); names.append(fn)
    feats=np.vstack(feats).astype("float32")
    np.savez(out_npz, names=np.array(names), feats=feats)
    print(f"✓ Photo embeddings: {feats.shape} → {out_npz}")

def search(args):
    S=np.load(args.signs_npz); P=np.load(args.photos_npz)
    s_names=S["names"]; s_feats=S["feats"].astype("float32")
    p_names=P["names"]; p_feats=P["feats"].astype("float32")

    if _FAISS:
        index = faiss.IndexFlatIP(s_feats.shape[1])
        faiss.normalize_L2(p_feats); faiss.normalize_L2(s_feats)
        index.add(p_feats)
        idx = np.where(s_names==args.query_sign)[0]
        if idx.size==0: raise SystemExit("Segno non trovato nell'npz")
        q = s_feats[idx[0]][None,:]
        D,I=index.search(q, args.topk)
        print(f"Top-{args.topk} reperti per segno {args.query_sign}:")
        for d,i in zip(D[0],I[0]):
            print(f"  {p_names[i]}  score={float(d):.3f}")
    else:
        p = p_feats / np.linalg.norm(p_feats,axis=1,keepdims=True)
        s = s_feats / np.linalg.norm(s_feats,axis=1,keepdims=True)
        idx = np.where(s_names==args.query_sign)[0]
        if idx.size==0: raise SystemExit("Segno non trovato nell'npz")
        q = s[idx[0]]
        sims = (p @ q)
        order = np.argsort(-sims)[:args.topk]
        print(f"Top-{args.topk} reperti per segno {args.query_sign}:")
        for i in order:
            print(f"  {p_names[i]}  score={float(sims[i]):.3f}")


def search_signs(args):
    """
    Cerca segni simili nel solo spazio dei segni.
    Richiede un file .npz generato da embed_folder_signs, con:
        - names: nomi file dei segni
        - feats: embedding (float32, già normalizzati)
    """
    S = np.load(args.signs_npz)
    s_names = S["names"]
    s_feats = S["feats"].astype("float32")

    # in caso di tipi strani, normalizziamo i nomi a stringa
    s_names = np.array(s_names).astype(str)

    # indice del segno di query
    idx = np.where(s_names == args.query_sign)[0]
    if idx.size == 0:
        raise SystemExit(f"Segno '{args.query_sign}' non trovato in {args.signs_npz}")
    q_idx = int(idx[0])

    if _FAISS:
        # indicizzazione FAISS in spazio coseno (IP su vettori normalizzati)
        dim = s_feats.shape[1]
        index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(s_feats)          # in-place
        index.add(s_feats)                   # tutti i segni

        q = s_feats[q_idx][None, :]          # [1,dim]
        D, I = index.search(q, args.topk + 1)  # +1 per includere il segno stesso

        print(f"Top-{args.topk} segni simili a {s_names[q_idx]}:")
        shown = 0
        for d, i in zip(D[0], I[0]):
            name = s_names[i]
            if name == s_names[q_idx]:
                # saltiamo il segno identico (se vuoi tenerlo, togli questo if)
                continue
            print(f"  {name}  score={float(d):.3f}")
            shown += 1
            if shown >= args.topk:
                break
    else:
        # fallback senza FAISS: prodotto scalare su vettori normalizzati
        s_norm = s_feats / (np.linalg.norm(s_feats, axis=1, keepdims=True) + 1e-8)
        q = s_norm[q_idx]
        sims = (s_norm @ q)                  # [N]

        # ordina per similarità decrescente
        order = np.argsort(-sims)
        print(f"Top-{args.topk} segni simili a {s_names[q_idx]}:")
        shown = 0
        for i in order:
            if i == q_idx:
                continue      # salta il segno stesso
            print(f"  {s_names[i]}  score={float(sims[i]):.3f}")
            shown += 1
            if shown >= args.topk:
                break



# ---------------------- CLI ------------------------

def parse_args():
    p=argparse.ArgumentParser("Training/ricerca per segni incisi — dual-encoder contrastivo")
    sub=p.add_subparsers(dest="cmd", required=True)

    pt=sub.add_parser("train")
    pt.add_argument("--pairs", required=True, help="CSV: path_segno,path_reperto (o viceversa con --csv-order)")
    pt.add_argument("--csv-order", choices=["sign_photo","photo_sign"], default="sign_photo",
                    help="Ordine colonne nel CSV (default: sign_photo = prima segno, poi reperto)")
    pt.add_argument("--cache-dir", default=os.path.expanduser("~/.cache/sign_dl"))
    pt.add_argument("--out", default="runs/exp1")
    pt.add_argument("--epochs", type=int, default=30)
    pt.add_argument("--bs", type=int, default=16)
    pt.add_argument("--lr", type=float, default=1e-4)
    pt.add_argument("--temperature", type=float, default=0.07)
    pt.add_argument("--val-split", type=float, default=0.1)
    pt.add_argument("--img-size", type=int, default=256)
    pt.add_argument("--crop", type=int, default=224)
    pt.add_argument("--no-aug", action="store_true")
    pt.add_argument("--workers", type=int, default=4)
    pt.add_argument("--cuda", action="store_true")
    # debug options
    pt.add_argument("--debug-dir", default=None, help="Se impostato, salva immagini/heatmap di debug qui")
    pt.add_argument("--log-every", type=int, default=200, help="Salva debug ogni N batch")
    pt.add_argument("--max-debug-samples", type=int, default=3, help="Quanti esempi salvare per batch di debug")

    ph=sub.add_parser("heatmap")
    ph.add_argument("--ckpt", required=True)
    ph.add_argument("--photo", required=True)
    ph.add_argument("--sign", required=True)
    ph.add_argument("--cache-dir", default=os.path.expanduser("~/.cache/sign_dl"))
    ph.add_argument("--out", default="heatmap.png")
    ph.add_argument("--img-size", type=int, default=256)
    ph.add_argument("--crop", type=int, default=224)
    ph.add_argument("--cuda", action="store_true")

    pe=sub.add_parser("embed")
    pe.add_argument("--ckpt", required=True)
    pe.add_argument("--signs-dir", required=True)
    pe.add_argument("--photos-dir", required=True)
    pe.add_argument("--out-signs", default="signs.npz")
    pe.add_argument("--out-photos", default="photos.npz")
    pe.add_argument("--cuda", action="store_true")

    ps=sub.add_parser("search")
    ps.add_argument("--signs-npz", required=True)
    ps.add_argument("--photos-npz", required=True)
    ps.add_argument("--query-sign", required=True, help="nome file del segno presente nell'npz (es. 998.png)")
    ps.add_argument("--topk", type=int, default=20)
    ps2 = sub.add_parser("search-signs")
    ps2.add_argument(
        "--signs-npz", required=True,
        help="File npz con embedding dei segni (da embed_folder_signs)"
    )
    ps2.add_argument(
        "--query-sign", required=True,
        help="Nome file del segno presente nell'npz (es. 998.jpg)"
    )
    ps2.add_argument(
        "--topk", type=int, default=20,
        help="Quanti segni simili restituire"
    )


    return p.parse_args()


if __name__=="__main__":
    args=parse_args()
    if args.cmd=="train":
        train(args)
    elif args.cmd=="heatmap":
        heatmap(args)
    elif args.cmd=="embed":
        device = "cuda" if (args.cuda and torch.cuda.is_available()) else "cpu"
        model = load_model(args.ckpt, device)
        embed_folder_signs(model, args.signs_dir, device, args.out_signs)
        embed_folder_photos(model, args.photos_dir, device, args.out_photos)
    elif args.cmd=="search":
        search(args)
    elif args.cmd=="search-signs":
        search_signs(args)
    else:
        search(args)
