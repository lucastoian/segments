from typing import Optional, List
import io

import numpy as np
import requests
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from PIL import Image
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T


# ================== CONFIG ==================

# ================== CONFIG ==================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SIGNS_NPZ  = os.path.join(BASE_DIR, "signs.npz")
PHOTOS_NPZ = os.path.join(BASE_DIR, "photos.npz")
CKPT_PATH  = os.path.join(BASE_DIR, "best.pth")

# URL pubbliche (consigliato: metterle come env var su Render)
CKPT_URL   = os.environ.get("CKPT_URL", "")
SIGNS_URL  = os.environ.get("SIGNS_URL", "")
PHOTOS_URL = os.environ.get("PHOTOS_URL", "")

def _download_if_missing(dst_path: str, url: str, timeout: int = 60):
    """
    Scarica url -> dst_path se il file non esiste.
    Usa streaming per non caricare tutto in RAM.
    """
    if os.path.exists(dst_path) and os.path.getsize(dst_path) > 0:
        return

    if not url:
        raise RuntimeError(f"Manca {os.path.basename(dst_path)} e la URL non è impostata (env var).")

    tmp_path = dst_path + ".tmp"
    print(f"[bootstrap] Download {url} -> {dst_path}")

    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):  # 1MB
                if chunk:
                    f.write(chunk)

    os.replace(tmp_path, dst_path)

_download_if_missing(CKPT_PATH, CKPT_URL, timeout=300)     # best.pth è grande
_download_if_missing(SIGNS_NPZ, SIGNS_URL, timeout=120)
_download_if_missing(PHOTOS_NPZ, PHOTOS_URL, timeout=120)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)




class ConvEncoder(nn.Module):
    """ResNet152 backbone; può restituire mappa o vettore globale"""
    def __init__(self, out_dim=2048):
        super().__init__()
        base = models.resnet152(weights=None)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = (
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
        )
        self.out_dim = 2048


    def forward(self, x, return_map: bool = False):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if return_map:
            fmap = F.normalize(x, dim=1)
            return fmap
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        x = F.normalize(x, dim=1)
        return x


class DualEncoder(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.photo = ConvEncoder(2048)
        self.sign  = ConvEncoder(2048)
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / temperature)))

    def forward(self, img, sign):
        raise NotImplementedError


device = "cuda" if torch.cuda.is_available() else "cpu"

model = DualEncoder()
ckpt = torch.load(CKPT_PATH, map_location=device)
model.load_state_dict(ckpt["model"], strict=True)
model.to(device).eval()

t_sign = T.Compose([
    T.Resize((224, 224), antialias=True),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

t_photo = T.Compose([
    T.Resize(256, antialias=True),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def _embed_sign_pil(img: Image.Image) -> np.ndarray:
    if img.mode != "RGB":
        img = img.convert("RGB")
    x = t_sign(img).unsqueeze(0).to(device)
    with torch.no_grad():
        vec = model.sign(x)[0]
        vec = vec / (vec.norm() + 1e-8)
    return vec.detach().cpu().numpy().astype("float32")


def _embed_photo_pil(img: Image.Image) -> np.ndarray:
    if img.mode != "RGB":
        img = img.convert("RGB")
    x = t_photo(img).unsqueeze(0).to(device)
    with torch.no_grad():
        vec = model.photo(x)[0]
        vec = vec / (vec.norm() + 1e-8)
    return vec.detach().cpu().numpy().astype("float32")


def _embed_from_url(url: str, is_sign: bool) -> np.ndarray:
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Errore nel download dell'immagine: {e}")
    try:
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Immagine non valida: {e}")
    if is_sign:
        return _embed_sign_pil(img)
    else:
        return _embed_photo_pil(img)



S = np.load(SIGNS_NPZ)
sign_names = S["names"].astype(str)
sign_feats = S["feats"].astype("float32")

P = np.load(PHOTOS_NPZ)
photo_names = P["names"].astype(str)
photo_feats = P["feats"].astype("float32")

sign_feats /= (np.linalg.norm(sign_feats, axis=1, keepdims=True) + 1e-8)
photo_feats /= (np.linalg.norm(photo_feats, axis=1, keepdims=True) + 1e-8)



class ItemScore(BaseModel):
    name: str
    score: float


class SearchAllResponse(BaseModel):
    query: str
    query_type: str   # "filename" o "url"
    similar_signs: List[ItemScore]
    similar_photos: List[ItemScore]


class SearchPhotoResponse(BaseModel):
    query: str
    query_type: str   # "photo_filename" o "photo_url"
    similar_signs: List[ItemScore]


def find_sign_index(name: str) -> int:
    idx = np.where(sign_names == name)[0]
    if idx.size == 0:
        raise HTTPException(
            status_code=404,
            detail=f"Segno '{name}' non trovato in {SIGNS_NPZ}",
        )
    return int(idx[0])


def find_photo_index(name: str) -> int:
    idx = np.where(photo_names == name)[0]
    if idx.size == 0:
        raise HTTPException(
            status_code=404,
            detail=f"Foto '{name}' non trovata in {PHOTOS_NPZ}",
        )
    return int(idx[0])


def similar_signs_from_vec(q_vec: np.ndarray, topk: int,
                           exclude_name: Optional[str] = None) -> List[ItemScore]:
    sims = sign_feats @ q_vec
    order = np.argsort(-sims)
    out: List[ItemScore] = []
    for i in order:
        name = sign_names[i]
        if exclude_name is not None and name == exclude_name:
            continue
        out.append(ItemScore(name=name, score=float(sims[i])))
        if len(out) >= topk:
            break
    return out


def similar_photos_from_vec(q_vec: np.ndarray, topk: int) -> List[ItemScore]:
    sims = photo_feats @ q_vec
    order = np.argsort(-sims)[:topk]
    return [ItemScore(name=photo_names[i], score=float(sims[i])) for i in order]



app = FastAPI(title="Etruscan Signs Search API")



@app.get("/search", response_model=SearchAllResponse)
def search(
    query_sign: Optional[str] = Query(
        None,
        description="Nome file del segno presente in signs.npz (es. '998.jpg')"
    ),
    sign_url: Optional[str] = Query(
        None,
        description="URL di un'immagine di segno (può anche non essere nel dataset)"
    ),
    topk_signs: int = Query(10, ge=1, le=100),
    topk_photos: int = Query(10, ge=1, le=100),
):
    """
    Ricerca segni e reperti simili a un SEGNO.

    - query_sign=998.jpg
    - oppure sign_url=https://.../segno.jpg
    """
    if not query_sign and not sign_url:
        raise HTTPException(
            status_code=400,
            detail="Devi specificare almeno uno tra 'query_sign' e 'sign_url'.",
        )
    if query_sign and sign_url:
        raise HTTPException(
            status_code=400,
            detail="Specifica solo 'query_sign' oppure solo 'sign_url', non entrambi.",
        )

    if sign_url is not None:
        q_vec = _embed_from_url(sign_url, is_sign=True)
        q_type = "url"
        q_str = sign_url
        exclude_name = None
    else:
        q_idx = find_sign_index(query_sign)
        q_vec = sign_feats[q_idx]
        q_type = "filename"
        q_str = query_sign
        exclude_name = query_sign

    return SearchAllResponse(
        query=q_str,
        query_type=q_type,
        similar_signs=similar_signs_from_vec(q_vec, topk_signs, exclude_name=exclude_name),
        similar_photos=similar_photos_from_vec(q_vec, topk_photos),
    )



@app.get("/search_photo", response_model=SearchPhotoResponse)
def search_photo(
    query_photo: Optional[str] = Query(
        None,
        description="Nome file della foto presente in photos.npz (es. '1200.jpg')"
    ),
    photo_url: Optional[str] = Query(
        None,
        description="URL di una foto di reperto"
    ),
    topk_signs: int = Query(10, ge=1, le=100),
):
    """
    Ricerca SOLO segni compatibili con una FOTO di reperto.

    Puoi usare:
      - query_photo=1200.jpg  (nome presente in photos.npz)
      - oppure photo_url=https://.../reperto.jpg
    """
    if not query_photo and not photo_url:
        raise HTTPException(
            status_code=400,
            detail="Devi specificare almeno uno tra 'query_photo' e 'photo_url'.",
        )
    if query_photo and photo_url:
        raise HTTPException(
            status_code=400,
            detail="Specifica solo 'query_photo' oppure solo 'photo_url', non entrambi.",
        )

    if photo_url is not None:
        # embedding calcolato al volo dalla foto URL
        q_vec = _embed_from_url(photo_url, is_sign=False)
        q_type = "photo_url"
        q_str = photo_url
    else:
        # embedding precalcolato della foto nel dataset
        q_idx = find_photo_index(query_photo)
        q_vec = photo_feats[q_idx]
        q_type = "photo_filename"
        q_str = query_photo

    return SearchPhotoResponse(
        query=q_str,
        query_type=q_type,
        similar_signs=similar_signs_from_vec(q_vec, topk_signs),
    )
