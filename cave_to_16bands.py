#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, glob
from pathlib import Path
import numpy as np
import tifffile as tiff
from scipy.io import loadmat

def _from_mat(md):
    for k,v in md.items():
        if k.startswith('__'): continue
        if isinstance(v, np.ndarray) and v.ndim==3 and 31 in v.shape:
            arr=v; break
    else:
        raise ValueError("Aucune variable 3D (31 bandes) trouvée dans le .mat")
    if arr.shape[0]==31 and arr.shape[1]==arr.shape[2]:
        arr=np.transpose(arr,(1,2,0))     # (C,H,W)->(H,W,C)
    elif arr.shape[-1]!=31:
        raise ValueError(f"Forme inattendue: {arr.shape}")
    arr=arr.astype(np.float32)
    vmax=arr.max() if arr.size else 1.0
    if vmax>1.5: arr/=65535.0 if vmax>255.0 else 255.0
    return arr

def load_cube_any(path):
    p=Path(path)
    if p.suffix.lower()=='.mat':
        return _from_mat(loadmat(p.as_posix()))
    elif p.suffix.lower() in ('.tif','.tiff'):
        arr=tiff.imread(p.as_posix())
        if arr.ndim==3 and arr.shape[0]==31:
            arr=np.transpose(arr,(1,2,0))
        arr=arr.astype(np.float32)
        vmax=arr.max() if arr.size else 1.0
        if vmax>1.5: arr/=65535.0 if vmax>255.0 else 255.0
        return arr
    else:
        raise ValueError(f"Extension non supportée: {p.suffix}")

def pick_indices(pick=None):
    if pick:
        idx=np.array([int(x) for x in pick.split(',')],dtype=int)
        assert idx.size==16 and idx.min()>=0 and idx.max()<=30
        return idx
    return np.linspace(0,30,16,dtype=int)

def convert_pick16(cube31, idx):
    return cube31[..., idx]

def convert_bin16(cube31):
    edges=np.linspace(0,31,17,dtype=int)
    out=np.zeros((*cube31.shape[:2],16),dtype=np.float32)
    for i in range(16):
        a,b=edges[i],edges[i+1]
        out[...,i]=cube31[...,a:b].mean(axis=-1)
    return out

def save_16(out, cube16, float32=False):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    if float32:
        arr=np.transpose(cube16,(2,0,1)).astype(np.float32)  # (C,H,W)
        tiff.imwrite(out, arr, dtype=np.float32)
    else:
        arr=np.clip(cube16,0,1)
        arr=(arr*65535.0+0.5).astype(np.uint16)
        arr=np.transpose(arr,(2,0,1))  # (C,H,W)
        tiff.imwrite(out, arr, dtype=np.uint16)

def parse_tile(tile_str):
    nums=[int(x.strip()) for x in tile_str.split(',')]
    assert len(nums)==16 and min(nums)>=0 and max(nums)<=15 and len(set(nums))==16
    return np.array(nums,dtype=int).reshape(4,4)

def mosaic_4x4(cube16, tile):
    H,W,C=cube16.shape
    if C!=16: raise ValueError("cube16 doit avoir 16 canaux")
    m=np.zeros((H,W),dtype=cube16.dtype)
    for i in range(H):
        ii=i%4
        for j in range(W):
            jj=j%4
            c=tile[ii,jj]
            m[i,j]=cube16[i,j,c]
    return m

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='Dossier ou motif glob (.mat/.tif) CAVE')
    ap.add_argument('--output', required=True, help='Dossier de sortie')
    ap.add_argument('--mode', choices=['pick16','bin16'], default='pick16')
    ap.add_argument('--pick', default=None, help='16 indices (0..30) pour pick16, ex: "0,2,4,...,30"')
    ap.add_argument('--also-mosaic', action='store_true')
    ap.add_argument('--tile', default=",".join(str(i) for i in range(16)))
    ap.add_argument('--save-float', action='store_true')
    args=ap.parse_args()

    # Fichiers d'entrée
    p=Path(args.input)
    if p.exists():
        files=sorted(list(p.glob('**/*.mat'))+list(p.glob('**/*.tif'))+list(p.glob('**/*.tiff')))
        if p.is_file(): files=[p]
    else:
        files=sorted(glob.glob(args.input))
    if not files: raise SystemExit("Aucun fichier trouvé pour --input")

    tile=parse_tile(args.tile)
    idx=pick_indices(args.pick)
    outdir=Path(args.output); outdir.mkdir(parents=True, exist_ok=True)

    for ip in files:
        ip=Path(ip)
        try:
            cube31=load_cube_any(ip)
            if cube31.shape[-1]!=31: raise ValueError(f"{ip.name}: {cube31.shape}")
            cube16=convert_pick16(cube31, idx) if args.mode=='pick16' else convert_bin16(cube31)
            out16=outdir/f"{ip.stem}_16bands.tif"
            save_16(out16.as_posix(), cube16, float32=args.save_float)
            if args.also_mosaic:
                m=mosaic_4x4(cube16, tile)
                if args.save_float:
                    tiff.imwrite((outdir/f"{ip.stem}_ms_mosaic.tif").as_posix(), m.astype(np.float32), dtype=np.float32)
                else:
                    mu16=(np.clip(m,0,1)*65535.0+0.5).astype(np.uint16)
                    tiff.imwrite((outdir/f"{ip.stem}_ms_mosaic.tif").as_posix(), mu16, dtype=np.uint16)
            print(f"[OK] {ip.name}")
        except Exception as e:
            print(f"[ERR] {ip}: {e}")

if __name__=='__main__':
    main()
