# AutoPhotogrammetry

[![Test](https://github.com/KAFKA2306/AutoPhotogrammetry/actions/workflows/test.yml/badge.svg)](https://github.com/KAFKA2306/AutoPhotogrammetry/actions/workflows/test.yml)

**実写動画 → camera poses → Gaussian Splat PLY** を1タスクで再現します。

## Local responsibility

ローカルの責務は **RUNするだけ**です。

```bash
./task run
```

初回だけcloneします。

```bash
git clone https://github.com/KAFKA2306/AutoPhotogrammetry.git
cd AutoPhotogrammetry
./task run
```

環境確認、Docker image build、入力取得、hash検証、3D再構成、training、PLY export、成功判定は `./task run` 側が行います。

実行入口: [`task`](https://github.com/KAFKA2306/AutoPhotogrammetry/blob/main/task)  
GPU環境: [`Dockerfile`](https://github.com/KAFKA2306/AutoPhotogrammetry/blob/main/Dockerfile)  
E2E実装: [`processing/huejotzingo.py`](https://github.com/KAFKA2306/AutoPhotogrammetry/blob/main/processing/huejotzingo.py)

## What runs

```text
[Wikimedia CC0 video]
  -> SHA-256 verification
  -> [FFmpeg] frame extraction
  -> blur / duplicate filtering
  -> [COLMAP] feature_extractor
  -> [COLMAP] sequential_matcher
  -> [COLMAP] mapper
  -> [Nerfstudio] ns-process-data
  -> [Nerfstudio Splatfacto] ns-train splatfacto
  -> [Nerfstudio export] ns-export gaussian-splat
  -> PLY + manifest
```

- Input: [Wikimedia Commons — Ex Convento de San Miguel Arcángel, Huejotzingo](https://commons.wikimedia.org/wiki/File:Vista_del_Ex_Convento_de_San_Miguel_Arc%C3%A1ngel,_Huejotzingo,_desde_un_dron.webm)
- FFmpeg: https://ffmpeg.org/
- COLMAP CLI: https://colmap.github.io/cli.html
- Nerfstudio custom data: https://docs.nerf.studio/quickstart/custom_dataset.html
- Nerfstudio Splatfacto: https://docs.nerf.studio/nerfology/methods/splat.html
- Nerfstudio export: https://docs.nerf.studio/reference/cli/ns_export.html
- gsplat: https://github.com/nerfstudio-project/gsplat

## Input

固定検証データ:

- author: Luisalvaz
- license: CC0 1.0
- resolution: 1920×1080
- duration: 232.766 s
- SHA-256: `c9723df1af171d40a5bf1f9530aa3ea881c6f95252ef3f2004f0f1013ab92e30`

確認済みCOLMAP結果:

- registered: **78 / 78 images**
- sparse points: **32,782**
- mean reprojection error: **0.370830 px**

## Output

```text
output/huejotzingo/
├── frames/
├── selected/
├── colmap/
├── nerfstudio-data/
├── runs/
└── manifest.json
```

成功条件は `manifest.json` が次を持つことです。

```json
{
  "status": "success",
  "output": {
    "ply_path": "...",
    "ply_sha256": "...",
    "ply_size_bytes": 123456
  }
}
```

PLY hashが得られるまで成功扱いにしません。

## Scope

このrepoの終点は **Gaussian Splat PLY + provenance** です。

PLYのWeb / Unity / VRChat利用は [`KAFKA2306/vrmine`](https://github.com/KAFKA2306/vrmine) が担当します。
