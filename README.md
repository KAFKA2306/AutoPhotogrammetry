# AutoPhotogrammetry

[![Test](https://github.com/KAFKA2306/AutoPhotogrammetry/actions/workflows/test.yml/badge.svg)](https://github.com/KAFKA2306/AutoPhotogrammetry/actions/workflows/test.yml)

**実写動画 → COLMAP camera poses → Nerfstudio Splatfacto → Gaussian Splat PLY** を、1タスクで再現するリポジトリです。

```text
input/ -> processing/ -> output/
```

## Run

必要なのは、NVIDIA GPUを使えるDocker環境です。

```bash
git clone https://github.com/KAFKA2306/AutoPhotogrammetry.git
cd AutoPhotogrammetry
./task run
```

`make run` でも同じです。

VS Codeでは **Terminal → Run Build Task → AutoPhotogrammetry: Run Huejotzingo**。

実行内容:

```text
Wikimedia CC0 video
  -> SHA-256 verification
  -> FFmpeg frame extraction
  -> blur / duplicate filtering
  -> COLMAP feature_extractor
  -> COLMAP sequential_matcher
  -> COLMAP mapper
  -> ns-process-data
  -> ns-train splatfacto
  -> ns-export gaussian-splat
  -> PLY + manifest
```

実装: [`processing/huejotzingo.py`](https://github.com/KAFKA2306/AutoPhotogrammetry/blob/main/processing/huejotzingo.py)  
実行入口: [`task`](https://github.com/KAFKA2306/AutoPhotogrammetry/blob/main/task)  
GPU環境: [`Dockerfile`](https://github.com/KAFKA2306/AutoPhotogrammetry/blob/main/Dockerfile)

## Output

成果物は `output/huejotzingo/` に生成されます。

```text
output/huejotzingo/
├── frames/
├── selected/
├── colmap/
├── nerfstudio-data/
├── runs/
└── manifest.json
```

成功条件:

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

`manifest.json` が `success` で、PLYのSHA-256が記録されるまでE2E成功とは扱いません。

## Input

固定検証データは、メキシコ・Huejotzingoの **Ex Convento de San Miguel Arcángel** のドローン動画です。

- source: [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Vista_del_Ex_Convento_de_San_Miguel_Arc%C3%A1ngel,_Huejotzingo,_desde_un_dron.webm)
- author: Luisalvaz
- license: CC0 1.0
- resolution: 1920×1080
- duration: 232.766 s
- SHA-256: `c9723df1af171d40a5bf1f9530aa3ea881c6f95252ef3f2004f0f1013ab92e30`

確認済みCOLMAP結果:

- 78 / 78 images registered
- 32,782 sparse points
- mean reprojection error: 0.370830 px

## Tasks

```bash
./task run        # video -> PLY
./task doctor     # Docker / GPU / CUDA / CLI check
./task test       # unit tests
./task image      # build Docker image
./task clean      # remove output/huejotzingo
./task clean-all  # remove output + downloaded source
```

VS Code tasks: [`.vscode/tasks.json`](https://github.com/KAFKA2306/AutoPhotogrammetry/blob/main/.vscode/tasks.json)

## Stack

- [FFmpeg](https://ffmpeg.org/)
- [COLMAP CLI](https://colmap.github.io/cli.html)
- [Nerfstudio custom data](https://docs.nerf.studio/quickstart/custom_dataset.html)
- [Nerfstudio Splatfacto](https://docs.nerf.studio/nerfology/methods/splat.html)
- [Nerfstudio export](https://docs.nerf.studio/reference/cli/ns_export.html)
- [gsplat](https://github.com/nerfstudio-project/gsplat)
- [PyTorch CUDA builds](https://pytorch.org/get-started/previous-versions/)

RTX 50-series用のCUDA / PyTorch / gsplat構成は [`Dockerfile`](https://github.com/KAFKA2306/AutoPhotogrammetry/blob/main/Dockerfile) に固定しています。

## Scope

このrepoの終点は **Gaussian Splat PLY + provenance** です。

Web / Unity / VRChatでの利用検証は [`KAFKA2306/vrmine`](https://github.com/KAFKA2306/vrmine) で扱います。

大容量source、checkpoint、PLYはGitへcommitしません。外部CLI欠損やhash不一致ではfallbackせず停止します。
