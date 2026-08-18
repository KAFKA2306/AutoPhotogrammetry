# AutoPhotogrammetry

[![Test](https://github.com/KAFKA2306/AutoPhotogrammetry/actions/workflows/test.yml/badge.svg)](https://github.com/KAFKA2306/AutoPhotogrammetry/actions/workflows/test.yml)

![AutoPhotogrammetry の動作原理](assets/autophotogrammetry-principle.webp)

実写動画・写真群を、**出典と入力hashを残したまま再構成工程へ渡し、どこまで成功したかを確認できる形にする**ためのリポジトリです。

現時点で実データ検証済みなのは、Huejotzingo動画の取得・frame選別・COLMAP camera registrationまでです。Nerfstudio Splatfactoを外部CLIとして実行し、Gaussian Splat PLYのpath・SHA-256・sizeをmanifestへ保存する実装はありますが、**実GPUでのSplatfacto学習と実データPLY exportはまだ未検証**です。

- 実測済み: licensed video → FFmpeg → selected frames → COLMAP
- 実装済み・実GPU未検証: Nerfstudio data conversion → Splatfacto training → Gaussian Splat PLY export
- 別リポジトリの責務: PLYのWeb / Unity / VRChat利用は [`KAFKA2306/vrmine`](https://github.com/KAFKA2306/vrmine)

## Vision

写真群が再構成に向いているかを確認し、失敗した工程を切り分けながら、入力から3D成果物までの再現経路を残せるようにします。

## Design philosophy

- 入力の出典・ライセンス・SHA-256を再構成前に固定する
- blur、重複、scene cut、camera registrationなど中間結果を観測可能にする
- pipelineの終了コードと3D品質を同じ意味にしない
- COLMAPやSplatfactoは交換可能な外部toolとして扱い、独自実装で置き換えない
- 失敗時に別backendへ自動fallbackして成功扱いしない
- 大容量の動画、checkpoint、PLYをGit履歴へcommitしない
- 未測定値を推定値や0で埋めない

## Why

一発変換の成功表示ではなく、**source → frames → camera poses → training → export** の各段階を確認できることを重視します。どこで失敗したか、どの入力とtool versionから成果物が作られたかを追跡できるため、再撮影・frame選別・backend変更の判断を分離できます。

## Run

ローカルの正規入口は次です。

```bash
./task run
```

初回だけcloneします。

```bash
git clone https://github.com/KAFKA2306/AutoPhotogrammetry.git
cd AutoPhotogrammetry
./task run
```

`./task run` はDocker image build、GPU・tool確認、入力取得、hash検証、frame抽出、COLMAP、Splatfacto、PLY export、最終manifest判定を順に実行します。PLYのpath・SHA-256・sizeが得られない限り成功扱いにしません。

実行入口: [`task`](https://github.com/KAFKA2306/AutoPhotogrammetry/blob/main/task)  
GPU環境: [`Dockerfile`](https://github.com/KAFKA2306/AutoPhotogrammetry/blob/main/Dockerfile)  
E2E実装: [`processing/huejotzingo.py`](https://github.com/KAFKA2306/AutoPhotogrammetry/blob/main/processing/huejotzingo.py)

### Current environment contract

現在のDockerfile / `task doctor` が要求している環境です。これはNerfstudio一般の最小要件ではなく、このrepositoryの現在の固定環境です。

- NVIDIA container runtimeで `docker run --gpus all` が利用可能
- CUDA image: `nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04`
- GPU compute capability: `12.0` 以上を現在の`task doctor`が要求
- PyTorch: `2.7.1`
- torchvision: `0.22.1`
- Nerfstudio: `1.1.5`
- gsplat: `1.4.0`
- FFmpeg / FFprobe / COLMAP / `ns-process-data` / `ns-train` / `ns-export`

Nerfstudio公式ではSplatfactoを `ns-train splatfacto --data <data>` で学習し、学習済みsplatを `ns-export gaussian-splat --load-config <config> --output-dir <dir>` でPLYへexportできます。

- [Nerfstudio Splatfacto](https://docs.nerf.studio/nerfology/methods/splat.html)
- [Nerfstudio ns-train](https://docs.nerf.studio/reference/cli/ns_train.html)
- [Nerfstudio ns-export](https://docs.nerf.studio/reference/cli/ns_export.html)
- [COLMAP CLI](https://colmap.github.io/cli.html)
- [FFmpeg](https://ffmpeg.org/)
- [gsplat](https://github.com/nerfstudio-project/gsplat)

## Pipeline and current verification

```text
video / photos
  -> source + license + SHA-256
  -> FFmpeg frame extraction
  -> blur / duplicate filtering
  -> COLMAP camera poses
  -> Nerfstudio data conversion
  -> Splatfacto training
  -> Gaussian Splat PLY
  -> manifest
```

| Stage | Current state |
| --- | --- |
| source / license / input SHA-256 | verified on Huejotzingo |
| frame extraction / filtering | verified on Huejotzingo |
| COLMAP camera registration | verified on Huejotzingo |
| Splatfacto command / manifest / failure handling | implemented and unit-tested |
| real GPU Splatfacto training | not yet verified |
| real Gaussian Splat PLY export | not yet verified |
| Web / Unity / VRChat consumption | out of scope; handled by `vrmine` |

## Verified dataset

候補動画の正本は [`sources/videos.json`](https://github.com/KAFKA2306/AutoPhotogrammetry/blob/main/sources/videos.json) です。metadataだけでは成功点数を付けず、実測したstageだけを比較します。

現在のdefaultは `huejotzingo` です。

- source: [Wikimedia Commons — Ex Convento de San Miguel Arcángel, Huejotzingo](https://commons.wikimedia.org/wiki/File:Vista_del_Ex_Convento_de_San_Miguel_Arc%C3%A1ngel,_Huejotzingo,_desde_un_dron.webm)
- author: Luisalvaz
- license: CC0 1.0
- duration: 232.766 s
- local input: 1920×1080 VP9 transcode
- SHA-256: `c9723df1af171d40a5bf1f9530aa3ea881c6f95252ef3f2004f0f1013ab92e30`
- COLMAP: **78 / 78 registered**, **1 model**, **32,782 sparse points**, **0.370830 px mean reprojection error**

探索元: [Wikimedia Commons — Drone videos from Mexico](https://commons.wikimedia.org/wiki/Category:Drone_videos_from_Mexico)

## Candidate evaluation

候補は同じ評価段階に到達したもの同士だけを比較します。

```text
metadata
  -> preflight
     scene cuts / sharpness / overlap / camera translation
     dynamic pixels / exposure variation
  -> COLMAP
     registration / largest model / submodels / sparse points
     reprojection error / track length
  -> Splatfacto
     train / export / PSNR / SSIM / LPIPS / PLY hash
```

未測定候補へmetadata由来の疑似精密scoreは付けません。詳細は [Issue #23](https://github.com/KAFKA2306/AutoPhotogrammetry/issues/23) で管理しています。

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

最終成功条件は、実在するPLYとそのhashがmanifestで確認できることです。

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

## Validation and limitations

COLMAP registration成功や低いreprojection errorだけで、Gaussian Splatの見た目品質を保証しません。Splatfactoまで完走したdatasetでは、hold-out renderからPSNR / SSIM / LPIPSを測定する方針ですが、未実測値は表示しません。

実データPLY生成が完了するまで、README上でも「写真・動画からGaussian Splatを生成済み」とは扱いません。現在の未完了条件は [Issue #14](https://github.com/KAFKA2306/AutoPhotogrammetry/issues/14) と [Issue #15](https://github.com/KAFKA2306/AutoPhotogrammetry/issues/15) に残しています。

## Rights and third-party tools

- 入力写真・動画を利用できる権利は入力ごとに確認する必要があります。このrepositoryの存在自体は、任意の画像を再利用する権利を与えません。
- Huejotzingo sampleはWikimedia Commons上でCC0 1.0と表示されているsourceを利用しています。
- FFmpeg、COLMAP、Nerfstudio、gsplat、PyTorch等はそれぞれ独立したthird-party projectです。利用条件は各upstreamのlicenseを確認してください。
- 生成された3D assetの利用可否は、入力素材の権利・利用目的・適用される条件を別途確認してください。

## Scope

このrepositoryの終点は **Gaussian Splat PLY + provenance** です。

```text
AutoPhotogrammetry:
source media -> reconstruction / training -> Gaussian Splat PLY + provenance

vrmine:
Gaussian Splat PLY + provenance -> Web / Unity / VRChat validation
```
