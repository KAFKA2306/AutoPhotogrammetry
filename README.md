# AutoPhotogrammetry — clone → task → PLY

[![Test](https://github.com/KAFKA2306/AutoPhotogrammetry/actions/workflows/test.yml/badge.svg)](https://github.com/KAFKA2306/AutoPhotogrammetry/actions/workflows/test.yml)

許諾済みの実写入力を `input/` に固定し、`processing/` だけで処理して、camera pose・Nerfstudio dataset・Gaussian Splat PLY・manifestを `output/` に生成します。

```text
input/ -> processing/ -> output/
```

独自のSfM / 3DGS training / rasterizerは実装しません。FFmpeg、COLMAP、Nerfstudio Splatfacto、gsplatを外部実装として利用し、入力hash、command、version、return code、logs、最終PLY hashを残します。

## ローカル実行

想定環境は Windows 11 + WSL2 / Linux + NVIDIA RTX 50-series です。ホスト側で必要なのは、`docker run --gpus all` が使えるDockerだけです。Python、FFmpeg、COLMAP、Nerfstudioをホストへ個別installする必要はありません。

```bash
git clone https://github.com/KAFKA2306/AutoPhotogrammetry.git
cd AutoPhotogrammetry
./task run
```

`make run` でも同じです。VS Codeでは **Terminal → Run Build Task** の `AutoPhotogrammetry: Run Huejotzingo` が同じ処理を呼びます。

`./task run` は次を順番に実行します。

```text
Docker image build / GPU doctor
  -> verified Wikimedia source download + SHA-256
  -> FFmpeg: 3秒間隔 / 1024px frame生成
  -> blur / near-duplicate selection
  -> COLMAP feature_extractor
  -> COLMAP sequential_matcher
  -> COLMAP mapper
  -> existing COLMAP modelをns-process-dataへ渡す
  -> ns-train splatfacto
  -> ns-export gaussian-splat
  -> PLY + manifest
```

原本 `input/huejotzingo/source.webm` はSHA-256が一致すれば再利用します。`output/huejotzingo/` は実行開始時に作り直すため、古い中間生成物を今回の成果と混同しません。

成功時は `output/huejotzingo/manifest.json` が `status: "success"` になり、最終PLYへのpathとSHA-256を持ちます。

## Tasks

```bash
./task run        # E2E: source -> Gaussian Splat PLY
./task doctor     # Docker / GPU / CUDA / CLIを確認
./task test       # unit tests
./task image      # 実行imageだけbuild
./task clean      # output/huejotzingo を削除
./task clean-all  # outputと取得済みsourceを削除
```

Makeからも同名targetを実行できます。

## Blackwell / RTX 50-series

Nerfstudioの従来のCUDA 11.8構成はRTX 5070等の `sm_120` に適合しないため、このrepositoryの実行imageは次に固定しています。

- NVIDIA CUDA 12.8.1 development image
- PyTorch 2.7.1 + CUDA 12.8
- Nerfstudio 1.1.5
- gsplat 1.4.0をCUDA 12.8 / `TORCH_CUDA_ARCH_LIST=12.0` でsource build

Nerfstudio 1.1.5はgsplat 1.4.0を要求するため、別versionへ差し替えず同じversionをsourceからbuildします。PyTorch 2.6以降の`torch.load`既定変更に対しては、**このtask自身が生成したcheckpointだけ**を再読込する用途で `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` を設定します。外部から取得した未知のcheckpointをこのtaskへ渡しません。

## 固定入力: Huejotzingo

対象はメキシコ・Huejotzingoの **Ex Convento de San Miguel Arcángel**。Wikimedia CommonsのCC0動画を使用します。

- author: Luisalvaz
- license: CC0 1.0 Universal
- 1920×1080 VP9 WebM
- duration: 232.766 s
- size: 115,502,605 bytes
- SHA-256: `c9723df1af171d40a5bf1f9530aa3ea881c6f95252ef3f2004f0f1013ab92e30`

既存のCOLMAP検証実績:

- input images: 78
- registered images: 78 / 78
- registration rate: 100%
- sparse points: 32,782
- mean reprojection error: 0.370830 px

GPU Splatfacto / PLYはローカル `./task run` が最終ゲートです。CPU GitHub Actionsのmock testsをGPU E2E成功とは扱いません。

## 構造

```text
AutoPhotogrammetry/
├── input/                  # 原本
├── processing/
│   ├── collection.py
│   ├── huejotzingo.py      # 正準E2E
│   ├── image_selection.py
│   ├── video.py
│   ├── photogrammetry.py
│   ├── nerfstudio.py
│   └── provenance.py
├── output/                 # 中間生成物 + PLY + manifests
├── tests/
├── .vscode/tasks.json
├── Dockerfile
├── task
├── Makefile
└── main.py
```

`KAFKA2306/AutoPhotogrammetry` の責務は実写入力からGaussian Splat PLY + provenanceまでです。PLYのWeb / Unity / VRChat互換性検証は `KAFKA2306/vrmine` の責務です。

## 制約

- source video、checkpoint、大容量PLYはGitへcommitしない
- 生成AIの別角度画像をSfM / 3DGS入力へ混ぜない
- 検索engineをscrapeしない
- 外部CLI欠損やhash不一致でfallbackしない
- COLMAP登録成功だけで3DGS成功を主張しない
- `manifest.json` が `success` かつPLY hashが得られるまでE2E成功としない

## 一次資料

- PyTorch 2.7.1 CUDA 12.8: https://pytorch.org/get-started/previous-versions/
- PyTorch serialization: https://docs.pytorch.org/docs/main/notes/serialization.html
- Nerfstudio custom data: https://docs.nerf.studio/quickstart/custom_dataset.html
- Nerfstudio Splatfacto: https://docs.nerf.studio/nerfology/methods/splat.html
- Nerfstudio 1.1.5 dependencies: https://github.com/nerfstudio-project/nerfstudio/blob/v1.1.5/pyproject.toml
- gsplat 1.4.0 source build: https://github.com/nerfstudio-project/gsplat/tree/v1.4.0
- COLMAP CLI: https://colmap.github.io/cli.html
