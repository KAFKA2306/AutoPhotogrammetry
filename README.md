# AutoPhotogrammetry

[![Test](https://github.com/KAFKA2306/AutoPhotogrammetry/actions/workflows/test.yml/badge.svg)](https://github.com/KAFKA2306/AutoPhotogrammetry/actions/workflows/test.yml)
[![GPU Image](https://github.com/KAFKA2306/AutoPhotogrammetry/actions/workflows/gpu-image.yml/badge.svg)](https://github.com/KAFKA2306/AutoPhotogrammetry/actions/workflows/gpu-image.yml)

**実写動画 → camera poses → Gaussian Splat PLY** を1タスクで再現します。

## Run

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

GPU用のCUDA / PyTorch / gsplat / Nerfstudio環境をbuild/pushする `GPU Image` workflow はmainに存在します。通常のGPU実行ではローカルでNerfstudio checkoutやpip installを行いません。`task` は指定imageがlocalにあればそのimageを使い、なければGHCRからpullを試みます。imageが取得できない場合はlocal buildへ自動fallbackせず失敗します。

**2026-08-20時点でrepositoryから確認済みなのはworkflow定義・Dockerfile・runner・通常Test workflowのPASSまでです。GHCR上のtag publish成功や対象PCからのpull成功は、実GPU runのevidenceが残るまで確認済みとは扱いません。**

Docker CLIまたはDocker daemonが利用できない場合、`./task doctor` / `./task run` / `./task run-all` / `./task quality` は明示的に停止します。このrepositoryのtaskはDocker Desktop、WSL、GPU driver、OS設定、Docker storageを修復・resetしません。host環境の修復はrepository実行とは分離してください。

実行入口: [`task`](https://github.com/KAFKA2306/AutoPhotogrammetry/blob/main/task)  
GPU環境: [`Dockerfile`](https://github.com/KAFKA2306/AutoPhotogrammetry/blob/main/Dockerfile)  
E2E実装: [`processing/huejotzingo.py`](https://github.com/KAFKA2306/AutoPhotogrammetry/blob/main/processing/huejotzingo.py)

### 既存processed datasetで品質候補をGPU比較する

既に `output/<scene>/nerfstudio-data/transforms.json` があるsceneでは、操作入口は次の1コマンドです。

```bash
./task quality huejotzingo 30000
```

`task quality` はhost側でDocker daemon、GPU compute capability、指定imageの存在/pullをpreflightした後、**1回のGPU container invocation**でquality sweepを起動します。source download、FFmpeg、COLMAP、Nerfstudio checkout、pip install、Docker buildはquality pathでは行いません。

GPU container内では同じprocessed dataset、同じ30,000 iteration budget、同じexact Nerfstudio / gsplat環境で以下を独立実行します。

- upstream default Splatfacto
- `use_scale_regularization=True`
- `strategy=mcmc`

各PLYについてSHA-256、size、primitive count、opacity < 0.1、scale anisotropy > 10を自動集計し、GPU名、compute capability、PyTorch/CUDA runtime、container image ref/IDとともに `output/<scene>/quality-sweep/quality-sweep.json` に保存します。semantic maskが必要なClean-GSはtraining strategyの比較と混ぜず、別のpost-process実験として扱います。

DockerfileはNerfstudio commit `50e0e3c70c775e89333256213363badbf074f29d`、gsplat `1.4.0`、CUDA 12.8 / PyTorch 2.7.1、`sm_120`を固定します。floating `main` には依存しません。

現時点ではrunnerとunit testはmainにmerge済みですが、bad scene / good controlでの実GPU quality sweep、PSNR / SSIM / LPIPS、fixed novel-view、winner strategyは未確定です。

## 撮影セットを3D化前に監査する

収集済みの写真セットは、既存の非破壊選別を使ってJSON / HTMLの入力監査レポートにできます。

```bash
python main.py audit --dataset <dataset-id>
```

このコマンドは元画像を削除せず、`readiness-report.json`、`readiness-report.html`、`selected-manifest.json` と別の `selected/` を生成します。登録画像率、再投影誤差、mesh completenessを未測定のまま3D品質保証には使いません。

実装上の監査仕様は [`docs/business/photogrammetry-input-audit.md`](docs/business/photogrammetry-input-audit.md)、無料sample・有償PoC・batch相談・権利条件・3つのCTA・60日KPI契約は [`docs/business/photogrammetry-readiness-service.md`](docs/business/photogrammetry-readiness-service.md) を参照してください。

## 軽量な開発環境

編集はZed、Python環境はuv、GPU処理はDockerに分離しています。ZedのAI機能や常駐エージェントは使わず、通常の編集・診断・テストはホスト上で完結します。

初回セットアップ:

```bash
uv sync
zed .
```

Zedのタスク（`Ctrl+Shift+P` → `task: spawn`）には、次を用意しています。

- `Host: Run unit tests` — Dockerを使わないユニットテスト
- `Host: Compile check` — Python構文・コンパイル確認
- `Host: Ruff check` — 軽量Lint
- `Pipeline: Doctor (GPU + Docker)` — CUDA/Nerfstudioイメージを必要時だけ検査
- `Pipeline: Run Huejotzingo` — GPUを使うフルパイプライン

Zed設定は [`.zed/settings.json`](.zed/settings.json)、タスク定義は [`.zed/tasks.json`](.zed/tasks.json) です。

## Pipeline

```text
video
  -> SHA-256 verification
  -> scene-cut segmentation
  -> continuous-shot selection
  -> FFmpeg frame extraction / blur・duplicate filtering
  -> shot-level overlap / parallax / pose preflight
  -> COLMAP camera poses / sparse reconstruction
  -> Nerfstudio ns-process-data
  -> Splatfacto training / hold-out evaluation
  -> Gaussian Splat PLY
  -> manifest
```

Repository-sideのshot-level Stage B/C routingはPR #125で実装済みです。scene-cutから連続shotを作り、shotごとのpose-aware evidenceで選択してからCOLMAPへ渡します。これは実動画20件のStage B/C/Dが完走済みであることを意味しません。

- [FFmpeg](https://ffmpeg.org/)
- [COLMAP CLI](https://colmap.github.io/cli.html)
- [Nerfstudio custom data](https://docs.nerf.studio/quickstart/custom_dataset.html)
- [Nerfstudio Splatfacto](https://docs.nerf.studio/nerfology/methods/splat.html)
- [Nerfstudio export](https://docs.nerf.studio/reference/cli/ns_export.html)
- [gsplat](https://github.com/nerfstudio-project/gsplat)

## Video candidates

候補動画の正本は [`sources/videos.json`](https://github.com/KAFKA2306/AutoPhotogrammetry/blob/main/sources/videos.json) です。現在20候補を一元管理しています。

**metadataだけでは成功点数を付けません。** 評価は実測で進めます。

```text
metadata
  -> continuous shots
  -> shot-level preflight
     scene cuts / sharpness / overlap / pose-aware translation / parallax
     dynamic pixels / exposure variation / triangulation evidence
  -> COLMAP
     registration / largest model / submodels / sparse points
     reprojection error / track length
  -> Splatfacto
     train / export / PSNR / SSIM / LPIPS / PLY hash
```

### 現在の実測状態

2026-08-20時点で、commit [`1fa7f35`](https://github.com/KAFKA2306/AutoPhotogrammetry/commit/1fa7f35b1fc9fce669f97d5ec7c7f46ce4601206) に **9件の実Gaussian Splat PLY** が存在します。最終展示要件は20件なので、完成状態ではありません。

- 実PLY: **9 / 20**
- Huejotzingo PLY: **38,085,465 bytes**
- Huejotzingo SHA-256: `6dc1d2546ab848eee4587fdaaebe1b60b1f2495f8a9a9b6a58de9356222c4571`
- 20展示の完成authority: [Issue #33](https://github.com/KAFKA2306/AutoPhotogrammetry/issues/33)
- 1本のsource-to-PLY lineage完成authority: [Issue #15](https://github.com/KAFKA2306/AutoPhotogrammetry/issues/15)

既存9 PLYの存在・代表PLYのsize/hashは確認済みですが、古いHuejotzingo run directoryにはtraining config/checkpoint/logが残っていないため、その過去runのexact `ns-train` / `ns-export` lineageを後から復元したとは扱いません。現在のrunnerは次回runからtool version、command、return code、config/checkpoint、PLY hash/sizeをmanifestへ保存します。

今後生成する大容量PLYは通常のGit履歴へ追加せず、`output/**/runs/**/export/*.ply` を生成artifactとして扱います。

現在のdefaultは `huejotzingo` です。registryの20本をすべて処理する場合は次を実行します。

```bash
./task run-all
```

各動画は `output/<id>/manifest.json` と独立したGaussian Splat PLYを生成し、全体の
`output/batch-manifest.json` に20件の成功・失敗とPLY hashを記録します。`run-all` は
途中生成物を消してから全件を再実行します。個別に再実行したい場合も `task` を入口にします。
実証時のSplatfactoは `--max-num-iterations 2000` で実行します。品質比較は上記 `./task quality <scene> 30000` を使います。

- source: [Wikimedia Commons — Ex Convento de San Miguel Arcángel, Huejotzingo](https://commons.wikimedia.org/wiki/File:Vista_del_Ex_Convento_de_San_Miguel_Arc%C3%A1ngel,_Huejotzingo,_desde_un_dron.webm)
- author: Luisalvaz
- license: CC0 1.0
- duration: 232.766 s
- local input: 1920×1080 VP9 transcode
- SHA-256: `c9723df1af171d40a5bf1f9530aa3ea881c6f95252ef3f2004f0f1013ab92e30`
- COLMAP: **78 / 78 registered**, **1 model**, **32,782 sparse points**, **0.370830 px mean reprojection error**