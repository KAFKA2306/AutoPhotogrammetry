# Gaussian Splat artifact storage

AutoPhotogrammetryの終点は、検証済みGaussian Splat PLYとそのprovenanceです。大容量PLYはGitへcommitせず、`KAFKA2306/hf-cache-hub` の共通artifact contractを通してHugging Face Storage Bucketへpublishします。

## Responsibility boundary

```text
AutoPhotogrammetry
  generate PLY
  verify local size + SHA-256
  build run-local artifact-manifest.yaml
  call hf-cache-hub publisher
        |
        v
hf-cache-hub
  validate manifest
  upload object
  read back object
  verify remote size + SHA-256
        |
        v
Hugging Face Storage Bucket
```

AutoPhotogrammetryはStorage Bucket APIを独自実装しません。upload/download、retry、remote readback判定はhf-cache-hubへ委譲します。Hugging Face token、password、API key等のcredentialもrepository、run manifest、artifact manifestへ保存しません。

## Prerequisites

```bash
git clone https://github.com/KAFKA2306/hf-cache-hub.git ~/src/hf-cache-hub
export HF_CACHE_HUB_ROOT="$HOME/src/hf-cache-hub"
export HF_ARTIFACT_BUCKET="<namespace>/<bucket>"
```

`HF_ARTIFACT_BUCKET` は実在し書き込み可能なStorage Bucketを指定してください。実際のupload/readbackを観測していないbucketをpublish済みとは扱いません。

## Publish one successful run

まず通常pipelineでPLYを生成します。run manifestが `status: success` であり、`splatfacto.ply_path`、`ply_size_bytes`、`ply_sha256` を持つことが前提です。

```bash
python main.py publish-splat \
  --run-manifest output/<dataset>/manifest.json
```

明示的に指定する場合:

```bash
python main.py publish-splat \
  --run-manifest output/<dataset>/manifest.json \
  --bucket '<namespace>/<bucket>' \
  --hf-cache-hub-root ~/src/hf-cache-hub
```

publish前にlocal PLYを再hashし、successful run manifestと一致しなければremote処理へ進みません。成功時だけrun manifestへ `artifact_publish.status: published` と `remote_verified: true` を記録します。

## Failure semantics

PLY生成成功とremote publish成功は別状態です。upload、readback、hash検証のどこかで失敗しても、top-levelのreconstruction `status: success` とlocal PLYは保持します。そのためGPU trainingをやり直さずpublishだけ再実行できます。

生成された `artifact-manifest.yaml` は、そのrunのimmutable identityとして `size_bytes`、SHA-256、exact Git revision、run/source provenanceを持ちます。Storage Bucket pathはmutableなので、path単独をartifact identityとして扱いません。

## Consumer handoff

remote publish/readbackが成功したartifactだけを、consumer側の正準manifest/registryへ移します。現在のconsumer例は `KAFKA2306/vrmine` です。consumerがhf-cache-hub shared cacheから同じSHA-256 objectを解決できることを確認する前に、既存の復元可能なsourceを削除しません。
