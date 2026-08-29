# 3D化前の撮影セット監査

AutoPhotogrammetryは、利用権を確認した写真セットを3D再構成へ投入する前に監査します。元画像を削除・上書きせず、JSON / HTMLレポート、選別済み画像セット、来歴とSHA-256を残します。合意した再構成処理を含む場合だけ、その実行証跡も納品対象にします。

対象は、所蔵品を3D化する博物館・資料館・研究室、商品3D表示を検討するEC事業者・メーカー、受領写真の選別を標準化したい3D制作会社です。

## 提供するもの

初期サービスは、**写真セットの監査と再構成前処理**です。完全な3Dモデルや実寸精度を保証するサービスではありません。

監査では、実測できた範囲で次を返します。

- 入力画像数と選別画像数
- 完全重複・類似画像の証拠
- 低鮮明度の警告
- 来歴情報の充足率とSHA-256
- 画像寸法とcontent type
- 元画像とは別の選別済み画像セットとmanifest
- `generated_views_used=false`
- 再構成を実施した場合のbackend実行状態とrun manifest
- 未測定項目を推測値で埋めない明示的な状態

登録画像率、再投影誤差、形状の完全性、texture品質、実寸精度を実測していない場合、それらを保証しません。

## 必要な素材と権利

顧客データは次を満たすものだけを扱います。

1. 顧客自身が写真を利用できる権利を持つ、または目的に必要な利用許諾を確認している。
2. 顧客画像、個人情報、認証情報、未公開契約、その他の機密情報を公開GitHub Issueへ添付しない。
3. 画像本体を受け渡す前に、非公開の転送方法を別途合意する。
4. 顧客画像を別途の明示許可なく公開fixtureへ転用しない。

## 無料sample

`sample-readiness-report.json` はレポート形式を説明するためのsampleです。`sample_kind` は `illustrative_synthetic_no_customer_data` であり、顧客実績や3D再構成性能の証拠ではありません。

sampleで確認できるもの:

- readiness reportの項目
- selected manifestの考え方
- 除外・警告理由の集計
- 来歴とhashの項目
- 未測定品質をnullのまま保持する契約

## 有償の1対象物PoC

1対象物と合意した写真セットについて、次を組み合わせます。

- 撮影セット監査
- 再構成投入用の選別済み画像セット
- 提供manifestから確認できる来歴・SHA-256
- 必要な実行環境が利用できる場合の、合意した再構成backend 1系統
- backend run manifestと生成artifact一覧
- 失敗した条件・未測定条件の明示

価格、納期、画像枚数、利用する再構成backend、機密保持、権利条件は案件ごとに合意します。初期の支払意思検証ではSaaS化や自動決済を前提にしません。

## 複数対象物・社内導入

10対象物以上では、batch処理、撮影ガイド、再撮影ラウンド、CLIのprivate deploymentを相談対象にします。実際の有効相談がない段階では継続需要があるとは扱いません。

## 相談する

相談入口は既存のGitHub Issue Form 1つに統一しています。公開Issueには機密情報を入れず、対象物種別、概算件数、画像枚数、権利状態、3Dの用途、今回判断したいことだけを入力します。

- [撮影セットを監査する](https://github.com/KAFKA2306/AutoPhotogrammetry/issues/new?template=photogrammetry-service.yml&title=%5BPhotogrammetry+inquiry%5D+Input+audit)
- [1対象物のPoCを相談する](https://github.com/KAFKA2306/AutoPhotogrammetry/issues/new?template=photogrammetry-service.yml&title=%5BPhotogrammetry+inquiry%5D+One-object+PoC)
- [大量3D化を相談する](https://github.com/KAFKA2306/AutoPhotogrammetry/issues/new?template=photogrammetry-service.yml&title=%5BPhotogrammetry+inquiry%5D+Batch+or+private+deployment)

## 成果の測定

初期検証で区別する事実は、サービス面の閲覧、sample閲覧、相談開始、有効相談、PoC合意、有償PoCです。顧客画像、氏名、メールアドレス、電話番号、Issue本文などの個人・機密情報を計測データへ保存しません。

Issue #3の外部成果は、実際の相談・PoC・支払いが発生して初めて達成とします。この文書や問い合わせ導線の公開だけを有償化成功とは扱いません。
