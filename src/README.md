# 増加バッチサイズを利用するオプティマイザの比較

分類クラス数$k=100$、訓練データの総数$n=50,000$、テストデータの総数$t=10,000$からなる画像分類用のベンチマークデータセットCIFAR-100(Canadian Institute For Advanced Research)を用いて、畳み込みニューラルネットワークResNet-18(ResidualNetwork；残差ネットワーク)を訓練します。

## Wandb Setup

`"cifar100.py"`のentity名の XXXXXX を、あなたの wandb エンティティ名に変更してください。

```bash
wandb.init(config=config, project=wandb_project_name, name=wandb_exp_name, entity="XXXXXX")
```

## 使用方法

```bash
python cifar100.py XXXXXX.json
```

## JSON 設定ファイルの例

以下は、次の条件となるように設定した JSON 設定ファイルの例です。

- オプティマイザ：SGD
- 学習率：ウォームアップ
  - 初期値 0.1 から最大 1.0 まで
  - 最初の 30 エポックで、3 エポックごとに段階的に増加
- バッチサイズ：指数増加
  - 初期値 8
  - 30 エポックごとに 2 倍

```json
{
  "optimizer": "sgd",
  "bs_method": "exp_growth",
  "lr_method": "warmup_const",
  "init_bs": 8,
  "init_lr": 0.1,
  "lr_max": 1.0,
  "epochs": 300,
  "incr_interval": 30,
  "warmup_epochs": 30,
  "warmup_interval": 3,
  "bs_growth_rate": 2.0,
  "use_wandb": true
}
```

以下は、上記 JSON 例で使用されている各設定パラメータの詳細説明です。

| パラメータ | データ型 & 例 | 説明 |
| :- | :- | :- |
| `optimizer` | `str` (`"nshb"`, `"shb"`, `"sgd"`, `"rmsprop"`, `"adam"`, `"adamw"`) | 学習中に使用するオプティマイザを指定します。 |
| `bs_method` | `str` (`"constant"`, `"exp_growth"`) | バッチサイズの調整方法を指定します。 |
| `lr_method` | `str` (`"constant"`, `"cosine"`, `"diminishing"`,<br>`"linear"`, `"poly"`, `"exp_growth"`,<br>`"warmup_const"`, `"warmup_cosine"`) | 学習率の調整方法を指定します。 |
| `init_bs` | `int` (`128`) | 初期バッチサイズ。 |
| `init_lr` | `float` (`0.1`) | 初期学習率。 |
| `bs_max` | `int` (`4096`) | バッチサイズを増加させる場合の最大バッチサイズ。`bs_method="exp_growth"` のときに使用されます。 |
| `lr_max` | `float` (`0.2`) | 学習率を増加させる場合の最大学習率。`lr_method="exp_growth"`, `"warmup_const"`, `"warmup_cosine"`のときに使用されます。 |
| `lr_min` | `float` (`0.001`, default `0`) | コサインアニーリングにおける最小学習率。 `lr_method="cosine"` または `"warmup_cosine"` のときに使用されます。 |
| `epochs` | `int` (`300`) | 学習の総エポック数。 |
| `incr_interval` | `int` (`30`) | バッチサイズや学習率を増加させる間隔（エポック単位）。 `bs_method="exp_growth"` のときに使用されます。 |
| `warmup_epochs` | `int` (`30`) | ウォームアップのエポック数。 `lr_method="warmup_const"` または `"warmup_cosine"` のときに使用されます。 |
| `warmup_interval` | `int` (`3`) | ウォームアップ中に学習率を増加させる間隔（エポック単位）。 `lr_method="warmup_const"` または `"warmup_cosine"` のときに使用されます。 |
| `bs_growth_rate` | `float` (`2.0`) | バッチサイズの増加率。`bs_method="exp_growth"` のときに使用されます。 |
| `lr_growth_rate` | `float` (`1.2`) | 学習率の増加率。 `lr_method="exp_growth"`, `"warmup_const"`, `"warmup_cosine"`のときに使用されます。 |
| `power` | `float` (`2.0`) | 多項式減衰における指数。`lr_method="poly"`のときに使用されます。 |
| `use_wandb` | `boolean` (`true`/`false`) | Weights & Biases（wandb）へのログ記録を有効化します。 |
