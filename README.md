# 増加バッチサイズを利用するオプティマイザの比較

分類クラス数 $k=100$ 、訓練データの総数 $n=50,000$ 、テストデータの総数 $t=10,000$ からなる画像分類用のベンチマークデータセットCIFAR-100(Canadian Institute For Advanced Research)を用いて、畳み込みニューラルネットワークResNet-18(ResidualNetwork；残差ネットワーク)を訓練します。

## オプティマイザ

- SGD：ウォームアップステップサイズ（ $\overline{\eta}=1.0$ ）と指数増加バッチサイズ$b_t$を利用する確率的勾配降下法

$$
\theta_{t+1}=\theta_{t}-\eta_t\nabla f_{B_t}(\theta_t)
$$

- SHB：ウォームアップステップサイズ（ $\overline{\eta}=1.0$ ）と指数増加バッチサイズ $b_t$ を利用する確率的ヘビーボール法（ $\beta=0.9$ ）
  
$$
m_t = \beta m_{t-1} + \nabla f_{B_t}(\theta_t)
$$
$$
\theta_{t+1}=\theta_{t}-\eta_t m_{t}
$$

- NSHB：ウォームアップステップサイズ（ $\overline{\eta}=0.5$ ）と指数増加バッチサイズ $b_t$ を利用する正則化確率的ヘビーボール降下法（ $\beta_1=0.9$ ）

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)\nabla f_{B_t}(\theta_t)
$$
$$
\theta_{t+1}=\theta_{t}-\eta_t m_{t}
$$

- RMSProp：定数ステップサイズ（ $\eta=0.01$ ）と指数増加バッチサイズ $b_t$ を利用するRMSProp（ $\beta_2=0.99$ ）

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2)\nabla f_{B_t}(\theta_t)^2
$$
$$
\theta_{t+1}=\theta_{t}-\frac{\eta}{\sqrt{v_t}+\epsilon}\nabla f_{B_t}(\theta_t)
$$

- Adam：定数ステップサイズ（ $\eta=0.001$ ）と指数増加バッチサイズ $b_t$ を利用するAdam（ $\beta_1=0.9,\beta_2=0.999$ ）
  
$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)\nabla f_{B_t}(\theta_t), \quad \hat{m}_t = \frac{m_t}{1-\beta_1^t}
$$
$$
v_t = \beta_2 v_{t-1} + (1-\beta_2)\nabla f_{B_t}(\theta_t)^2, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}
$$
$$
\theta_{t+1}=\theta_{t}-\frac{\eta}{\sqrt{\hat{v}_t}+\epsilon}\hat{m}_t
$$

- AdamW：定数ステップサイズ（ $\eta=0.001$ ）と指数増加バッチサイズ$b_t$を利用するAdamW（ $\beta_1=0.9,\beta_2=0.999,\lambda=0.01$ ）
  
$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)\nabla f_{B_t}(\theta_t), \quad \hat{m}_t = \frac{m_t}{1-\beta_1^t}
$$
$$
v_t = \beta_2 v_{t-1} + (1-\beta_2)\nabla f_{B_t}(\theta_t)^2, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}
$$
$$
\theta_{t+1}=(1-\lambda\eta)\theta_{t}-\frac{\eta}{\sqrt{\hat{v}_t}+\epsilon}\hat{m}_t
$$

## 実験結果

バッチ勾配ノルムと経験損失の比較

![結果1](graphs/norm_train.png)

テスト精度の比較

![結果2](graphs/test.png)
