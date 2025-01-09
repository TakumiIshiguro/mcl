# MCL
このレポジトリはモンテカルロ位置推定（Monte Carlo Localization）を１次元の数直線上に実装したものです。

## デモ

## 内容
#### mcl の実装

* Monte Carlo Localization : [mcl.ipynb](https://github.com/TakumiIshiguro/mcl/blob/main/mcl/mcl.ipynb)

#### ロボットの準備 

* ロボットのモデル化 : [ideal_robot.ipynb](https://github.com/TakumiIshiguro/mcl/blob/main/scripts/ideal_robot.ipynb)
* 不確かさのモデル化 : [robot.ipynb](https://github.com/TakumiIshiguro/mcl/blob/main/mcl/robot.ipynb)

#### パーティクルのテスト

* 準備 : [particle_test.ipynb](https://github.com/TakumiIshiguro/mcl/blob/main/mcl/particle_test.ipynb)
* 移動後の更新 : [particle_move.ipynb](https://github.com/TakumiIshiguro/mcl/blob/main/mcl/particle_move.ipynb)
* 重みの付与, 尤度関数の実装: [particle_weight.ipynb](https://github.com/TakumiIshiguro/mcl/blob/main/mcl/particle_weight.ipynb)

## 動作確認済み環境
* Jupyter Notebook : 6.5.4  
* Python : 3.11.3 

## 参考文献
『詳解 確率ロボティクス ― Pythonによる基礎アルゴリズムの実装 ―』講談社〈KS理工学専門書〉、2019年、ISBN 978-406-51-7006-9

## ライセンス
このレポジトリは MIT ライセンスのもとで公開されています。