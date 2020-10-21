
## get_best_weight.pyの使い方
### デフォルトのディレクトリ構成
```
├── get_best_weight_run.bat
├── get_best_weight.py
├── get_best_weight_conf.yaml
├── 入力画像フォルダ
│   ├── 条件A
|   |   ├── OK
|   |   └── NG
│   ├── 条件B
|   |   ├── OK
|   |   └── NG
~~~~~
|   └── 条件Z
|       ├── OK
|       └── NG
├── 最適重み画像出力フォルダ
│   ├── OK
│   └── NG
└── 0_POC
    ├── yama_workspace
    ├── tools
    └── argo_workspace
         ├── run.bat
         └── Image
              ├── learn
              ├── ngmasks
              ├── test_ng
              ├── test_ok
              └── マスク画像
```
### get_best_weight_run.bat
`get_best_weight_conf.yaml`のパスを記述するのみ。

### get_best_weight_conf.yamlについて
画像条件数に応じて以下のように記述する。
```
Image_A: 
- path: .\入力画像フォルダパス\A\
  resolution: 0.1
Image_B: 
- path: .\入力画像フォルダパス\B\
  resolution: 0.1
  ~~~
Image_Z: 
- path: .\入力画像フォルダパス\Z\
  resolution: 0.1
```
`key`の名前は変える必要なし。なんなら何でもいい。

### get_best_weight_conf.yamlのラベルについて

| Label | Description |
| --- | --- |
| `argo_path` | `argo_workspace`までのパス |
| `Image_〇` | `path`: ある条件画像までのパス<br>`resolution` : ある画像条件の重み最適化の分解能 |
| `minmax_values` | `min_value <= sum(WEIGHT_LIST[x]) <= max_value` となるWEIGHT_LISTのみを作成。両方`1.0`にすると重みの合計が`1.0`の平均画像のみが生成される |
| `label_list` | ラベルの種類に制限はないので`label_list`内をいじればカスタマイズ可能。しかしOKとNGはマスト |
| `best_image_output_path` | 一番良かった重み付き平均画像をこのフォルダに生成する。for_yamaのフラグをTrueにするとこのフォルダに任意重み画像がラベル付きで出力されるようなる |
| `csv_output_path` | 最終的に出力される`good_evaluation_values.csv`の出力先パス |
| `top_num` | 各重み画像の`evaluation_values.csv`からトップ`${top_num}`の`evaluation value`を持ってくる。グリッドサーチの回転数以内にしないと学習結果が全て消滅するので細心の注意を払って定義すること |
| `header_rows` | `evaluation_values.csv`のヘッダー行数。現在は3行だが、argoのアップデートとかでヘッダー行数が変わったときにはここをいじる|
| `header_cols` | `evaluation_values.csv`のヘッダー列数。現在は31列だが、argoのアップデートとかでヘッダー列数が変わったときにはここをいじる|
| `image_type_num` | 画像の条件数 |
| `train_data_rate` | 学習用OK画像の割合`(train:test = train_data_rate:(1-train_data_rate))` |
| `for_yama` | `yama`を使いたい時に使用。詳しい挙動は以下に記載 |


### `for_yama`について
* 重み付き合成画像だけを作りたい時、画像ファイル名にラベル情報を付加したい時にこれを使う。

* `True`にして実行すると、コマンドラインから各条件画像に対して重み情報を入力できるようになる。

* 入力後、`best_image_output_path`に重み付き合成画像が出力される。`yama`に是非。

### 使用上の注意
* `argo_workspace`フォルダに以降のフォルダ名はコード内で直接指定しているので、フォルダ名を同じにする必要あり（`run.bat`, `Image`フォルダ以降）。
* `argo_workspace`フォルダの`run.bat`内で`pause`コマンドを削除しないと動作がいちいち止まるので注意。
* `ngmasks`フォルダに入れるアノテーション画像の名前は`average_imageXXX.png`とする必要あり。
（参考コード：`cv2.imwrite('./○○/ngmasks/average_image' + str(i).zfill(3) + '.png', image)`, `image`変数には`cv2.imread`した`ndarray`が格納されているとする。）

### 自分で重みのリストを定義したい時

main文内に以下のように直接定義する。
```
WEIGHT_LIST = np.array([
  [1.0, 0.0, 0.0, 0.0],
  [0.80, 0.05, 0.05, 0.05, 0.05],
  [0.75, 0.05, 0.05, 0.05, 0.05, 0.05],
  ],dtype=np.float64)
```
（※）合計が1である必要はない。
