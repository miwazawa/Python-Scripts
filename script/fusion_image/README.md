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
### get_best_weight_run.batの引数
* 第1引数：`get_best_weight_conf.yaml`のパス

* 第2引数：最適重み画像の出力フォルダパス

* 第3引数：画像条件数

* 第4引数：分解能((0.1, 0.125, 0.2, 0.25, 0.5, 1.0)など割り切れそうな数に対応)

* (第5引数：`True`を与えるとラベル付きの任意重み画像を最適重み画像出力フォルダに生成する。主にyamaを使うときに使用。重みはコマンドラインに従ってその都度入力する。)

### 第5引数について
重み付き合成画像だけを作りたい時、画像ファイル名にラベル情報を付加したい時にこれを使う。

第5引数に`True`を与えると、コマンドラインから各条件画像に対して重み情報を入力できるようになる。

入力後、第2引数の最適重み画像の出力フォルダに重み付き合成画像が出力される。yamaに是非。

### get_best_weight_conf.yaml内容
上記ディレクトリ構成に従ってyamlファイルを書いた例を以下に示す。
```
argo_path: .\0_POC\argo_workspace\
Image_A: .\入力画像フォルダパス\A\
Image_B: .\入力画像フォルダパス\B\
  ~~~
Image_Z: .\入力画像フォルダパス\Z\

```
`key`の名前は変える必要なし。
### 使用上の注意
* 入力画像フォルダ、最適重み画像フォルダ内のラベルフォルダ(OKとNG)は手作業で作成する必要あり。
* `argo_workspace`フォルダに以降のフォルダ名はコード内で直接指定しているので、フォルダ名を同じにする必要あり（`run.bat`, `Image`フォルダ以降）。
* `argo_workspace`フォルダの`run.bat`内で`pause`コマンドを削除しないと動作がいちいち止まるので注意。
* `ngmasks`フォルダに入れるアノテーション画像の名前は`average_imageXXX.png`とする必要あり。
（参考コード：`cv2.imwrite('./○○/ngmasks/average_image' + str(i).zfill(3) + '.png', image)`, `image`変数には`cv2.imread`した`ndarray`が格納されているとする。）

###グローバル変数説明
| Global variables | Description |
| --- | --- |
| `YAML_DIR` | `yaml`ファイルのパス |
| `OUTPUT_DIR` | 最適重み画像または任意重み画像の出力先パス |
| `IMAGE_TYPE_NUM` | 画像条件数 |
| `RESOLUTION` | 分解能 |
| `LABEL_LIST` | 画像のラベル名に制限はないので`LABEL_LIST`内をいじればカスタマイズ可能。しかしOKととNGはマスト|
| `SPLIT` | 学習用OK画像の割合`(train:test = SPLIT:(1-SPLIT))` |
| `TOP_NUM` | 各重み画像の`evaluation_values.csv`からトップ`${TOP_NUM}`の`evaluation value`を持ってくる。グリッドサーチの回転数以内にしないと学習結果が全て消滅するので細心の注意を払って定義すること |
| `HEADER_ROWS` | `evaluation_values.csv`のヘッダー行数。現在は3行だが、アップデートとかでヘッダー行数が変わったときにはここをいじる|
| `HEADER_COLS` | `evaluation_values.csv`のヘッダー列数。現在は31列だが、アップデートとかでヘッダー列数が変わったときにはここをいじる|
| `CSV_OUTPUT_DIR` | 最終的に出力される`good_evaluation_values.csv`の出力先パス |
| `WEIGHT_LIST` | 通常関数で自動的に生成されるが、手動で設定したい時にも直接main文で定義できる|

