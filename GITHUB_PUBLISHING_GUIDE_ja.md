# GitHub公開ガイド（このプロジェクト向け）

この文書は、`ICI_predict_v2.py` を中心とした研究用スクリプト群を **GitHubで公開する前に何を準備すればよいか** を、実務ベースでわかりやすく整理したものです。

---

## まず結論

このプロジェクトを GitHub で公開するなら、最低限やることは次の5つです。

1. **README を置く**
2. **requirements.txt を置く**
3. **.gitignore を置く**
4. **公開してよいデータだけにする**
5. **ライセンスを決める**

研究ソフトとして見栄えをよくするなら、さらに次の2つがあると強いです。

- `CITATION.cff` を置く
- GitHub Releases を作る

---

## このコードを公開する前に、特に直した方がよい点

### 1. ローカル絶対パスを消す
現在のスクリプトには、`/Users/you/...` のようなローカルパス前提の設定があります。

例:
- `ICI_predict_v2.py` の `INPUT_FILES`, `OUTPUT_ROOT`
- `Compare_method_v2.py` の `NESTED_RESULTS_ROOT`, `COMPARE_OUTPUT_ROOT`
- `03_Calibration.py` の各種 root path

これが残ったままだと、他人がそのまま実行できません。

**おすすめ**
- 公開版では、`data/` や `outputs/` のような相対パスにする
- できれば将来的には `argparse` や `config.yaml` で設定できる形にする

### 2. データをそのまま上げない
このプロジェクトは医療・メタボロミクス系なので、**患者由来データや制限付きデータは原則そのまま公開しない** 方が安全です。

GitHub に上げるのは基本的に以下だけにすると安心です。

- Python スクリプト
- README
- requirements.txt
- .gitignore
- 共有可能な図や概要表
- 合成データまたはフォーマット例

### 3. 生成物を全部コミットしない
`.joblib`, `.tiff`, `.pdf`, 大きな `.xlsx`, 中間出力CSV などは、あとからどんどん増えます。

そのため、**成果物や中間ファイルは `.gitignore` に入れて、必要なものだけ手動で公開** するのがよいです。

### 4. 今のコードは「パッケージ」ではなく「研究用スクリプト集」
このコードは十分有用ですが、現状は

- Python ファイルを直接編集して使う
- スクリプトごとに設定欄を埋める

という形です。

そのため、README では **「すぐ pip install して使えるツール」ではなく、研究用解析スクリプト集である** と明記しておくのが親切です。

---

## 公開前チェックリスト

以下を上から順に確認すれば、かなり安全に公開できます。

### 必須
- [ ] README.md を置いた
- [ ] requirements.txt を置いた
- [ ] .gitignore を置いた
- [ ] 公開してよいファイルだけにした
- [ ] ローカルパスを見直した
- [ ] データ共有可否を確認した
- [ ] ライセンス方針を決めた

### あると良い
- [ ] `CITATION.cff` を置いた
- [ ] リポジトリ説明文を書いた
- [ ] トピック（topics）を付けた
- [ ] 最初の Release を作った
- [ ] Zenodo DOI を付けた

---

## GitHubで公開する手順

### いちばん簡単な流れ

#### 1. ローカルで公開用フォルダを整える
このフォルダに、少なくとも以下を入れます。

- `README.md`
- `requirements.txt`
- `.gitignore`
- `ICI_predict_v2.py`
- `Compare_method_v2.py`
- `03_Calibration.py`
- `04_Partial nest analysis.py`
- `05_Repeat holdout analysis.py`

#### 2. GitHubで新しい repository を作る
おすすめのリポジトリ名:

- `metabolomics-ici-ml-pipeline`
- `ici-metabolomics-nested-cv`
- `metabolomics-response-prediction`

説明文の例:

> Research scripts for nested CV, non-nested CV, holdout comparison, calibration, and sensitivity analyses in metabolomics-based ICI response prediction.

公開するなら **Public** を選びます。

**注意**
すでにローカルに README や .gitignore を作ってある場合は、GitHub側で新規 repo を作るときに余計な初期化をしない方が扱いやすいです。

#### 3. ローカルから push する

```bash
git init
git add .
git commit -m "Initial public release"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY-NAME.git
git push -u origin main
```

#### 4. GitHub上で説明を整える
公開後に以下を入れると見栄えがかなり良くなります。

- Description
- Topics
- License
- Releases

---

## トピック（topics）のおすすめ

この研究内容なら、たとえば以下が使いやすいです。

- `metabolomics`
- `machine-learning`
- `nested-cross-validation`
- `biomarker-discovery`
- `logistic-regression`
- `python`
- `immunotherapy`
- `omics`

---

## ライセンスはどうするか

### 迷ったら考えること

- **他人に自由に使ってほしい** → MIT / BSD 系
- **改変版も同じ条件で公開してほしい** → GPL 系
- **共同研究や所属先ルールがある** → 先に所属先確認

### 重要
ライセンスを書かないと、基本的には「他人は再利用しにくい」状態になります。

---

## 研究ソフトとしては `CITATION.cff` がかなりおすすめ

論文や学会で使う可能性があるなら、`CITATION.cff` を置くと GitHub 上で **“Cite this repository”** が出せます。

今回、テンプレートとして `CITATION.cff.template` を用意しています。必要事項を埋めて `CITATION.cff` に変更すれば使えます。

---

## Release は作った方がよい？

はい。研究用途ではかなり有用です。

最初の公開時に、たとえば次のようなタグを付けるとわかりやすいです。

- `v1.0.0`
- `v2026.04`

Release ノートには、最低限以下を書くと十分です。

- 何を含む版か
- どのスクリプトが含まれるか
- 動作確認した Python バージョン
- データは含まないこと

---

## Zenodo DOI を付けるべき？

研究者なら、かなりおすすめです。

GitHub の公開リポジトリを Zenodo と連携すると、**Release ごとに DOI を発行** できます。論文の Methods / Code Availability に書きやすくなります。

---

## このプロジェクトに対して、公開時におすすめの最小構成

```text
metabolomics-ici-ml-pipeline/
├── README.md
├── requirements.txt
├── .gitignore
├── ICI_predict_v2.py
├── Compare_method_v2.py
├── 03_Calibration.py
├── 04_Partial nest analysis.py
├── 05_Repeat holdout analysis.py
├── CITATION.cff            # 任意だが推奨
└── LICENSE                 # 公開するなら推奨
```

---

## 逆に、最初は入れない方がよいもの

- 生データ
- 匿名化が不十分な表
- 患者識別につながる情報
- 容量の大きい中間生成物
- 非公開データから作った最終モデル
- 個人PCの作業ログ

---

## ひとことアドバイス

このプロジェクトは、内容としては十分に公開価値があります。
ただし、**「そのまま公開」より「公開用に少し整えてから出す」方が、他人に伝わりやすく、再利用もされやすい** です。

今回用意した README / requirements / .gitignore / CITATION テンプレートを土台にすると、かなりスムーズに公開できます。
