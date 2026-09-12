# symmetry-viewer

結晶・分子の対称性を、3D で動かして学ぶための教材です。

- **クイズモード**: 回転軸・対称操作・操作の合成・原子の移り先・点群を、
  構造を回して確かめながら答えます。結晶学の用語を知らない初学者向けです。
- **解析モード**: 構造が持つ対称操作をすべて一覧し、1つずつアニメーションで
  再生します。対称要素（回転軸・鏡映面・対称中心など）も表示します。

結晶17種・分子21種を収録しています。手元の CIF（結晶）や XYZ（分子）の
ファイルを読み込んで調べることもできます。

## ダウンロード

[Releases](../../releases/latest) から、使っているパソコンに合わせて1つ選んでください。
Python などのインストールは要りません。

| パソコン | ファイル |
|---|---|
| Windows 10 / 11（64ビット） | `symmetry-viewer-windows-x64.zip` |
| Linux（x86_64、glibc 2.27 以降） | `symmetry-viewer-linux-x86_64.tar.gz` |

どちらも WebGL が使えるブラウザ（Edge・Chrome・Firefox など）で表示します。

## Windows で使う

1. ダウンロードした zip を右クリックして「**すべて展開**」を選びます。
   zip の中を直接開いて起動すると動きません。
2. 展開したフォルダの `symmetry-viewer.exe` をダブルクリックします。
3. 「Windows によって PC が保護されました」と出たら、「**詳細情報**」を押して
   「**実行**」を押します。発行元の電子署名が付いていないアプリに出る警告です。
4. 「アプリケーションがポリシーによってこのファイルをブロックしました」と出て
   起動しない場合は、Windows 11 の「**スマート アプリ コントロール**」が有効です
   （買ったばかりの PC や Windows を入れ直した PC で有効なことがあります）。
   「設定」→「プライバシーとセキュリティ」→「Windows セキュリティ」→
   「アプリとブラウザー制御」→「スマート アプリ コントロール」を「オフ」に
   すると起動できます。**一度オフにすると Windows を入れ直すまで元に戻せない**
   ため、オフにしたくない場合は Linux 版か、下の「ソースから動かす」を
   ご検討ください。
5. 黒いウィンドウが開き、ブラウザでモード選択画面が表示されます。

**終了**: 黒いウィンドウを閉じます。**削除**: フォルダごと削除します
（ほかの場所には何もインストールしません）。

## Linux で使う

```bash
tar -xzf symmetry-viewer-linux-x86_64.tar.gz
cd symmetry-viewer
./symmetry-viewer
```

**終了**: 端末で Ctrl+C を押します。**削除**: フォルダごと削除します。

## 収録している構造について

結晶の格子定数・原子位置は文献値に基づいています。出典は
[`examples/cif/README.md`](examples/cif/README.md) と
[`tools/generate_example_structures.py`](tools/generate_example_structures.py) の
`source` 欄にあります。`Halite.cif`・`BaTiO3.cif` は ReciPro の結晶データベース
（元の登録は COD・AMCSD）から書き出したもので、元論文の情報をファイル内に
残しています。

## ソースから動かす・配布物を作る

Python 3.11 以上がある Linux / macOS なら、ソースからそのまま動かせます。

```bash
scripts/setup.sh
scripts/serve.sh --mode puzzle
```

配布物は GitHub Actions（`.github/workflows/build.yml`）が作ります。手元で作る
場合は `packaging/build_linux.sh`（Linux）または `packaging/build_windows.ps1`
（Windows）を実行してください。

## ライセンス

MIT ライセンスです（[LICENSE](LICENSE)）。配布物に同梱しているライブラリ
（numpy・scipy・pymatgen・spglib・Three.js など）のライセンス文書は、展開した
フォルダの `_internal/licenses/` にあります。
