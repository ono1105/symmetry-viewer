# Molecule XYZ Samples

対称性の教材として選んだ分子。教科書に出る分子から、点群がなるべく重ならないように
選んでいる。座標は理想化した幾何で、実測構造でも最適化構造でもない。

| ファイル | 点群 | 原子 | 見どころ |
| --- | --- | --- | --- |
| `hydrogen_chloride.xyz` | C∞v | 2 | 直線・極性 |
| `carbon_dioxide.xyz` | D∞h | 3 | 直線・反転あり |
| `water.xyz` | C2v | 3 | 最も基本的な折れ線 |
| `ammonia.xyz` | C3v | 4 | 三角錐 |
| `hydrogen_peroxide.xyz` | C2 | 4 | 回転のみ。鏡映も反転も無い |
| `boron_trifluoride.xyz` | D3h | 4 | 平面三角形 |
| `methane.xyz` | Td | 5 | 正四面体 |
| `chlorofluoromethane.xyz` | Cs | 5 | 鏡映が 1 枚だけ |
| `bromochlorofluoromethane.xyz` | C1 | 5 | 対称性なし・掌性（クイズには出ない） |
| `xenon_tetrafluoride.xyz` | D4h | 5 | 平面四角形 |
| `ethene.xyz` | D2h | 6 | 平面・直交する 3 本の C2 |
| `trans_dichloroethene.xyz` | C2h | 6 | C2 と σh、その積の反転中心 |
| `phosphorus_pentafluoride.xyz` | D3h | 6 | 三方両錐（平面 BF3 との対比） |
| `bromine_pentafluoride.xyz` | C4v | 6 | 四角錐 |
| `allene.xyz` | D2d | 7 | ねじれた二重結合 |
| `sulfur_hexafluoride.xyz` | Oh | 7 | 正八面体 |
| `ethane.xyz` | D3d | 8 | ねじれ形（重なり形なら D3h） |
| `anti_dibromodichloroethane.xyz` | Ci | 8 | 反転中心だけを持つ |
| `benzene.xyz` | D6h | 12 | 主軸が S6 と S3 の両方を担う |
| `al12w_icosahedron.xyz` | Ih | 13 | 5 回軸を持つ最小の物体（クイズには出ない） |
| `mackay_icosahedron.xyz` | Ih | 54 | Mackay 二十面体。中心は空位（クイズには出ない） |

## 追加・変更のしかた

`water` `ammonia` `methane` `benzene` `ethene` `allene` `boron_trifluoride`
`sulfur_hexafluoride` `xenon_tetrafluoride` `carbon_dioxide` `hydrogen_chloride`
は手書き。それ以外は `tools/generate_example_structures.py` の `MOLECULES` 表からの生成物。

```bash
.venv/bin/python tools/generate_example_structures.py
.venv/bin/python tools/regenerate_example_assets.py --clean
```

**手書きの XYZ は黙って過剰対称化する。** `PointGroupAnalyzer` の許容誤差が 0.3 Å と
緩いので、少しずれた幾何が上位の点群として報告される（trans-ジクロロエチレンを
作り損ねたとき C2h ではなく D2h と判定された）。生成側の検証は 0.3 Å と 0.05 Å の
両方で同じ点群になることを要求している。手で座標を書くときも同じ確認をすること。

## 正 20 面体クラスターについて

`al12w_icosahedron.xyz` と `mackay_icosahedron.xyz` は**準結晶そのものではなく、
準結晶の構成単位**。この区別は曖昧にしない。

- 準結晶は並進周期を持たないので**単位胞も空間群も無く**、このアプリの結晶経路には
  そもそも載せられない（`analyze_cif` は格子を必須で読み、spglib は何を渡しても
  何らかの空間群を返す。二十面体を箱に入れると `Pm-3` や `P-1` という嘘が出る）
- 一方**有限のクラスターは 5 回軸を持ってよい**。「5 回対称は存在するが、それで
  空間を周期的に埋めることはできない」——これが準結晶の話の入口で、
  分子として解析すればこのアプリでも正しく `Ih`・120 操作として出る

**この 2 件はクイズに出ない。** 回答の選択肢が 2/3/4/6 回に閉じていて 5 回軸に
名前を付けられないため、出題すると C2 と C3 だけを問うことになり、
「この物体に 5 回軸は無い」と教えてしまう。除外はカタログの
`beyond_quiz_vocabulary`（`crystal_viewer/game/catalog.py` が計算）で自動的に決まる。
