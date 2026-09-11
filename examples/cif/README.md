# Crystal Examples

対称性の教材として選んだ結晶。**見覚えのある物質**で、**原子数が少なく**、
対称操作を動かして見せる価値があるものだけを収録している。

32 の結晶類を 1 件ずつ網羅していた旧セット（大きな胞や馴染みのない鉱物を含む）は
`tests/fixtures/cif/` へ移した。あちらはテストのカバレッジ用で、UI には出ない。

| ファイル | 空間群 | 原子 | 見どころ |
|---|---|---|---|
| `Halite.cif` | 225 Fm-3m | 8 | 岩塩型。最も基本的な立方晶 |
| `CsCl.cif` | 221 Pm-3m | 2 | 2 原子だけの単純立方 |
| `Copper.cif` | 225 Fm-3m | 4 | 面心立方最密充填 |
| `Iron-alpha.cif` | 229 Im-3m | 2 | 体心立方。体心の並進が単独で見える |
| `Iron-gamma.cif` | 225 Fm-3m | 4 | γ鉄（オーステナイト）。α鉄のBCCとの比較 |
| `Magnesium.cif` | 194 P6_3/mmc | 2 | 六方最密。6₃ らせん軸 |
| `Diamond.cif` | 227 Fd-3m | 8 | d 映進とらせんが最も豊富 |
| `Graphite.cif` | 194 P6_3/mmc | 4 | Mg と同じ空間群を層状構造で |
| `Sphalerite.cif` | 216 F-43m | 8 | 閃亜鉛鉱。反転中心を持たない |
| `Zincite.cif` | 186 P6_3mc | 4 | ウルツ鉱。極性 6mm、c 映進 |
| `Fluorite.cif` | 225 Fm-3m | 12 | 1 つの空間群に 2 種類のサイト対称 |
| `Rutile.cif` | 136 P4_2/mnm | 6 | 4₂ らせんと n 映進が 6 原子で見える |
| `Anatase.cif` | 141 I4_1/amd | 12 | TiO₂の別多形。ルチルとの結晶構造比較 |
| `Perovskite.cif` | 221 Pm-3m | 5 | 理想ペロブスカイト（SrTiO3） |
| `Quartz.cif` | 152 P3_121 | 9 | 掌性。3₁ らせん、鏡映なし |
| `Pyrite.cif` | 205 Pa-3 | 12 | 立方晶だが 4 回軸を持たない m-3 |
| `BaTiO3.cif` | 160 R3m | 5 | R 格子。菱面体↔六方の設定切替のデモ |

## 追加・変更のしかた

`Halite.cif` と `BaTiO3.cif` 以外は **生成物**。手で編集せず、表を直して再生成する。

```bash
.venv/bin/python tools/generate_example_structures.py             # 書き出す
.venv/bin/python tools/generate_example_structures.py --check-only # 検証だけ
.venv/bin/python tools/regenerate_example_assets.py --clean        # JSON とカタログを更新
```

出典・格子定数・Wyckoff 位置は `tools/generate_example_structures.py` の
`CRYSTALS` 表にある。生成後に空間群・原子数・組成を読み直して照合するので、
値の転記ミスはその場で落ちる。

**ファイル名がそのまま表示名とエクスポートのスラグになる。** 空白や括弧を入れない
（旧セットの `Gd I S.cif` は `gd_i_s.json` という読みにくいスラグになっていた）。

`Halite.cif` と `BaTiO3.cif` を再生成してはいけない理由:

- `Halite.cif` は `tests/test_structure_analysis.py` が `generation_operation_index`
  の実値列を固定している。原子や操作の順序が変わると落ちる。
- `BaTiO3.cif` は同梱 CIF で唯一 `_symmetry_equiv_pos_as_xyz` ループが空で、
  `structure_analysis.py` の菱面体フォールバック経路を実データで踏む唯一のサンプル。

`tools/generate_example_structures.py` の `PROTECTED` がこの 2 件を守っている。
