# drawio-llm-bridge

draw.io（*.drawio.svg / mxfile）をLLMで扱いやすいJSONグラフに変換するスクリプトです．

## 使い方

1. 事前にPython 3環境を用意します（外部ライブラリ不要）．
2. 変換コマンドを実行します（`.drawio.svg` / `.drawio` 対応）:

   ```bash
   python3 drawio_struct_export_json_legend.py <input.drawio.svg> -o out.json [オプション]
   python3 drawio_struct_export_json_legend.py <input.drawio> -o out.json [オプション]
   ```

### 主なオプション

- `--diagram <name>`: 複数ページがある場合に対象のダイアグラム名を指定．
- `--include-legend`: 凡例のノード/エッジも出力に含める．
- `--keep-isolated`: 孤立ノードも削除せず出力する．
- `--no-rich-relations`: calls/reads/writes などの推論リレーション付与を無効化．
- `--debug`: 学習した凡例マップや検出情報を `debug` セクションに出力．

### 出力フォーマット

- `nodes`: `{"path/to/node": {"kind": "<種別>", "state": "<todo|wip|done>"?}, ...}`
  - ノードIDはラベル階層を `/` で結合したもの．状態は凡例のマーカー絵文字から自動抽出．
- `edges`: `[[src, rel, dst, inferred], ...]`
  - `inferred=false` は凡例ラベル/スタイルや明示ラベルから得た関係，`true` はヒューリスティクスによる推論．
- `warnings`: 凡例検出や学習の警告（空なら正常）．

### ヒント

- 凡例セクション（Legend/Node/Edge/Marker）があると，ノード種別・関係・ステータスを安定して学習できます．
- draw.io がポート用セルを作る場合でも，親のラベル付きセルに自動で解決されます．
