# Instructions-for-performing-imitation-learning-on-the-DOBOT-E6-using-LeRobot
PS4 コントローラで E6 を操作して「お手本」を記録し、そのお手本から AI（ACT）に動きを学習させて、最後に E6 が自分で吸着ピック＆プレースを行うようにすることが目的です。  この手順書を最後まで進めると、次のことができるようになります。  PS4 コントローラで E6 を操作しながら、カメラ映像と動きをデータセットとして記録する Google Colab の GPU で模倣学習モデル（ACT）を学習する 学習したモデルで E6 を自動で動かす

# LeRobot で DOBOT E6 に模倣学習をさせる手順書

## 目的

PS4 コントローラで **E6 を操作して「お手本」を記録**し、そのお手本から **AI（ACT）に動きを学習**させて、最後に **E6 が自分で吸着ピック＆プレースを行う**ようにすることが目的です。

この手順書を最後まで進めると、次のことができるようになります。

- PS4 コントローラで E6 を操作しながら、カメラ映像と動きをデータセットとして記録する
- Google Colab の GPU で模倣学習モデル（ACT）を学習する
- 学習したモデルで E6 を自動で動かす

> 📌 **前提**：「PS4 コントローラで DOBOT E6 を操作する手順書」で、E6 を PS4 コントローラで動かせていること。
>
> 📌 LeRobot は SO-101 など専用アーム向けのソフトで、E6 には標準で対応していません。そのため、この手順書では **E6 用のプラグイン（自作）** を使います。

---

## 全体の流れ

```
 ┌──────────── Windows PC ────────────┐        ┌──── Google Colab ────┐
 │                                     │        │                       │
 │ ① 記録（lerobot-record）            │        │ ② 学習（lerobot-train）│
 │   PS4 で操作 ＋ USB カメラ          │ ─────▶ │   GPU（T4）で ACT 学習  │
 │   → データセット                    │  Hub   │   → 学習済みモデル      │
 │                                     │ ◀───── │                       │
 │ ③ 自動運転（lerobot-rollout）        │  Hub   └───────────────────────┘
 │   学習済みモデルで E6 を動かす      │
 └─────────────┬───────────────────────┘
               │ LAN（192.168.5.1）
               ▼
           DOBOT E6
```

データセットと学習済みモデルの受け渡しには **Hugging Face Hub**（AI モデル・データの共有サイト）を使います。

### ROS2（Docker）環境との違い

| | ROS2 手順書の環境 | この手順書（LeRobot） |
|---|---|---|
| 動かす場所 | Docker コンテナ（Ubuntu） | **Windows に直接** |
| E6 との通信 | ROS2 の bringup 経由 | LeRobot プラグインが **直接** TCP/IP で通信 |

> ⚠️ **E6 の制御ポートは、同時に1つのプログラムしか使えません。** LeRobot を使う前に、ROS2 の bringup、PS4 操作プログラム（`e6_ps4_teleop.py`）、DobotStudio Pro を**必ず終了**してください。Docker は `docker compose down` で止めておくと確実です。

---

## 動作の仕組み（データの中身）

| 種類 | 内容 |
|---|---|
| 観測（ロボットの状態） | 先端の位置 `x.pos` `y.pos` `z.pos`（mm）、回転 `rz.pos`（度）、吸引 `suction.pos`（0/1） |
| 観測（画像） | USB カメラの映像（`front`） |
| 動作（AI が出力するもの） | **目標の位置** `x.pos` `y.pos` `z.pos`（mm）、`rz.pos`（度）、吸引 `suction.pos`（0/1） |

- AI は「次にどの座標へ行くか」を直接出力します。物体の高さ（Z）などを**座標で覚える**ので、正確に届きやすくなります。
- 安全のため、AI が遠くの座標を出しても、1ステップで動くのは最大 10mm（回転は 5度）までで、作業範囲の外には出ません。

> ⚠️ **プラグイン 0.1.x（動作を「移動量 `x.vel` など」で記録する旧方式）で記録したデータは、今のプラグイン（0.2.0）では学習・自動運転に使えません。** 新しい名前（例：`e6_pick_place_pos`）で記録し直してください。

- 吸着ハンドは**下向きで固定**（RX=180、RY=0）し、X / Y / Z と先端の回転（RZ）だけを動かします。
- 記録・学習・自動運転は、すべて **1秒に10回（10fps）** で行います。**この数値は全工程で揃えてください。**

---

## 準備するもの

| 項目 | 内容 |
|---|---|
| PC | Windows 10/11（GPU は不要） |
| ロボット | DOBOT Magician E6 ＋ 吸着ハンド（末端 DO に接続） |
| カメラ | USB Web カメラ 1台（作業台全体が映る位置に固定） |
| コントローラ | PS4 コントローラ（USB 接続がおすすめ） |
| アカウント | **Hugging Face**（無料）、**Google**（Colab 用） |
| ファイル | `lerobot_e6.zip`（プラグイン2つ＋Colab ノートブック） |

---

## STEP 1　Python 環境（Miniforge）を用意する

LeRobot は **Python 3.12** が必要です。今の Python（3.11）とは別に、専用の環境を作ります。

1. 「Miniforge」で検索し、GitHub の conda-forge/miniforge から **Windows 用インストーラ**（`Miniforge3-Windows-x86_64.exe`）をダウンロードしてインストールします（設定はすべて初期値のままで OK）。
2. スタートメニューから **「Miniforge Prompt」** を起動します。
3. 次を **1行ずつ** 実行します。

```bat
conda create -y -n lerobot python=3.12
conda activate lerobot
conda install -y ffmpeg -c conda-forge
pip install "lerobot[core_scripts]==0.6.1"
```

プロンプトの先頭が `(lerobot)` になっていれば、専用環境に入っています。

> 💡 **この手順書のコマンドは、すべて「Miniforge Prompt」で `conda activate lerobot` を実行してから入力します。** 次回以降も、Miniforge Prompt を開いたら最初に `conda activate lerobot` を実行してください。

---

## STEP 2　E6 用プラグインを入れる

1. `lerobot_e6.zip` を **`C:\e6_lerobot`** に展開します。

```
C:\e6_lerobot\
├── lerobot_robot_dobot_e6\          … E6 を LeRobot から動かすプラグイン
├── lerobot_teleoperator_ps4_e6\     … PS4 コントローラ用プラグイン
└── E6_ACT_train_colab.ipynb         … Colab 学習用ノートブック
```

2. Miniforge Prompt で実行します。

```bat
cd C:\e6_lerobot
pip install -e .\lerobot_robot_dobot_e6
pip install -e .\lerobot_teleoperator_ps4_e6
```

`Successfully installed lerobot_robot_dobot_e6-0.2.0 ...` と表示されれば OK です。

---

## STEP 3　Hugging Face にログインする

1. https://huggingface.co でアカウントを作成します。
2. 「Settings」→「Access Tokens」→「Create new token」で、種類を **Write** にしてトークンを作成し、コピーします。
3. Miniforge Prompt で実行し、トークンを貼り付けます（画面には表示されません）。

```bat
hf auth login
hf auth whoami
```

`hf auth whoami` で表示された**ユーザー名**を、以降のコマンドで使います。毎回入力しなくて済むように、変数に入れておきます（`あなたのユーザー名` を置き換え）。

```bat
set HF_USER=あなたのユーザー名
```

> 💡 `set HF_USER=...` は Miniforge Prompt を開き直すたびに必要です。

---

## STEP 4　カメラを確認する

USB カメラを PC につなぎ、**作業台の上から全体が映る位置にしっかり固定**します（記録後にカメラを動かすと、学習した内容が使えなくなります）。

```bat
lerobot-find-cameras opencv
```

見つかったカメラの一覧と番号（`Id: 0` など）が表示されます。ノート PC の内蔵カメラが `0` の場合、USB カメラは `1` などになります。**この手順書では `index_or_path: 2` と書いています（USB カメラが2番の場合）。番号が違う場合は置き換えてください。**

撮影されたテスト画像は `outputs\captured_images` フォルダに保存されるので、映り方を確認できます。

---

## STEP 5　E6 との通信を確認する

### ① DobotStudio Pro で座標系を 0 にする（最初に1回）

このプラグインは、位置（X / Y / Z / RZ）を **ユーザー座標系 0・ツール座標系 0** で扱います。別の番号が選ばれていると、開始位置や作業範囲がずれて危険です。

1. DobotStudio Pro で E6 に接続します。
2. ジョグ操作画面などにある座標系の選択で、次のように設定します。

| 項目 | 設定 |
|---|---|
| ユーザー座標系（User） | **0** |
| ツール座標系（Tool） | **0** |

3. 制御モードが **TCP/IP** になっていることも確認します。
4. 確認が終わったら、DobotStudio Pro を**切断して終了**します（接続したままだと LeRobot から操作できません）。

> 💡 DobotStudio Pro の画面に表示される座標が、この手順書の X / Y / Z（例：開始位置 X300 Y70 Z200）と同じ基準になります。

### ② 通信を確認する

> ⚠️ ROS2 の bringup・PS4 操作プログラム・DobotStudio Pro が**終了していること**を確認してください。E6 は TCP/IP モードのまま、Windows の固定 IP（192.168.5.10）も今まで通りです。

#### 状態を読むだけ（動きません）

```bat
python -m lerobot_robot_dobot_e6.check
```

```
接続OK
RobotMode : 4 （3:電源OFF 4:未イネーブル 5:イネーブル 9:エラー）
GetPose   : [300.0, 70.0, 200.0, 180.0, 0.0, -90.0]
```

のように表示されれば通信 OK です。

#### 実際に動かす（イネーブル → 開始位置 → Z を 20mm 上下）

```bat
python -m lerobot_robot_dobot_e6.check --move
```

電源 OFF の場合は自動で電源 ON（約20秒）→ イネーブル → 開始位置（X300 Y70 Z200 RZ-90）へ移動 → 20mm 上がって戻る、という動きをします。

---

## STEP 6　作業範囲（安全範囲）を決める

AI が予想外の動きをしても危なくないように、**アームが動ける範囲（箱）** を決めておきます。特に **Z の下限**は、吸着パッドが机に当たらない高さにする必要があります。

### ① PS4 で操作してみる

```bat
lerobot-teleoperate --robot.type=dobot_e6 --robot.ip=192.168.5.1 --robot.id=e6 --teleop.type=ps4_e6 --teleop.id=pad --fps=10
```

> ⚠️ **`--fps=10` を必ず付けてください。** 付け忘れると 1秒に60回動作指令が出て、想定より速く動きます。

| 入力 | 動作 |
|---|---|
| 右スティック 縦 / 横 | X / Y（右に倒すと Y−） |
| 左スティック 縦 / 横 | Z（上下）/ RZ（先端の回転）※横は**大きく倒したときだけ**反応 |
| △ / × | 吸引 ON / OFF |
| □ | 開始位置（X300 Y70 Z200 RZ-90）へ戻る |
| ○ | アラームクリア＋イネーブル ON（非常停止や衝突検知で止まったとき） |

> 💡 左スティック左右（RZ）は、**真横に大きく（約85%以上）倒したときだけ**動きます。縦（Z）に倒しているときは RZ は動きません。
>
> | 調整したいとき | コマンドに付けるもの |
> |---|---|
> | もっと反応しにくくする | `--teleop.rz_deadzone=0.95`（0〜1、大きいほど反応しにくい） |
> | RZ を一切動かさない | `--teleop.enable_rz=false` |

`Ctrl+C` で終了します。

### ② 物を吸着できる高さを調べる

ゆっくり下げて、**物を吸着できる高さ**まで来たら `Ctrl+C` で止め、位置を確認します。

```bat
python -m lerobot_robot_dobot_e6.check
```

`GetPose` の3番目の数値が Z の高さです。初期値の Z 下限は **60mm** です。吸着できる高さに合わせて、以降のコマンドに `--robot.z_min=（吸着できる高さ − 5）` を付けます。（開始位置の Z（200mm）より上にはしないでください）

### ③ 範囲の設定一覧

以降のコマンド（`lerobot-teleoperate` / `lerobot-record` / `lerobot-rollout`）に付けて変更できます。

| 設定 | 初期値 | 内容 |
|---|---|---|
| `--robot.x_min` / `--robot.x_max` | 150 / 400 | X の範囲（mm） |
| `--robot.y_min` / `--robot.y_max` | -250 / 250 | Y の範囲（mm） |
| `--robot.z_min` / `--robot.z_max` | **60** / 350 | Z の範囲（mm）⚠ 必ず調整 |
| `--robot.speed_ratio` | 30 | 速度（%） |
| `--robot.suction_do_index` | 1 | 吸着ポンプの末端 DO 番号 |
| `--robot.home_x` / `home_y` / `home_z` / `home_rz` | 300 / 70 / 200 / -90 | 開始位置（接続時と □ ボタンで移動） |

> 💡 範囲は記録・学習・自動運転で**同じ値**にしてください。

---

## STEP 7　お手本を記録する（lerobot-record）

### 記録のコツ

| ポイント | 内容 |
|---|---|
| 回数 | **50 回**が目安（最初は 10 回で流れを確認しても OK） |
| 物の置き場所 | 置き場所を少しずつ変える（**1か所あたり約10回**） |
| 動かし方 | **毎回同じ手順**で、ゆっくり・なめらかに |
| 環境 | カメラ位置・照明・背景を変えない |
| 開始位置 | 毎回、開始位置（X300 Y70 Z200）から始める。リセット中に **□ ボタン**を押すと戻れます |

### 記録コマンド

`--robot.z_min` は STEP 6 で決めた値に変えてください。

```bat
lerobot-record ^
  --robot.type=dobot_e6 --robot.ip=192.168.5.1 --robot.id=e6 --robot.z_min=60 ^
  --robot.cameras="{ front: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}" ^
  --teleop.type=ps4_e6 --teleop.id=pad ^
  --dataset.repo_id=%HF_USER%/e6_pick_place ^
  --dataset.single_task="Pick up the object and place it in the box" ^
  --dataset.fps=10 ^
  --dataset.num_episodes=50 ^
  --dataset.episode_time_s=30 ^
  --dataset.reset_time_s=20 ^
  --dataset.no_stamp=true ^
  --display_data=true
```

> 💡 行末の `^` は「次の行に続く」という意味です（Miniforge Prompt 用）。

### 記録中の流れ

```
 開始位置へ移動
      │
      ▼
 ┌─ 記録（30秒）── PS4 で ピック → プレース
 │       │
 │       ▼
 │  リセット（20秒）── □ ボタンで開始位置へ戻る ／ 物を元の位置へ戻す
 │       │
 └───────┘ × 50 回
      │
      ▼
 動画を保存 → Hugging Face Hub へ自動アップロード
```

### 記録中のキー操作（PC のキーボード）

**`lerobot-record` を実行した Miniforge Prompt の画面をクリックして選んだ状態**で、キーを押します。

| キー | 同じ働きのキー | 動作 |
|---|---|---|
| **→**（右矢印） | **N** | 今の記録を早めに終わらせて次へ |
| **←**（左矢印） | **R** | 今の記録をやり直す（失敗したとき） |
| **Esc** | **Q** | 記録を終了して保存・アップロード |

> 💡 Windows では、キー入力は**どの画面を選んでいても受け付けます**（カメラ映像の確認画面を選んでいても OK）。そのため、記録中に別のアプリで文字を打つと、N・R・Q や矢印キーに反応してしまいます。**記録中は他のアプリでキーボードを使わない**でください。

> 💡 作業が早く終わったら **→** を押して次に進めます。失敗したお手本は **←** でやり直してください（失敗したお手本も AI は真似してしまいます）。

### 記録の中断と再開

途中で止めた場合は、`--resume=true` を付け、`--dataset.num_episodes` を **追加したい回数** にして同じコマンドを実行します。

---

## STEP 8　Colab で学習する

1. https://colab.research.google.com を開き、「ファイル」→「ノートブックをアップロード」で **`E6_ACT_train_colab.ipynb`** を選びます。
2. 「ランタイム」→「ランタイムのタイプを変更」→ **T4 GPU** を選んで保存します。
3. ノートブックの説明に沿って、**上のセルから順番に ▶ を押して実行**します。

| セル | 内容 |
|---|---|
| 1 | GPU の確認 |
| 2 | LeRobot のインストール |
| 3 | Hugging Face ログイン（STEP 3 と同じトークン） |
| 4 | Google ドライブをつなぐ（途中で切れても再開できるように） |
| 5 | 設定（データセット名 `e6_pick_place`、学習ステップ数など）※ユーザー名は入れない |
| 6 | **学習**（1〜数時間） |
| 7 | 途中で切れたときの再開 |
| 7-2 | 学習の延長（STEP 10 ④） |
| 8 | アップロードの確認 |

学習が終わると、モデルが **`ユーザー名/act_e6_pick_place`** として Hugging Face Hub にアップロードされます。

> ⚠️ 無料版の Colab は、長時間使うと途中で切断されることがあります。チェックポイントは Google ドライブに 5000 ステップごとに保存されるので、切れたら 1〜5 を実行してから **7** で再開してください。
>
> 💡 学習の進み具合はログの `loss` で確認できます。数値が下がって落ち着いてくれば学習が進んでいます。

---

## STEP 9　学習したモデルで E6 を自動で動かす

> ⚠️ **安全確認**：アームの周囲に人がいないこと、**非常停止スイッチをすぐ押せる**ことを確認してください。最初は `--robot.speed_ratio=20` など低速で試してください。

物を、お手本で使った置き場所のどれかに置いてから実行します。

```bat
lerobot-rollout ^
  --strategy.type=base ^
  --policy.path=%HF_USER%/act_e6_pick_place ^
  --robot.type=dobot_e6 --robot.ip=192.168.5.1 --robot.id=e6 --robot.z_min=60 ^
  --robot.cameras="{ front: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}" ^
  --task="Pick up the object and place it in the box" ^
  --duration=60 ^
  --fps=10 ^
  --device=cpu ^
  --policy.n_action_steps=10
```

| 項目 | 内容 |
|---|---|
| `--policy.path` | 学習済みモデル（Hub の名前） |
| `--policy.n_action_steps=10` | AI が何ステップごとにカメラを見て予測し直すか（1〜100）。小さいほどずれに強くなりますが、計算が遅い PC では動きがカクカクします |
| `--task` | 記録時の `--dataset.single_task` と**同じ文章** |
| `--duration` | 自動で動かす時間（秒） |
| `--fps=10` | 記録と**同じ値**にする |
| `--device=cpu` | GPU がない PC でも動きます |

終了するときは `Ctrl+C` を押します。終了時、アームは開始時の位置へゆっくり戻ってから停止します。

---

## STEP 10　やり直す・追加で学習する

### どれを使えばよいか

| やりたいこと | 方法 |
|---|---|
| お手本を全部取り直したい | ① 記録を最初からやり直す |
| 学習だけ最初からやり直したい | ② 学習を最初からやり直す |
| お手本を足して賢くしたい | ③ お手本を追加して学習する |
| 同じお手本で、もっと長く学習したい | ④ 学習を延長する |

---

> 💡 この STEP のコマンドにある `あなたのユーザー名` は、STEP 3 の `hf auth whoami` で表示されたユーザー名に置き換えてください。

### ① 記録を最初からやり直す

**方法A：新しい名前で記録する（おすすめ）**

`--dataset.repo_id` の名前を変えて記録するだけです。前のデータは残るので、あとで比べられます。

```bat
--dataset.repo_id=あなたのユーザー名/e6_pick_place_v2 ^
```

Colab では、セル5 の `DATASET_NAME` を `"e6_pick_place_v2"` に、`POLICY_NAME` を `"act_e6_pick_place_v2"` に変えて学習します。

**方法B：同じ名前で記録し直す**

1. PC に保存されたデータを削除します（Miniforge Prompt）。

```bat
rmdir /s /q "%USERPROFILE%\.cache\huggingface\lerobot\あなたのユーザー名\e6_pick_place"
```

2. Hugging Face 上のデータも削除します。https://huggingface.co でデータセットのページを開き、「Settings」→ 一番下の「Delete this dataset」で削除します。
3. STEP 7 の記録コマンドを、そのまま実行します。

> ⚠️ 削除したデータは元に戻せません。

---

### ② 学習を最初からやり直す

Colab のノートブックで、次の順に実行します。

| 順番 | セル | 内容 |
|---|---|---|
| 1 | 1〜5 | いつもどおり準備 |
| 2 | **5-2** | 前回の学習結果を削除 |
| 3 | 6 | 学習 |

> 💡 前回の学習結果を残したい場合は、5-2 を実行せず、セル5 の `POLICY_NAME` を `"act_e6_pick_place_v2"` のように別の名前にしてから 6 を実行します。

---

### ③ お手本を追加して学習する

**1. お手本を追加で記録する**

STEP 7 の記録コマンドに `--resume=true` を付け、`--dataset.num_episodes` を**追加したい回数**にして実行します。`--dataset.repo_id` は前回と**同じ名前**にします。

```bat
lerobot-record ^
  --robot.type=dobot_e6 --robot.ip=192.168.5.1 --robot.id=e6 --robot.z_min=60 ^
  --robot.cameras="{ front: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}" ^
  --teleop.type=ps4_e6 --teleop.id=pad ^
  --dataset.repo_id=あなたのユーザー名/e6_pick_place ^
  --dataset.single_task="Pick up the object and place it in the box" ^
  --dataset.fps=10 ^
  --dataset.num_episodes=20 ^
  --dataset.episode_time_s=30 ^
  --dataset.reset_time_s=20 ^
  --display_data=true ^
  --resume=true
```

> ⚠️ カメラの位置・開始位置・`--dataset.single_task` の文章・`--dataset.fps` は、**前回と同じ**にしてください。
>
> 💡 前回、日時付きの名前（例：`e6_pick_place_20261020_141836`）で記録した場合は、`--dataset.repo_id` もその名前にします。

記録を **Esc**（または **Q**）で終了すると、前回分＋追加分が Hugging Face にアップロードされます。

**2. Colab で、増えたお手本を使って最初から学習する**

お手本が増えたら、② と同じ手順で学習し直します（1〜5 → 5-2 → 6）。前回のモデルを残したい場合は、`POLICY_NAME` を別の名前にします。

> 💡 お手本を足したときは、前回の続きからではなく**最初から学習し直す**のが確実です。

---

### ④ 学習を延長する（同じお手本で、もっと長く学習する）

学習が最後まで終わったあとで、動きがまだ不安定なときに使います。

1. Colab でセル 1〜5 を実行します。
2. セル **7-2** の `TOTAL_STEPS` に、**今までの分を含めた合計のステップ数**を入れて実行します。

```python
TOTAL_STEPS = 50000   # 例：30000 まで学習済み → さらに 20000 学習する
```

学習が終わると、新しいモデルが同じ名前（`あなたのユーザー名/act_e6_pick_place`）で Hugging Face に上書きされます。STEP 9 のコマンドはそのまま使えます。

> ⚠️ 延長できるのは、Google ドライブに前回の学習結果（`OUTPUT_DIR`）が残っている場合だけです。5-2 で削除した場合は ② で最初から学習します。

---

## うまくいかないときは（学習の改善）

| 症状 | 対策 |
|---|---|
| 物のところへ行かない・迷う | お手本の数を増やす（50→100回）。置き場所ごとに約10回ずつ |
| 動きがぎこちない・止まる | お手本の動かし方を毎回そろえる。学習ステップ数を増やす（30000→50000） |
| 吸着のタイミングがずれる | 吸着する前に一瞬止まるなど、お手本のタイミングを毎回同じにする |
| 学習時と結果が違う | カメラ位置・照明・背景が記録時と同じか確認する |

---

## 困ったときは（トラブルシューティング）

| 表示・症状 | 原因 | 対処 |
|---|---|---|
| `'lerobot-record' は認識されていません` | 環境に入っていない | Miniforge Prompt で `conda activate lerobot` |
| `ConnectionRefusedError` / `timed out`（E6） | E6 につながらない | 固定 IP・LAN ケーブル・TCP/IP モードを確認 |
| 接続が途中で切れる／`ErrorID` が出る | 他のプログラムが E6 を使っている | ROS2 bringup・`e6_ps4_teleop.py`・DobotStudio Pro を終了 |
| `イネーブルがタイムアウトしました` | 非常停止・アラーム | 非常停止を解除し、DobotStudio Pro でアラームを確認・解除 |
| `PS4 コントローラが見つかりません` | コントローラ未接続 | USB を挿し直す。`ps4_server.py` など他のプログラムを終了 |
| `ServoP エラー` が続く | 範囲外・特異点付近 | 作業範囲（`--robot.x_min` など）を狭くする |
| アームが速すぎる | fps の指定忘れ | `--fps=10`（teleoperate / rollout）、`--dataset.fps=10`（record） |
| カメラの映像が出ない | 番号が違う | `lerobot-find-cameras opencv` で番号を確認 |
| `Invalid username or password`（Hub） | ログインしていない | `hf auth login`（Write 権限のトークン） |

---

## 毎回の作業手順（まとめ）

| 順番 | 作業 |
|---|---|
| 1 | E6 の電源 ON、TCP/IP モード、LAN 接続を確認 |
| 2 | ROS2 bringup・DobotStudio Pro が**終了している**ことを確認 |
| 3 | Miniforge Prompt を開く → `conda activate lerobot` → `set HF_USER=ユーザー名` |
| 4 | 記録：`lerobot-record ...`（STEP 7） |
| 5 | 学習：Colab でノートブックを実行（STEP 8） |
| 6 | 自動運転：`lerobot-rollout ...`（STEP 9） |

---

## 最後に

- このサンプルは、基本**無料**で実施できます。
- **商用利用可能**です。
  - ※ LeRobot は Apache License 2.0 です。Google Colab・Hugging Face の無料枠については、各サービスの利用規約に従ってください。
- ロボットは、必ず**開始位置に毎回戻してから**記録してください。
- 記録は次の流れで行うと、うまくいきます。
  1. 動作が終わったら **N キー**（記録を終了）
  2. キューブとロボットを開始位置に戻す（ロボットは **□ ボタン**）
  3. 戻せたら **N キー**（次の記録へ）
- 試しに実施したところ、**2時間半**かかりました。
- サンプルプログラム（有償）を希望される方は、**haruki-k9614@afrel.co.jp** までお問い合わせください。
- DOBOT Magician E6 はこちらで購入できます：[株式会社アフレル DOBOT ラインナップ](https://afrel.co.jp/product/dobot-education/lineup/#pythonset-b)

---

## 参考資料

- [LeRobot ドキュメント](https://huggingface.co/docs/lerobot/en)
- [LeRobot：インストール](https://huggingface.co/docs/lerobot/installation)
- [LeRobot：独自ハードウェアの組み込み](https://huggingface.co/docs/lerobot/integrate_hardware)
- [LeRobot：実機での模倣学習](https://huggingface.co/docs/lerobot/il_robots)

---

## 免責事項

本記事・本ソフトウェアの使用によって生じた、データの消失、利益の損失、デバイスの故障（正常に起動しなくなる等）、またはその他の直接的、間接的、特別、偶発的、あるいは派生的な損害について、著作者は一切の責任を負いません。

すべての操作は、利用者ご自身の責任において行ってください。
