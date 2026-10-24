# MACE committee：大界面軌跡的離線力分歧評估

用數個已訓練的 MACE 對**相同完整 snapshots**計算力，找出適合後續局部裁切的中心。一次只載入一個模型，處理完抽取的所有幀才載入下一個；不會同時把三個模型放進 GPU，也不會重新跑 MD。這個工具不訓練模型、不裁切/鬆弛結構、不提交 VASP 工作。

## 1. 準備模型

至少兩個獨立訓練的 MACE，建議先用三個測試。若已有一個，可用同一 foundation、固定相同的 train/valid/test 檔案與訓練設定，另外使用不同 `--seed` 訓練兩個。不同 seed 仍可能高度相關；低分歧不保證正確。

每次訓練使用不同工作目錄、name、model_dir、checkpoints_dir，避免 `--restart_latest` 讀到其他成員的 checkpoint。不要在每個 seed 重新隨機切分資料。使用未轉換成 LAMMPS/TorchScript 的原生 `.model`，並保持同一 DFT 標準及模型 head。工具以 SHA256 拒絕完全相同的模型複本。

將模型放入 `models/`，或在 `settings.py` 填入實際的絕對路徑。程式不下載 foundation model，也不假裝已經有其他 committee 成員。

## 2. 修改設定

直接編輯 `settings.py`，不需要很長的命令列：

```python
MODEL_PATHS = ["/your/path/seed13.model", "/your/path/seed42.model", "/your/path/seed256.model"]
INPUT_FILE = "/your/path/dump.lammpstrj"
INPUT_FORMAT = "lammps-dump-text"
TYPE_MAP = {1: "Li", 2: "La", 3: "Zr", 4: "Co", 5: "O"}
MAX_FRAMES = 10
FRAME_STRIDE = 100
OUTPUT_DIR = "results/committee_test"
```

`TYPE_MAP` 範例對應上傳的 `LCO_LLZO_(2).data`；**必須確認你的實際 dump 與 LAMMPS `pair_coeff` 元素順序**，程式無法只從整數 type 推斷正確元素。不要直接套到 LNO/LIC。

- `START_FRAME` 從 0 起算；`STOP_FRAME` 不含上限；`FRAME_STRIDE` 的單位是**儲存幀數**，不是 timestep 或 ps。例：dump 每 100 步、stride=100，就每 10000 MD 步評估一次。
- 預設只取 10 幀，先量測成本。`MAX_FRAMES=None` 才處理所有選中幀。
- 支援單一完整 text dump（也可以是已串接檔案）、extxyz、POSCAR (`vasp`)、LAMMPS data (`lammps-data`)。
- text dump 必須包含 `id type` 和 `x y z`、`xu yu zu`、`xs ys zs` 或 `xsu ysu zsu`。假設 LAMMPS `units metal`，長度 Å；其他單位需事先轉換。
- 支援 orthogonal 與 restricted triclinic（`xy xz yz`）；保留每幀晶胞、PBC、原始 timestep、非連續 atom ID 和 box origin。拒絕 general triclinic (`abc origin`)。
- 不會把 timestep 當累積時間；串接/重設 timestep 的重複幀也不自動刪除，使用 `source_frame` 區分。
- POSCAR 沒有 LAMMPS ID 時會產生 1-based 行號，並明確標示 `generated_row_1based`，不能當成原始 dump ID。
- `CENTER_ELEMENTS` 可限制候選中心元素；完整結構的力仍照算。`MIN_SCORE=None` 只排序，不宣稱超過誤差門檻。
- 相對路徑均相對於此目錄。每次請改用新 `OUTPUT_DIR`；既有資料夾會被拒絕，以免混入舊結果。

## 3. 執行

在原本可用的 MACE 環境執行；依賴 `mace-torch`、PyTorch、ASE、NumPy（ASE 會帶入 SciPy）。不必為本工具升級原本環境。

```bash
cd /data01/tian_02/Code
git pull --ff-only origin main
cd Cut_interface/mace_committee
```

設定完成後，在 Gibbs 提交：

```bash
sbatch run_gibbs.sh
```

job 使用 repository 現有的 Gibbs GPU4090 / mace-mliap 環境路徑。若你的環境位置不同，請改 job。CPU 小測試可在 settings 設 `DEVICE="cpu"`，於已啟用環境直接跑 `python evaluate_committee.py`。

不建議在 login node 評估大結構。每次輸出含逐幀時間；第一幀可能包含初始化成本。`peak_cuda_allocated_MiB` 是 PyTorch allocated peak，含模型/計算張量，不等於整張 GPU 的 nvidia-smi 用量。載入模型時間另存於 `run.json`。

## 4. 分數定義

對第 i 顆原子與 M 個模型：

```text
Fmean[i] = mean_m(F[m, i])
std_xyz[i, a] = sqrt(mean_m((F[m, i, a] - Fmean[i, a])**2))
force_std[i] = sqrt(std_x**2 + std_y**2 + std_z**2)
```

使用 population variance (`ddof=0`)，單位 eV/Å。這是**力向量分歧**，不是先取力大小再算標準差，也不是每分量 RMS（兩者差 sqrt(3)）。使用 Welford 累積，因此不需保留 M 份全軌跡力在記憶體中。

`local_mean` = 中心原子及 `LOCAL_RADIUS` 內其他唯一原子的 `force_std` 算術平均。考慮週期影像，同一鄰居不重複計入。這是明確定義的區域排序指標，沒有直接複製任何論文的校準誤差公式。4 Å 只是可修改的起點，**不是裁切邊長，也不是 MACE 力計算所需的 buffer**。

`RANK_BY` 可選 `force_std` 或 `local_mean`。每幀依分數遞減挑選中心，並以最小影像距離避免候選中心太靠近；不做跨時間結構去重。尚未指定真正的界面空間範圍，因此預設搜尋整個輸入結構。

## 5. 輸出

| 檔案 | 用途 |
|---|---|
| `uncertainty.extxyz` | 每個選中完整 snapshot，含 `atom_id`、`force_std`、`local_mean`、平均力、分量標準差、`candidate_center`；可用 ASE/OVITO 檢視 |
| `candidates.csv` | 所有幀的候選中心依分數排序，含來源幀、timestep、atom ID、元素、座標、兩種分數與原幀檔案路徑 |
| `summary.csv` | 每幀最大/平均/p95 力分歧、能量分歧、候選數 |
| `timing_model_*.csv` | 各模型逐幀耗時、原子數、GPU peak 與總能量 |
| `frames/*.extxyz` | 抽取的完整原始幀，供後續裁切，保留原子順序/ID |
| `frames/*.npz` | 各幀累積的 mean/M2/模型能量，供重現與檢查；目前沒有續跑介面 |
| `run.json` | 設定、版本、模型 SHA256、來源檔案大小/修改時間、完成狀態與耗時 |

工具從 `get_forces()` / `get_potential_energy()` 逐模型取值，不依賴不同 MACE 版本可能不同的 committee `energies` 欄位。原結構的約束不套用到力上，以免固定原子的力被清零。只載入信任的模型檔。

這些分數**不是 DFT 誤差，也不是 MTP extrapolation grade**，不要套用 1.7 門檻。先用有 DFT 力標籤的獨立驗證資料檢查分數與誤差關係；選樣時也保留低分歧對照。若同一 foundation 造成成員高度相關，可能共同出錯。

候選中心可用於下一步裁切，但完整 snapshot 的力分歧不保證裁切後小週期結構的物理合理性。需另行驗證邊界、組成/電中性、核心環境和局部鬆弛；本工具不會自動宣稱輸出已能送 DFT。

## 6. 測試

```bash
python -m pytest -q test_committee.py
```

測試使用人工指定的力，驗證統計公式、完整資料流程、NPT 晶胞、triclinic/scaled/unwrapped 座標、atom ID、PBC 鄰域與候選間距。這不等於真實 MACE/GPU 測速；真實測速需你自己的模型與軌跡。

參考：
- MACE calculator: https://github.com/ACEsuit/mace/blob/main/mace/calculators/mace.py
- Spatially Resolved Uncertainties for Machine Learning Potentials: https://doi.org/10.1021/acs.jcim.4c00904
