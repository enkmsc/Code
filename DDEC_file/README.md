# DDEC6 Charge Analysis (VASP + Chargemol)

本教學使用 VASP 單點計算所得的電子密度，透過 Chargemol 09_26_2017 進行 DDEC6 atomic charge analysis。

**主要分析結果（DDEC charge data）：`DDEC6_even_tempered_net_atomic_charges.xyz`。此檔案包含每顆原子的 DDEC6 net atomic charge（單位：e）。**

## 1. VASP single-point calculation

使用最佳化後的結構作為 POSCAR，單點計算可參考 Bader charge 的計算流程。在原本的 INCAR 中確認以下設定：

```ini
NSW    = 0
IBRION = -1
ISIF   = 2
LCHARG = .TRUE.
LAECHG = .TRUE.
```

- LCHARG：輸出 CHGCAR。
- LAECHG：輸出 AECCAR0、AECCAR1 和 AECCAR2。
- NSW = 0：不進行結構最佳化。EDIFFG 在此不需要設定。

其餘參數（ENCUT、KPOINTS、ISPIN 等）沿用適合研究系統的收斂設定，確認電子自洽計算完成。

## 2. Prepare DDEC input files

建立獨立的 DDEC 計算資料夾，避免修改原始 VASP 計算的 POTCAR：

```bash
mkdir -p DDEC
cp CHGCAR AECCAR0 AECCAR2 POTCAR DDEC/
cd DDEC
```

下載本資料夾中的兩個 scripts：

```bash
wget https://raw.githubusercontent.com/enkmsc/Code/main/DDEC_file/ddec-potcar.sh
wget https://raw.githubusercontent.com/enkmsc/Code/main/DDEC_file/DDEC-sbatch.sh
chmod +x ddec-potcar.sh
```

也可以直接從 GitHub 下載 [ddec-potcar.sh](ddec-potcar.sh) 和 [DDEC-sbatch.sh](DDEC-sbatch.sh)。

## 3. Set up Chargemol

確認伺服器已安裝 Chargemol，且執行檔位於 PATH 中：

```bash
command -v Chargemol_09_26_2017_linux_serial
```

若找不到執行檔，需先設定 PATH 或請管理員提供安裝位置。程式可由 [Chargemol 官方網站](https://sourceforge.net/projects/ddec/files/) 取得。

本教學沿用實驗室既有的 Chargemol 環境設定，不另外建立 `job_control.txt`。若在其他環境執行，請確認 Chargemol 所需的控制設定已備妥。

送出計算前，資料夾應包含：

```text
DDEC/
├── CHGCAR
├── AECCAR0
├── AECCAR2
├── POTCAR
├── ddec-potcar.sh
└── DDEC-sbatch.sh
```

## 4. Submit DDEC calculation

```bash
sbatch DDEC-sbatch.sh
```

[DDEC-sbatch.sh](DDEC-sbatch.sh) 會自動執行以下步驟：

1. 將原始 POTCAR 備份為 POTCAR_calculate。
2. 執行 ddec-potcar.sh，轉換 POTCAR 格式。
3. 執行 Chargemol_09_26_2017_linux_serial。
4. 將 Chargemol 執行的起訖時間及耗時（秒）寫入 ddec_record_date。

### POTCAR conversion

[ddec-potcar.sh](ddec-potcar.sh) 是針對舊版 Chargemol 讀取 POTCAR 的格式相容性處理：

- 依 TITEL 分割 POTCAR。
- 從 VRHFIN 擷取元素名稱。
- 移除 SHA256、COPYR、LEXCH 所在行。
- 將 TITEL 簡化為 PAW 加上元素名稱。
- 合併暫存檔並覆寫 POTCAR。

此步驟僅用於 DDEC 分析，不是重新產生 VASP pseudopotential。不要在 VASP 計算前執行，也不需要在 sbatch 前手動執行，因為提交腳本已包含此步驟。

> 重新送出 DDEC job 前，務必從原始 VASP 資料夾重新複製未修改的 POTCAR，例如 `cp ../POTCAR ./POTCAR`（若使用上述資料夾結構）。否則腳本會用已轉換的 POTCAR 覆蓋備份。原始 POTCAR 請勿公開上傳。

### SLURM settings

目前的提交腳本使用 CPU partition、1 node、56 tasks、每 task 1 CPU、64 GB 總記憶體及 72 小時 walltime。請依伺服器設定調整。

注意：目前呼叫的是 serial executable，因此不會使用 56 個 tasks 進行平行計算。若維持 serial 版本，建議將 ntasks-per-node 改為 1；若需要平行計算，須另外設定 Chargemol 的 OpenMP parallel executable 與執行緒數。

## 5. Analyze DDEC results

| Output file | Description |
| --- | --- |
| DDEC6_even_tempered_net_atomic_charges.xyz | 每顆原子的 DDEC6 net atomic charge（e）。 |
| VASP_DDEC_analysis.output | Chargemol 分析紀錄，可用於檢查錯誤及計算是否完成。 |
| ddec_record_date | 執行時間紀錄。 |
| POTCAR_calculate | 送出計算時的 POTCAR 備份。 |

DDEC charge 為正代表相對於中性原子缺少電子，負值代表獲得電子。DDEC partial charge 不等同於 formal oxidation state。

確認輸出原子數與結構一致，並檢查所有原子電荷的總和是否接近 系統總電荷（net charge）（允許數值及輸出精度誤差）。單憑 ddec_record_date 有結束時間並不能確認計算成功，仍需檢查 Chargemol 的 log 與結果檔。

## References

- [Chargemol software and manual](https://sourceforge.net/projects/ddec/files/)
- [DDEC6 atomic population analysis: part 4](https://doi.org/10.1039/C7RA11829E)
- [VASP DDEC6 charge analysis example](https://github.com/Trebonius91/utils4VASP/wiki/DDEC6-Charges)
