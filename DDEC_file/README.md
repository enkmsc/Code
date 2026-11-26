# DDEC6 Charge Analysis (VASP + Chargemol)

**主要分析結果（DDEC charge data）：`DDEC6_even_tempered_net_atomic_charges.xyz`。此檔案包含每顆原子的 DDEC6 net atomic charge（單位：e）。**

## 1. VASP single-point calculation
使用最佳化後的結構作為 POSCAR，單點計算可參考 Bader charge 的計算流程。在原本的 INCAR 中確認以下設定：
```ini
EDIFFG = -0.03
NSW  = 0
IBRION =   -1
ISIF =   2
LCHARG =  T
LAECHG = .TRUE.(write files AECCAR0, AECCAR1, and AECCAR2)
```
- LCHARG：輸出 CHGCAR。
- LAECHG：輸出 AECCAR0、AECCAR1 和 AECCAR2。
- NSW = 0：不進行結構最佳化。EDIFFG 在此不需要設定。
其餘參數（ENCUT、KPOINTS、ISPIN 等）沿用結構最佳化時系統的收斂設定，確認電子自洽計算完成。

## 2. Prepare DDEC input files
下載本資料夾中的兩個 scripts：
```bash
wget https://raw.githubusercontent.com/enkmsc/Code/main/DDEC_file/ddec-potcar.sh #自動改POTCAR 的檔案
wget https://raw.githubusercontent.com/enkmsc/Code/main/DDEC_file/DDEC-sbatch.sh #送Job的`sbatch`檔
chmod +x ddec-potcar.sh 
chmod +x DDEC-sbatch
```
可以直接從 GitHub 下載 [ddec-potcar.sh](ddec-potcar.sh) 和 [DDEC-sbatch.sh](DDEC-sbatch.sh) 

## 3. Submit DDEC calculation
```bash
sbatch DDEC-sbatch.sh
```
[DDEC-sbatch.sh](DDEC-sbatch.sh) 會自動執行以下步驟：
1. 將原始 POTCAR 備份為 POTCAR_calculate。
2. 執行 ddec-potcar.sh，轉換 POTCAR 格式。
3. 執行 Chargemol_09_26_2017_linux_serial。
