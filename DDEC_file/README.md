單點計算，可以參考Bader charge的tutorial 
INCAR setting:
EDIFFG = -0.03   
NSW    =   0     
IBRION =      -1     
ISIF   =      2  
LCHARG =    T
LAECHG = .TRUE.(write files AECCAR0, AECCAR1, and AECCAR2)


需要改POTCAR 
請下載ddec-potcar.sh並放在執行的資料夾


DDEC-sbatch.sh 送job 的sbatch 檔
