#!/bin/bash 
set -e

base_path=$(pwd)

mkdir -p md_sample uncertainty_fps pre_relax dft_sp convert fine_tune


#fine_tune 會經過4round or until0 第一次是粗的train from 一些表面和bulk surface interface 
mkdir -p bootstrap #第0 round 裡面會有四個model 
mkdir -p AL_1
mkdir -p AL_2
mkdir -p AL_3


#md_sample #這邊先寫到md我等等整理我之前的code 
mkdir -p md_0 #第0 round會從bootstrap 裡面先找一個最好的model複製到
cp md_sample/inputs/* md_0 
#然後這裡我應該要開始從裡面取些幀開始做uncertainty 的評估if uncertainty 超過多少即開始

mkdir -p md_1
mkdir -p md_2
mkdir -p md_3

#uncertainty_FPS_select #這個部分可以先寫一些function py code
#所以這裡面還有cut.py 


#rough_relax #這邊我會給你inputs 



#single_point  #這邊我會給你inputs



#data_convert #這邊我會給一些to_data.py 之類的轉檔或許



#refine(一樣是fine_tuned這個資料夾所以是一個loop)
