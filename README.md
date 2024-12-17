# 2024ADL Final project

## 測試環境與套件
程式語言：Python 3.10.15<br>
套件：請參考requirements.txt<br>
作業系統：Windows 11 64bit
## 使用方式
### Setting:
設定好環境後。請先安裝套件：
```
pip install -r requirements.txt
```
如果torch要使用GPU版本，請再執行(以CUDA11.8為例):
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
再將main.py放到以下期末專案repo中，並執行step2之後的步驟：
https://github.com/appier-research/streambench-final-project
### Medical Diagnosis:
直接執行：<br>
```
python main.py --bench_name "classification_public"
```
### SQL Generation:
直接執行：<br>
```
python main.py --bench_name "sql_generation_public"
```
可使用參數：<br>
--output_path<br>
預測檔案輸出路徑<br><br>
--use_wandb<br>
使用Weight and Biases紀錄實驗(需先申請帳號並登入)<br><br>
--debug<br>
除錯模式，預設是使用前100個step，可依需求調整

## Reference:
此作業是以下repo的延伸：<br>
https://github.com/appier-research/streambench-final-project
