# CWS-Hmm_BiLSTM-CRF
## CWS中文分词 HMM BiLSTM+CRF pytorch 细致实现
数据集：人民日报<br>
数据处理：<br>
```
cd data
python data_u.py
```
HMM：<br>
```
cd HMM
python hmm_model.py
```
BiLSTM+CRF：<br>
```
cd BiLSTM-CRF
python train.py
```

BiLSTM+CRF pytorch实现 参考 https://github.com/buppt/ChineseNER<br>
Hmm实现 参考 https://github.com/ldanduo/HMM<br>

