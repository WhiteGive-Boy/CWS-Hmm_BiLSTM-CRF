# CWS-Hmm_BiLSTM-CRF
## CWS中文分词 HMM BiLSTM+CRF pytorch 细致实现
数据集：人民日报 训练：测试=4：1 <br>
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
我运行的结果：<br>
```
BiLSTM+CRF
embedding_dim=100 hidden_dim=200 epoch=1 lr=0.005
precision:0.96975528
recall:   0.96779571
fscore:   0.96877451
HMM:
precision:0.87980196
recall:   0.84381022
fscore:   0.86143031
```
BiLSTM+CRF pytorch实现 参考 {ChineseNER}(https://github.com/buppt/ChineseNER)<br>
Hmm实现 参考 {HMM}(https://github.com/ldanduo/HMM)<br>

