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
BiLSTM+CRF<br>
embedding_dim=100 hidden_dim=200 epoch=1 lr=0.005<br>
precision:0.96975528<br>
recall:   0.96779571<br>
fscore:   0.96877451<br>
HMM:<br>
precision:0.87980196<br>
recall:   0.84381022<br>
fscore:   0.86143031<br>

BiLSTM+CRF pytorch实现 参考 {ChineseNER}(https://github.com/buppt/ChineseNER)<br>
Hmm实现 参考 {HMM}(https://github.com/ldanduo/HMM)<br>

