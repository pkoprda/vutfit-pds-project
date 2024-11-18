# PDS project

## Large log file analyser and detector of anomalies 

- **Author**: Peter Koprda (xkoprd00)

### How to install and run this tool:
Requirements:
- python3
- pip3

Install dependencies:
```
pip3 install -r requirements.txt
```
Run:
```
python3 log-monitor -training <file> -testing <file> [-contamination VAL] [-threshold VAL]
```
where:\
`-training <file>`: a dataset used to train the model\
`-testing <file>`: a data set used for testing the classification\
`-contamination VAL`: a contamination value for training an Isolation Forest model (interval (0, 0.5])\
`-threshold VAL`: a threshold value for testing the trained model

### Citation
- Jieming Zhu, Shilin He, Pinjia He, Jinyang Liu, Michael R. Lyu. [Loghub: A Large Collection of System Log Datasets for AI-driven Log Analytics](https://arxiv.org/abs/2008.06448). IEEE International Symposium on Software Reliability Engineering (ISSRE), 2023.
