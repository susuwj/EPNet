## Efficient Edge-Preserving Multi-View Stereo Network for Depth Estimation (AAAI 2023 Oral) 

### Environment
- PyTorch 1.4.0
- Python 3.7
- open3d 0.9.0.0
- numpy 1.20.3

### Scripts
#### 1. train on DTU
```bash
bash scripts/train_dtu.sh
```
#### 2. test on DTU
```bash
bash scripts/test_dtu.sh
```
#### 3. finetune on BlendedMVS
```bash
bash scripts/train_blended.sh
```

#### 4. test on Tanks and Temple
```bash
bash scripts/test_tt.sh
```
#### 5. test on ETH3D
```bash
bash scripts/test_eth.sh
```

### Citation
```bibtex
@inproceedings{Su2023epnet,
  title={Efficient Edge-Preserving Multi-View Stereo Network for Depth Estimation},
  author={Su, Wanjuan and Tao, Wenbing},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2023},
  number={2},
  pages={2348--2356},
  number={37},
}
```

### Acknowledge
Our work is partially based on these opening source work: [Casmvsnet](https://github.com/alibaba/cascade-stereo) and [Vis-MVSNet](https://github.com/jzhangbs/Vis-MVSNet). Thanks for their contributions to the MVS community
