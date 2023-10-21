# NHDE
Code for NeurIPS2023 Paper: [Neural Multi-Objective Combinatorial Optimization with Diversity Enhancement](https://github.com/bill-cjb/NHDE)

**Quick Start For NHDE-P**

- To train a model, such as MOTSP with 20 nodes, run *train_motsp_n20.py* in the corresponding folder.
- To test a model, such as MOTSP with 20 nodes, run *test_motsp_n20.py* in the corresponding folder.
- Pretrained models for each problem can be found in the *result* folder.

**Quick Start For NHDE-M**

- To train a model, such as MOTSP with 20 nodes, set *TSP_SIZE=20* and *MODE=1* in *HYPER_PARAMS.py*, and then run *run.py* in the corresponding folder.
- To test a model, such as MOTSP with 20 nodes, set *TSP_SIZE=20* and *MODE=2* in *HYPER_PARAMS.py*, and then run *run.py* in the corresponding folder.
- Pretrained models for each problem can be found in the *result* folder.

**Reference**

If our work is helpful for your research, please cite our paper:
```
@inproceedings{chen2023neural,
  title={Neural Multi-Objective Combinatorial Optimization with Diversity Enhancement},
  author={Chen, Jinbiao and Zhang, Zizhen and Cao, Zhiguang and Wu, Yaoxin and Ma, Yining and Ye, Te and Wang, Jiahai},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023},
}
```

