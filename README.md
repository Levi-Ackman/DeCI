<div align="center">
  <h2><b> Code for Paper:</b></h2>
  <h2><b> Moving Beyond Functional Connectivity: Time-Series Modeling for fMRI-Based Brain Disorder Classification (IEEE TMI) </b></h2>
</div>

<div align="center">

**[<a href="https://arxiv.org/abs/2602.08262">Paper</a>]**
</div>

## Usage

1. Install requirements. ```pip install -r requirements.txt```
2. Download data. You can download all the datasets from [**datasets**](https://drive.google.com/u/0/uc?id=1EtxBoOulKMCJ8y6Zh5GtxH56pOYHDlD0&export=download). **All the datasets are well pre-processed** and can be used easily. Then place them under a folder `./dataset`.
3. Train the model. We provide the experiment scripts of all benchmarks under the folder `./scripts`. 
4. You can use bash commands to individually run scripts in the 'scripts' folder from the command line to obtain results for individual datasets. For example, you can use the command below to obtain the result of DeCI on TaoWu:
   
      ```bash scripts/DeCI/Taowu.sh ```

You can find the training history and results under the './logs' folder.

Meanwhile, the `scripts` folder contains all the execution scripts for our **DeCI** model, as well as scripts for **FC-based methods** (under the `FC` folder), **dFC-based methods** (under the `dFC` folder), **general time-series models** (under the `GeneralTS` folder), **Multi-View-based methods** (under the `Multi_View` folder), and **Attention-based methods** (under the `Attn` folder). 

To reproduce the full set of DeCI results reported in the paper, you can run:

```
python hrun.py --opt 1
```

To run all the FC-based baselines, use:

```
python hrun.py --opt 2
```

To run all the dFC-based baselines, use:

```
python hrun.py --opt 3
```

For general time-series methods, use:

```
python hrun.py --opt 4
```

For Multi-view based methods, use:

```
python hrun.py --opt 5
```

For Attention-based methods, use:

```
python hrun.py --opt 6
```

Once the experiments are complete, you can run:

```
python extract_re.py
```

This script will automatically aggregate and organize the logs, generating the final performance tables based on the best hyperparameter configurations.

## Acknowledgment

We appreciate the following GitHub repos a lot for their valuable code and efforts:

- Time-Series-Library (https://github.com/thuml/Time-Series-Library)
- ModernTCN (https://github.com/luodhhh/ModernTCN)
- Leddam (https://github.com/Levi-Ackman/Leddam)
- Jiaxing Xu *et al.* (https://github.com/brainnetuoa/data_driven_network_neuroscience)

## Citation
If you find this repo helpful, please cite our paper.

```
@article{Yu2026DeCI,
  title        = {Moving Beyond Functional Connectivity: Time-Series Modeling for fMRI-Based Brain Disorder Classification},
  author       = {Yu, Guoqi and Hu, Xiaowei and Aviles-Rivero, Angelica I. and Qiu, Anqi and Wang, Shujun},
  journal      = {arXiv preprint arXiv:2602.08262},
  year         = {2026},
  url          = {https://arxiv.org/abs/2602.08262},
  doi          = {10.48550/arXiv.2602.08262}
}
```

