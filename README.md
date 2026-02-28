<div align="center">
  <h2><b> Code for Paper:</b></h2>
  <h2><b> Moving Beyond Functional Connectivity: Time-Series Modeling for fMRI-Based Brain Disorder Classification (IEEE TMI) </b></h2>
</div>

<div align="center">

**[<a href="https://arxiv.org/abs/2602.08262">Paper</a>]**
**[<a href="http://xhslink.com/o/8xZBw4JYPbB">中文解读</a>]**

</div>

## Introduction
🌟 Although FC remains the dominant fMRI input, we find that GCNs/Transformers perform better when fed tokenized raw BOLD time series. Shuffling BOLD temporal order while preserving FC reduces performance to the FC baseline.

<img width="1134" height="407" alt="image" src="https://github.com/user-attachments/assets/470687ea-e3e4-4f63-8e12-62851ab593cc" />

🏆 We benchmark several recent time-series models (*e.g.*, [**Leddam**](https://arxiv.org/abs/2402.12694), [**iTransformer**](https://arxiv.org/abs/2310.06625); ***the red boxes below indicate time-series–based methods***) and find that they generally outperform traditional FC/dFC methods.

<p align="center">
<img width="837" height="372" alt="image" src="https://github.com/user-attachments/assets/5a0976af-1932-4bdf-8025-a6cd9e2f71f9" />
</p>

🌟 This motivates us to explore how recent advances in time-series analysis can benefit fMRI modeling. Building on two key principles in modern time-series research, **Channel-Independence (CI)** and **Seasonal–Trend Decomposition**, we propose **DeCI**, which performs deep **cycle (seasonal)** and **drift (trend)** decomposition via progressive residual extraction, models each ROI time series in a **CI** manner, and fuses predictions at the **logit** level.

<p align="center">
<img width="1705" height="662" alt="image" src="https://github.com/user-attachments/assets/d74d45cf-0209-4c0c-9a1a-9648f949a3ca" />
</p>

🏆 DeCI (Channel-Independent) is more **noise-robust** than other Channel-Dependent baselines (*e.g.*, **iTransformer**).

<p align="center">
<img width="829" height="327" alt="image" src="https://github.com/user-attachments/assets/d18abc07-2659-4391-af67-0832208d7c9a" />
</p>

🌟 **Seasonal–Trend Decomposition (or Cycle–Drift Decomposition for fMRI) substantially enhances the discriminability of raw features.**

<p align="center">
<img width="834" height="577" alt="image" src="https://github.com/user-attachments/assets/76e505ea-9105-4e73-8eac-7f900f3a12aa" />
</p>

🏆 **DeCI achieves strong performance with low computational overhead.**

<p align="center">
<img width="850" height="458" alt="image" src="https://github.com/user-attachments/assets/198538fb-b49c-4347-bfc4-4d447afa5c83" />
</p>

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

