<div align="center">
  <h2><b> Code for Paper:</b></h2>
  <h2><b> Moving Beyond Functional Connectivity: Time-Series Modeling for fMRI-Based Brain Disorder Classification </b></h2>
</div>

## Get Started

1. Install requirements. ```pip install -r requirements.txt```
2. Download data. You can download all the datasets from [**datasets**](https://drive.google.com/u/0/uc?id=1EtxBoOulKMCJ8y6Zh5GtxH56pOYHDlD0&export=download). **All the datasets are well pre-processed** and can be used easily.
3. Train the model. We provide the experiment scripts of all benchmarks under the folder `./scripts`. 
4. You can use bash commands to individually run scripts in the 'scripts' folder from the command line to obtain results for individual datasets, for example, you can use the below command line to obtain the result of DeCI on TaoWu:
   
      ```bash ./MTSF/96or12/ETTh1.sh ```

You can find:

The training history and results under './logs' folder.

The test results (MSE and MAE) under './test_dicts' folder.

The visualization of forecasting results under './visual/forecasting' folder.

The visualization of learning  representation (Such as ***Global Token***) under './visual/feature' folder.

The visualization of the weight under './visual/weight' folder.  **PS: The method used for getting the weight follows the approach outlined in [An Analysis of Linear Time Series Forecasting Models](https://arxiv.org/abs/2403.14587).**

## Several examples for better understanding the proposed method in the paper.

Within the storage limits of a GitHub repository, we provide training log files of tasks **input-96-predict-720** on **ECL&Traffic**, to assist readers in reproducing the results from the paper. 
