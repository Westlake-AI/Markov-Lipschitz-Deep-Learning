

# Markov-Lipschitz Deep Learning (MLDL)

## Summary

This is the PyTorch code for the following paper:

[
Stan Z. Li, Zelin Zang, Lirong Wu, "Markov-Lipschitz Deep Learning", arXiv preprint, arXiv:2006.08256, 2020.
](https://arxiv.org/abs/2006.08256)

Main features of MLDL in comparison with some others are summarized below:
<img src='./figs/MLDL_Features.png'>

The code includes the following modules:
* Datasets (Swiss rool, S-Curve, MNIST, Spheres)
* Training for ML-Enc and ML-AE (ML-Enc + ML-Dec)
* Test for manifold learning (ML-Enc) 
* Test for manifold generation (ML-Dec) 
* Visualization
* Evaluation metrics 

## Requirements

* pytorch == 1.3.1
* scipy == 1.4.1
* numpy == 1.18.5
* scikit-learn == 0.21.3
* csv == 1.0
* matplotlib == 3.1.1
* imageio == 2.6.0

## Running the code

1. Clone this repository

  ```
  git clone https://github.com/westlake-cairi/westlake-cairi-MAE_TESTREBAR/tree/New_Codes
  ```

2. Install the required dependency packages

3. To get the results for 10 seeds, run

  ```
  python autotrain.py
  ```

4. To get the metrics for ML-Enc and ML-AE, run

  ```
  python eval.py -M ML-Enc
  python eval.py -M ML-AE
  ```
The evaluation metrics are available in `./pic/indicators.csv`

5. To test the generalization to unseen data, run
  ```
  python main.py -M Test
  ```
The results are available in `./pic/Epoch_10000_test.png`

5. To test the manifold generation, run
  ```
  python main.py -M Generation
  ```
The results are available in `./pic/Generation.png`

## Results

1. Visualization of embeddings
* Swiss Roll and S-Curve <img src='./figs/swiss roll.png'>
* MNIST and Spheres <img src='./figs/mnist+spheres.png'>


2. Comparison of embedding quality for Swiss Roll (800 points)

   |        | #Succ | L-KL   | RRE      | Trust  | LGD     | K-Min | K-Max   | MPE    |
   | ------ | ----- | ------ | -------- | ------ | ------- | ----- | ------- | ------ |
   | ML-Enc | 10    | 0.0184 | 0.000414 | 0.9999 | 0.00385 | 1.00  | 2.14    | 0.0262 |
   | MLLE   | 6     | 0.1251 | 0.030702 | 0.9455 | 0.04534 | 7.37  | 238.74  | 0.1709 |
   | HLLE   | 6     | 0.1297 | 0.034619 | 0.9388 | 0.04542 | 7.44  | 218.38  | 0.0978 |
   | LTSA   | 6     | 0.1296 | 0.034933 | 0.9385 | 0.04542 | 7.44  | 215.93  | 0.0964 |
   | ISOMAP | 6     | 0.0234 | 0.009650 | 0.9827 | 0.02376 | 1.11  | 34.35   | 0.0429 |
   | t-SNE  | 0     | 0.0450 | 0.006108 | 0.9987 | 3.40665 | 11.1  | 1097.62 | 0.1071 |
   | LLE    | 0     | 0.1775 | 0.014249 | 0.9753 | 0.04671 | 6.17  | 451.58  | 0.1400 |

   

3. Performance metrics for the ML-AE with Swiss Roll (800 points) data

   |        | #Succ | L-KL    | RRE     | Trust  | Trust   | K-min | K-max   | MPE     | MRE     |
   | ------ | ----- | ------- | ------- | ------ | ------- | ----- | ------- | ------- | ------- |
   | ML-AE  | 10    | 0.00165 | 0.00070 | 0.9998 | 0.00514 | 1.01  | 2.54    | 0.04309 | 0.01846 |
   | AE     | 0     | 0.11537 | 0.13589 | 0.7742 | 0.03069 | 1.82  | 5985.74 | 0.01519 | 0.40685 |
   | VAE    | 0     | 0.23253 | 0.49784 | 0.5053 | 0.04000 | 1.49  | 5290.55 | 0.01977 | 0.78104 |
   | TopoAE | 0     | 0.05793 | 0.04891 | 0.9265 | 0.09651 | 1.10  | 228.11  | 0.12049 | 0.56013 |




4. The process of manifold data reconstruction and generation using ML-AE

<img src='./figs/generation.PNG'>



5. Generalization testing

<img src='./figs/generalization.PNG'>



6. The GIF for the training process, and as you can see, the the manifold unfolds smoothly

<img src='./figs/latent.gif'>

## Citation

If you use this code, please cite the following:

```bibtex
@article{MarLip-v1-2020,
  title={Markov-Lipschitz Deep Learning},
  author={Stan Z Li and Zelin Zhang and Lirong Wu},
  journal={arXiv preprint arXiv:2006.08256},
  year={2020}
}
```
