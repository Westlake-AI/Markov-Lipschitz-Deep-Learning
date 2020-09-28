
# Markov-Lipschitz Deep Learning (MLDL)


<p align="center">
<br>
</p>

<p align="center">
  <img src='./figs/MLDL.jpg' width="800">
</p>

<br>
<p align="center">
<img src='./figs/ML-AE.gif' width="270"  alt="ML-AE Result">
<a href="https://github.com/BorgwardtLab/topological-autoencoders/blob/master/animations/topoae.gif">
        <img src='./figs/topoae.gif' width="270" alt="TopoAE Result">
</a>
<a href="https://github.com/BorgwardtLab/topological-autoencoders/blob/master/animations/vanilla.gif">
        <img src='./figs/vanilla.gif' width="270" alt="Vanilla AE Result">
</a>
<br>Comparison of training processes of three autoencoders on the Spheres dataset.
</p>

<br>

This is a PyTorch implementation of the 
  [
  MLDL paper
  ](https://arxiv.org/abs/2006.08256)
:
```bibtex
@article{Li-MLDL-2020,
  title={Markov-Lipschitz Deep Learning},
  author={Stan Z Li and Zelin Zang and Lirong Wu},
  journal={arXiv preprint arXiv:2006.08256},
  year={2020}
}
```

The main features of MLDL for manifold learning and generation in comparison to other popular methods are summarized below:

|                                               | MLDL (ours)		| AE/TopoAE 	| MLLE 	| ISOMAP 	| t-SNE	|
| :----------------------------------------- 	| :--------:		| :-------: 	| :---: | :----: 	| :---: |
| Manifold Learning without decoder          	|    Yes     		|    No     	| Yes  	|   Yes   	| Yes  	|
| Learned NLDR model applicable to test data 	|    Yes     		|    Yes    	|  No  	|   No   	|  No  	|
| Able to generate data of learned manifold  	|    Yes     		|    No     	|  No  	|   No   	|  No  	|
| Compatible with other DL frameworks           |    Yes     		|    No     	|  No  	|   No   	|  No  	|	
| Scalable to large datasets                    |    Yes     		|    Yes     	|  No  	|   No   	|  No  	|




The code includes the following modules:
* Datasets (Swiss Roll, S-Curve, MNIST, Spheres)
* Training for ML-Enc and ML-AE (ML-Enc + ML-Dec)
* Test for manifold learning (ML-Enc) 
* Test for manifold generation (ML-Dec) 
* Visualization
* Evaluation metrics 
* The compared methods include: AutoEncoder (AE), <a href="https://github.com/BorgwardtLab/topological-autoencoders">Topological AutoEncoder (TopoAE)</a>, [Modified Locally Linear Embedding (MLLE)](https://github.com/scikit-learn/scikit-learn), [ISOMAP](https://github.com/scikit-learn/scikit-learn), [t-SNE](https://github.com/scikit-learn/scikit-learn). (Note: We modified the original TopoAE source code to make it able to run the Swiss roll dataset by adding a swiss roll dataset generation function and modifying the network structure for fair comparison.)

## Requirements

* pytorch == 1.3.1
* scipy == 1.4.1
* numpy == 1.18.5
* scikit-learn == 0.21.3
* csv == 1.0
* matplotlib == 3.1.1
* imageio == 2.6.0

## Description

* main.py  
  * SetParam() -- Parameters for training
  * Train() -- Train a new model (encoder and/or decoder)
  * Train_MultiRun() -- Run the training for multiple times, each with a different seed
  * Generation() -- Testing generation of new data of the learned manifold
  * Generalization() -- Testing dimension reduction from unseen data of the learned manifold
  * InlinePlot() -- Inline plot intermediate results during training
* dataset.py  
  * LoadData() -- Load data of selected dataset
* loss.py  
  * MLDL_Loss() -- Calculate six losses: ℒ<sub>Enc</sub>, ℒ<sub>Dec</sub>, ℒ<sub>AE</sub>, ℒ<sub>lis</sub>, ℒ<sub>push</sub>, ℒ<sub>ang</sub>  
* model.py  
  * Encoder() -- For latent feature extraction
  * Decoder() -- For generating new data on the learned manifold 
* eval.py -- Calculate performance metrics from results, each being the average of 10 seeds
* utils.py  
  * GIFPloter() -- Auxiliary tool for online plot
  * CompPerformMetrics() -- Auxiliary tool for evaluating metric 
  * Sampling() -- Sampling in the latent space for generating new data on the learned manifold 

## Running the code

1. Clone this repository

  ```
  git clone https://github.com/westlake-cairi/Markov-Lipschitz-Deep-Learning
  ```

2. Install the required dependency packages

3. To get the results for 10 seeds, run

  ```
python main.py -MultiRun
  ```

4. To get the metrics for ML-Enc and ML-AE

  ```
  python eval.py -M ML-Enc
  python eval.py -M ML-AE
  ```
The evaluation metrics are available in `./pic/PerformMetrics.csv`

5. To choose a dataset among SwissRoll, Scurve, MNIST, Spheres5500 and Spheres10000 for tow modes (ML-Enc and ML-AE)

  ```
  python main.py -D "dataset name" -M "mode"
  ```
6. To test the generalization to unseen data
  ```
  python main.py -M Test
  ```
The results are available in `./pic/file_name/Test.png`

7. To test the manifold generation
  ```
  python main.py -M Generation
  ```
The results are available in `./pic/file_name/Generation.png`

## Results

### 1. ML-Enc: Dimension reduction results -- embeddings in latent spaces
* Swiss Roll and S-Curve

   A symbol √ or X represents a success or failure in unfolding the manifold. The methods in the upper-row ML-Enc succeed and by calculation, the ML-Enc best maintains the true aspect ratio.

<p align="center">
  <img src='./figs/swiss roll.png'  width="700">
</p>

* MNIST (10 digits)

<p align="center">
  <img src='./figs/MNIST-results.png'  width="400" >
</p>

* Spheres (data designed by the 
  [
  TopoAE project
  ](https://github.com/BorgwardtLab/topological-autoencoders) )

<p align="center">
  <img src='./figs/Spheres-results.png' align="center">
</p>

### 2. ML-Enc: Performance metrics for dimension reduction on Swiss Roll (800 points) data

   This table demonstrates that the ML-Enc outperforms all the other 6 methods in all the evaluation metrics, particularly significant in terms of the isometry (LGD, RRE, Cont and Trust) and Lipschitz (*K*-Min and *K*-Max) related metrics. 

<p align="center">

|        | #Succ | L-KL   | RRE      | Trust  | Cont   | LGD     | K-Min | K-Max   | MPE    |
| ------ | :-----: | ------: | --------: | ------: | ------: | -------: | -----: | -------: | ------: |
| ML-Enc | **10**    | **0.0184** | **0.000414** | **0.9999** | **0.9985** | **0.00385** | **1.00**  | **2.14**    | **0.0718** |
| TopoAE | 0     | 0.0349 | 0.022174 | 0.9661 | 0.9884 | 0.13294 | 1.27  | 189.95  | 0.1307 |
| t-SNE  | 0     | 0.0450 | 0.006108 | 0.9987 | 0.9843 | 3.40665 | 11.1  | 1097.62 | 0.1071 |
| MLLE   | 6     | 0.1251 | 0.030702 | 0.9455 | 0.9844 | 0.04534 | 7.37  | 238.74  | 0.1709 |
| HLLE   | 6     | 0.1297 | 0.034619 | 0.9388 | 0.9859 | 0.04542 | 7.44  | 218.38  | 0.0978 |
| LTSA   | 6     | 0.1296 | 0.034933 | 0.9385 | 0.9859 | 0.04542 | 7.44  | 215.93  | 0.0964 |
| ISOMAP | 6     | 0.0234 | 0.009650 | 0.9827 | 0.9950 | 0.02376 | 1.11  | 34.35   | 0.0429 |
| LLE    | 0     | 0.1775 | 0.014249 | 0.9753 | 0.9895 | 0.04671 | 6.17  | 451.58  | 0.1400 |

</p>


### 3. ML-Enc: Ability to generalize on unseen data of the learned manifold

   The learned ML-Enc network can unfold unseen data of the learned manifold, demonstrated using the Swiss-roll with a hole, whereas the compared methods cannot.  

<p align="center">
	<img src='./figs/generalization.PNG'  width="800">
</p>


### 4. ML-AE: For dimension reduction and manifold data generation

   In the learning phase, the ML-AE taking (a) the training data as input, output (b) embedding in the learned latent space, and then reconstruct back (c). In the generation phase, the ML-Dec takes (d) random input samples in the latent space, and maps the samples to the manifold (e).

<p align="center">
<img src='./figs/generation.PNG'  width="650">
</p>


### 5. ML-AE: Evolution of training evolution

   The ML-AE training gradually unfolds the manifold from input layer to the latent layer and reconstructs the latent embedding back to data in the input space.

<p align="center">
 <img src='./figs/latent.gif'  width="400" align="middle">
</p>

## Feedback
If you have any issue about the implementation, please feel free to contact us by email:  
* Zelin Zang: zangzelin@westlake.edu.cn
* Lirong Wu: wulirong@westlake.edu.cn
