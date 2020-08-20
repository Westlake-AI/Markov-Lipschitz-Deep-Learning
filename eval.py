import csv
import argparse
import numpy as np
from utils import CompPerformMetrics

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-M", "--mode", default='ML-Enc', type=str)   # Calculating the metrics for the ML-AE or ML-Enc
    args = parser.parse_args()

    # Calculate the average result of 10 seeds
    out_seeds = []
    for i in range(10):
        path = 'pic/MLDL_SwissRoll_N800_SD{}/out/'.format(i)
        data0 = np.loadtxt(path + '0.txt')
        data8 = np.loadtxt(path + '8.txt')
        data11 = np.loadtxt(path + '11.txt')

        if args.mode == 'ML-Enc':
            data9 = np.loadtxt(path + '9.txt')
            indicator = CompPerformMetrics(data=data0, latent=data9, lat=[data8])
        if args.mode == 'ML-AE':
            data18 = np.loadtxt(path + '18.txt')
            indicator = CompPerformMetrics(data=data0, latent=data18, lat=[data8, data11])    

        out_seeds.append(np.array(list(indicator.values())))
        print(indicator)

    out_seeds = np.array(out_seeds)
    out_seeds = out_seeds.mean(axis=0)

    # Save metrics results to a csv file
    outFile = open('./pic/PerformMetrics.csv','a+', newline='')
    writer = csv.writer(outFile, dialect='excel')

    names = []
    for v, k in indicator.items():
        names.append(v)

    writer.writerow(names)
    writer.writerow(out_seeds)