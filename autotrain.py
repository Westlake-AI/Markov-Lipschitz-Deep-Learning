import os
import signal
from multiprocessing import Process, Manager

# Combination of multiple parallel training parameters (only SEED is set below, different parameters can be set as needed)
cmd=[]
for i in range(10):
    cmd.append('CUDA_VISIBLE_DEVICES={} '+'python main.py -SD {seed}'.format(seed=i))


def run(command, gpuid, gpustate):
    os.system(command.format(gpuid))
    gpustate[str(gpuid)] = True


def term(sig_num, addtion):
    print('terminate process {}'.format(os.getpid()))
    try:
        print('the processes is {}'.format(processes) )
        for p in processes:
            print('process {} terminate'.format(p.pid))
            p.terminate()

    except Exception as e:
        print(str(e))


if __name__ =='__main__':
    signal.signal(signal.SIGTERM, term)

    gpustate=Manager().dict({str(i):True for i in range(1,8)})
    processes=[]
    idx=0

    # Open multiple threads to perform multiple GPU parallel training
    while idx<len(cmd):
        for gpuid in range(1,7):
            if gpustate[str(gpuid)]==True:
                print(idx)
                gpustate[str(gpuid)]=False
                p=Process(target=run,args=(cmd[idx],gpuid,gpustate),name=str(gpuid))
                p.start()

                print(gpustate)
                processes.append(p)
                idx+=1

                break

    for p in processes:
        p.join()
