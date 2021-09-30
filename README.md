# Sim with Real: Better Together 
This project code is based on the code from [this repository](https://github.com/TianhongDai/hindsight-experience-replay).
We extend the DDPG with HER ([Hindsight Experience Replay](https://arxiv.org/abs/1707.01495)) to sim2real scenario.

## Acknowledgement:
- [Openai Baselines](https://github.com/openai/baselines)
- [HER code](https://github.com/TianhongDai/hindsight-experience-replay).

## Main Requirements
- python=3.8
- gym==0.18.0
- mpi4py==3.0.3
- mujoco-py==2.0.2.13
- torch==1.7.1



## Instruction to run the code

Train **FetchPush**:
```bash
mpirun -np 8 python -u train.py --env-name='FetchPush' 2>&1 | tee push.log
```
