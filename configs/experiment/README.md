# Evaluate a single task, no pruning (original GLUE)


To evaluate on `cola`, run:
```bash
python run.py experiment=cola
```
and so on. You can choose among following tasks:
cola, sst2, mrpc, qqp, qnli, rte, wnli.

---

To evaluate on the whole GLUE suite:
```bash
python run.py --multirun \
    experiment=cola,sst2,mrpc,qqp,qnli,rte,wnli 
```
it simply sweeps among multiple experiment files.


### TODO
Implement `mnli` (2x val), `stsb` (one class)