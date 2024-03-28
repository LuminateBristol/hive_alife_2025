from simulator.wh_proc import *
from simulator.wh_sim import DataModel

batch_id = "test"
ex_ids = ['e_1', 'e_2', 'e_3', 'e_4', 'e_5', 'e_6']
data_dir = "data"
metrics = DataModel.metrics + DataModel.roc_metrics
fault_range = range(11)
no_agents = 10

for ex_id in ex_ids:
    print("Compiling %s"%ex_id)
    # Compile min samples
    for it in fault_range:
        try:
            print("--Compiling fault %d"%it)
            cm = CompileMinimalData(batch_id, ex_id, data_dir, metrics, it, no_agents)
            n, f = cm.compile(export=True)
        except Exception as e:
            print("--Error compiling fault %d: %s"%(it,e))

    # Compile into standard samples
    print("--Compiling min to std samples")
    ms = MinToStandard(data_dir, batch_id, ex_id, metrics)
    ms.convert(export=True)