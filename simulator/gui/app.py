from pathlib import Path
import sys
import os

dir_root = Path(__file__).resolve().parents[1]
sys.path.append(str(dir_root))
dir_root_2 = Path(__file__).resolve().parents[2]
sys.path.append(str(dir_root_2))
from lib import RedisConn, RedisKeys, Config
from wh_sim import ExportRedisData, Simulator, ExportThresholdModel
from simulator import CFG_FILES, MODEL_ROOT, STATS_ROOT

from flask import Flask, request, render_template, redirect, jsonify
import json
import yaml

###### Setup #####

app = Flask(__name__, static_url_path='/s/', static_folder='static')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

cfg_file = CFG_FILES['default']
cfg = Config(cfg_path=cfg_file)

ex_cfg = CFG_FILES['ex_1']
cfg2 = Config(cfg_path=ex_cfg)
ex_avail = list(cfg2.__dict__.keys())

GLOBAL = {
    'metadata': {
        'arena_w': cfg.get('warehouse', 'width'),
        'arena_h': cfg.get('warehouse', 'height'),
        'robot_radius': cfg.get('robot', 'radius'),
        'box_radius': cfg.get('warehouse', 'box_radius'),
        'deposit_zones': [],
        'sim_runtime': cfg.get('time_limit')+1,
        'number_of_agents': cfg.get('warehouse', 'number_of_agents')
    },
    'scenarios': [
        'box_pickup',
    ],
    'config': ex_avail,
    'simdata': {}
}

r = RedisConn()

####################

def render_page(template, data):
    return render_template(template, data=data, global_data=GLOBAL)

@app.route("/")#, defaults={'scenario': None})
@app.route("/<scenario>/")
def sim(scenario='box_pickup'):
    # fetch_metadata()
    return render_page('main.html', {'scenario': scenario})     

# def fetch_metadata():
#     rk = RedisKeys()
#     mkeys = rk.gen_metadata_keys()
#     if not r.is_connected():
#         r.reconnect()
    
#     for meta, key in mkeys.items():
#         val = r.get(key)
#         if val is not None:
#             GLOBAL['metadata'][meta] = json.loads(val.strip().decode())

def fetch_simdata(timestep, scenario, json_decode=False):
    rk = RedisKeys()
    sckeys = ExportRedisData.vis_keys
    tkeys = rk.gen_timestep_keys(timestep, sckeys)
    if not r.is_connected():
        r.reconnect()

    simdata = {}#GLOBAL['simdata']
    for it, key in tkeys.items():
        try:
            val = r.get(key).strip().decode()
            if json_decode:
                val = json.loads(val)
            
            if timestep not in simdata:
                simdata[timestep] = {} # second timestep - unneeded... remove flatten

            simdata[timestep][it] = val                
        except Exception as e:
            print("Could not retrieve key data: t %d, k %s, e %s"%(timestep,key,e))

    return simdata

def fetch_metadata(json_decode=False):
    rk = RedisKeys()
    mdkeys = ExportRedisData.setup_keys
    mkeys = rk.gen_metadata_keys(mdkeys)
    if not r.is_connected():
        r.reconnect()

    metadata = {}
    for it, key in mkeys.items():
        try:
            val = r.get(key).strip().decode()
            if json_decode:
                val = json.loads(val)

            metadata[it] = val                
        except Exception as e:
            print("Could not retrieve key data: k %s, e %s"%(key,e))

    return metadata

def fetch_metricdata(timestep, json_decode=False):
    metrics = ExportRedisData.metrics + ExportRedisData.roc_metrics
    if not r.is_connected():
        r.reconnect()

    rk = RedisKeys()
    metricdata = {}
    for i, metric in enumerate(metrics):
        try:
            key = rk.gen_metric_timestep_key(timestep, i)
            val = r.get(key).strip().decode()
            if json_decode:
                val = json.loads(val)
            metricdata[metric] = val
        except Exception as e:
            print("Could not retrieve key data: t %d, k %s, e %s"%(timestep,key,e))

    return metricdata

def fetch_ad_prediction(timestep, json_decode=False):
    rk = RedisKeys()
    if not r.is_connected():
        r.reconnect()

    try:
        key = rk.gen_timestep_key(timestep, ExportThresholdModel.KEY)
        val = r.get(key).strip().decode()
        if json_decode:
            val = json.loads(val)
        return val
    except Exception as e:
        print("Could not retrieve key data: t %d, k %s, e %s"%(timestep,key,e))

@app.route('/fetch-redis/<runtime>/<scenario>')
def fetch_vis_data_from_run(runtime, scenario):
    simdata = {}
    metricdata = {}
    ad_pred = {}
    for i in range(1, int(runtime)):
        simdata[i] = fetch_simdata(i, scenario)
        metricdata[i] = fetch_metricdata(i)
        ad_pred[i] = fetch_ad_prediction(i)
        
    metadata = fetch_metadata()
    data = {'metadata': metadata, 'simdata': simdata, 'metricdata': metricdata, 'ad_pred': ad_pred}
    return jsonify(data)

@app.route('/run-ex/<ex_id>/<no_faults>', methods=['POST'])
def run_experiment(ex_id, no_faults):
    default_cfg_file = CFG_FILES['default']
    cfg_file = CFG_FILES['ex_1']
    cfg_obj = Config(cfg_file, default_cfg_file, ex_id=ex_id)
    data_model = ExportRedisData(export_vis_data=True, compute_roc=True)
    thresh_file = os.path.join(MODEL_ROOT, "%s_%s.txt"%(ex_id, "emin_sc"))
    stats_file = os.path.join(STATS_ROOT, "%s_%s.txt"%(ex_id, "emin_sc"))
    ad_model = ExportThresholdModel(10, thresh_file, stats_file, 3, 0.15, 2)

    faults = [int(no_faults)]
    # Create simulator object
    sim = Simulator(cfg_obj,
        data_model=data_model,
        fault_count=faults,
        ad_model=ad_model,
        random_seed=66764970)

    sim.run()
    return jsonify(sim.random_seed)