import tqdm
import psutil
import subprocess

def get_bar(*args, **kwargs):
    return tqdm.tqdm(*args, ncols=80, **kwargs)

def get_usable_cpu_cnt():
    p = psutil.Process()
    ret = 0
    with p.oneshot():
        ret = len(p.cpu_affinity())
    return ret

def flatten_json(y):
    out = {}

    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '_')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + '_')
                i += 1
        else:
            out[name[:-1]] = x

    flatten(y)
    return out

def run_cmd(args):
    if isinstance(args,str):
        args = args.split()
    p = subprocess.run(args, stdout=subprocess.PIPE, check=True)
    return p.stdout.decode().strip()

