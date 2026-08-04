import os
import sys
import math
import pandas as pd
import argparse
import subprocess
import concurrent.futures

# ── Absolute paths (all relative to this file's location) ────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))
CENT_ROOT    = os.path.abspath(os.path.join(_HERE, "../../.."))
CENT_SIM_DIR = os.path.join(CENT_ROOT, "cent_simulation")
AIM_SIM      = os.path.join(CENT_ROOT, "aim_simulator", "build", "ramulator2")
AIM_CFG      = os.path.join(CENT_ROOT, "aim_simulator", "test", "example.yaml")
ORIG_TRACE   = os.path.join(CENT_ROOT, "trace")          # compile.sh / compile.py live here
TRACE_BASE   = os.path.abspath(os.path.join(_HERE, "../traces"))  # DDA-specific traces

# Add cent_simulation to path so we can import its modules
sys.path.insert(0, CENT_SIM_DIR)

from cxl_latency import llama_latency, gpt_latency, vector_latency
from cent_power_calculator import (DRAM_POWER, ACCEL_CYCLE, ACCEL_POWER, SRAM_POWER,
    CTRL_POWER, commands, isrs, power_calculator, command_processor,
    KILO, MEGA, GIGA, FREQ, WORD_SIZE, tRC, tBL, tCCDL, RV_COUNT,
    SB_RD_CYCLE, SB_WR_CYCLE, EXP_LANE_CYCLE, RV_RMSNorm_CYCLE,
    RV_ROTEmbed_CYCLE, RV_SFT_CYCLE_PIPELINE, RV_SFT_CYCLE_SINGLE)
from utils import (InOut_latency, n_heads, gqa_factor, embedding_size, ffn_size,
    TransformerBlock_number, minimal_channel_per_block,
    pipeline_parallel_mode_list, model_parallel_mode_list)

# ── Trace path helper ─────────────────────────────────────────────────────────
def trace_dir(num_channels, num_devices, mode, model):
    """DDA trace dir — includes num_devices level to avoid collisions."""
    return os.path.join(TRACE_BASE,
        f"{num_channels}_channels_per_device",
        f"{num_devices}_devices",
        mode, model)

def trace_path(num_channels, num_devices, mode, model, filename):
    return os.path.join(trace_dir(num_channels, num_devices, mode, model), filename)


def get_args():
    parser = argparse.ArgumentParser('run_sim_dda.py')
    parser.add_argument("--num_channels", type=int, default=32)
    parser.add_argument("--num_devices", type=int, default=32)
    parser.add_argument("--PCIE_lanes", type=int, default=144)
    parser.add_argument("--reuse_size", type=int, default=32)
    parser.add_argument("--generate_trace_max_workers", type=int, default=20)
    parser.add_argument("--run_simulation_max_workers", type=int, default=4)
    parser.add_argument("--model", choices=["Llama2-7B", "Llama2-13B", "Llama2-70B"], required=True)
    parser.add_argument("--generate_trace", action="store_true")
    parser.add_argument("--simulate_trace", action="store_true")
    parser.add_argument("--process_results", action="store_true")
    parser.add_argument("--update_csv", action="store_true")
    parser.add_argument("--simulation_result_path", type=str,
                        default=os.path.join(_HERE, "../results/simulation_results_dda.csv"))
    parser.add_argument("--process_throughputs", action="store_true")
    parser.add_argument("--processed_result_path", type=str,
                        default=os.path.join(_HERE, "../results/processed_results_dda.csv"))
    parser.add_argument("--phase", choices=["end2end", "prefill", "decoding"], default="end2end")
    parser.add_argument("--prefill", type=int, default=512)
    parser.add_argument("--decoding", type=int, default=3584)
    parser.add_argument("--seqlen", type=int, nargs='+')
    parser.add_argument("--seqlen_gap", type=int, default=128)
    parser.add_argument("--model_parallel", action="store_true")
    parser.add_argument("--inter-device-attention", action="store_true")
    # DDA-specific: total devices in the full system, for correct proportional PCIe allocation.
    # Baseline: leave unset (defaults to num_devices → identical to original run_sim.py).
    # Pool runs: set to N_total (8/20/32). PCIe_lanes_per_device = PCIE_lanes // total_devices.
    parser.add_argument("--total_devices", type=int, default=None,
                        help="Total devices in full system (for proportional PCIe lanes). "
                             "Defaults to --num_devices (baseline-compatible).")
    # Restrict the TP sweep. A DDA Pool L only ever uses TP=N_L (PP=1), so sweeping
    # every factor of N_L generates ~5x the traces for rows we throw away.
    parser.add_argument("--fc_devices", type=int, nargs='+', default=None,
                        help="TP degrees to sweep in model_parallel mode. "
                             "Defaults to every factor of --num_devices.")
    args = parser.parse_args()
    if args.total_devices is None:
        args.total_devices = args.num_devices
    return args


def factorize(n):
    factors = []
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            factors.append(i)
            if i != n // i:
                factors.append(n // i)
    return sorted(factors)


def fc_list(args):
    """TP degrees to sweep in model_parallel mode.

    Defaults to every factor of num_devices (original behaviour). --fc_devices
    narrows it: a DDA Pool L only reads the PP=1/TP=N_L row, so generating the
    other factors' traces is wasted work.
    """
    if getattr(args, "fc_devices", None):
        return sorted(args.fc_devices)
    return factorize(args.num_devices)


def _run_function_sim(cmd):
    """Run function_sim.py subprocess from cent_simulation directory."""
    subprocess.run(cmd, cwd=CENT_SIM_DIR)


def generate_trace(args, seqlen_list):
    print(f"Generating traces for {args.model} ({args.num_devices} devices) ...")

    if args.model == "Llama2-70B" or "Llama3" in args.model:
        model_flag = "--Llama-GQA"
    elif "Llama2" in args.model:
        model_flag = "--Llama"
    else:
        model_flag = "--GPT3-175B"

    cmds = []
    blocks_per_device = (TransformerBlock_number[args.model] - 1) // args.num_devices + 1
    channels_per_block = args.num_channels // blocks_per_device
    FC_devices_list = fc_list(args)

    # Embedding trace
    seqlen = args.prefill + args.decoding
    if args.model_parallel:
        for FC_devices in FC_devices_list:
            tf = trace_path(args.num_channels, args.num_devices,
                            "model_parallel_embedding", args.model,
                            f"trace_{FC_devices}_FC_devices_seqlen_{seqlen}.txt")
            if not os.path.exists(tf):
                cmds.append(["python3", os.path.join(CENT_SIM_DIR, "function_sim.py"),
                    model_flag, "--n_heads", str(n_heads[args.model]),
                    "--ffn_dim", str(ffn_size[args.model]),
                    "--embedding", "--only-trace",
                    "--num-channels", str(args.num_channels),
                    "--FC-devices", str(FC_devices),
                    "--model-parallel", "--seqlen", str(seqlen),
                    "--op-trace", "--GEMV", "reuse-GB",
                    "--reuse-size", str(args.reuse_size),
                    "--trace-file", tf])
    else:
        tf = trace_path(args.num_channels, args.num_devices,
                        "pipeline_parallel_embedding", args.model,
                        f"trace_{channels_per_block}_channels_per_block_seqlen_{seqlen}.txt")
        if not os.path.exists(tf):
            cmds.append(["python3", os.path.join(CENT_SIM_DIR, "function_sim.py"),
                model_flag, "--n_heads", str(n_heads[args.model]),
                "--ffn_dim", str(ffn_size[args.model]),
                "--embedding", "--only-trace",
                "--num-channels", str(args.num_channels),
                "--channels-per-block", str(channels_per_block),
                "--pipeline-parallel", "--multi-tb-per-device",
                "--seqlen", str(seqlen),
                "--op-trace", "--GEMV", "reuse-GB",
                "--reuse-size", str(args.reuse_size),
                "--trace-file", tf])

    # Per-seqlen traces
    for seqlen in seqlen_list:
        if args.model_parallel:
            for FC_devices in FC_devices_list:
                tf = trace_path(args.num_channels, args.num_devices,
                                "model_parallel", args.model,
                                f"trace_{FC_devices}_FC_devices_seqlen_{seqlen}.txt")
                if not os.path.exists(tf):
                    cmd = ["python3", os.path.join(CENT_SIM_DIR, "function_sim.py"),
                        model_flag, "--n_heads", str(n_heads[args.model]),
                        "--ffn_dim", str(ffn_size[args.model]),
                        "--only-trace",
                        "--num-channels", str(args.num_channels),
                        "--FC-devices", str(FC_devices),
                        "--model-parallel", "--seqlen", str(seqlen),
                        "--op-trace", "--GEMV", "reuse-GB",
                        "--reuse-size", str(args.reuse_size),
                        "--trace-file", tf]
                    if args.inter_device_attention:
                        cmd.append("--inter-device-attention")
                    cmds.append(cmd)

                tf_fc = trace_path(args.num_channels, args.num_devices,
                                   "model_parallel_FC", args.model,
                                   f"trace_{FC_devices}_FC_devices_seqlen_{seqlen}.txt")
                if not os.path.exists(tf_fc):
                    cmds.append(["python3", os.path.join(CENT_SIM_DIR, "function_sim.py"),
                        model_flag, "--n_heads", str(n_heads[args.model]),
                        "--ffn_dim", str(ffn_size[args.model]),
                        "--only-FC", "--only-trace",
                        "--num-channels", str(args.num_channels),
                        "--FC-devices", str(FC_devices),
                        "--model-parallel", "--seqlen", str(seqlen),
                        "--op-trace", "--GEMV", "reuse-GB",
                        "--reuse-size", str(args.reuse_size),
                        "--trace-file", tf_fc])
        else:
            if channels_per_block < minimal_channel_per_block[args.model]:
                raise ValueError(f"channels_per_block {channels_per_block} < min {minimal_channel_per_block[args.model]}")
            tf = trace_path(args.num_channels, args.num_devices,
                            "pipeline_parallel", args.model,
                            f"trace_{channels_per_block}_channels_per_block_seqlen_{seqlen}.txt")
            if not os.path.exists(tf):
                cmds.append(["python3", os.path.join(CENT_SIM_DIR, "function_sim.py"),
                    model_flag, "--n_heads", str(n_heads[args.model]),
                    "--ffn_dim", str(ffn_size[args.model]),
                    "--only-trace",
                    "--num-channels", str(args.num_channels),
                    "--channels-per-block", str(channels_per_block),
                    "--pipeline-parallel", "--multi-tb-per-device",
                    "--seqlen", str(seqlen),
                    "--op-trace", "--GEMV", "reuse-GB",
                    "--reuse-size", str(args.reuse_size),
                    "--trace-file", tf])

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.generate_trace_max_workers) as ex:
        futures = [ex.submit(_run_function_sim, cmd) for cmd in cmds]
        for f in concurrent.futures.as_completed(futures):
            f.result()


def run_command(command, log_file):
    print(command)
    result = subprocess.run(command, shell=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    filtered = "\n".join(l for l in result.stdout.splitlines() if not l.startswith('['))
    with open(log_file, "w") as fh:
        fh.write(filtered)


def detect_emtpy_file(path):
    return os.stat(path).st_size == 0


def simulate_trace(args, seqlen_list):
    cmds = []
    blocks_per_device = (TransformerBlock_number[args.model] - 1) // args.num_devices + 1
    channels_per_block = args.num_channels // blocks_per_device
    FC_devices_list = fc_list(args)

    # Embedding
    seqlen = args.prefill + args.decoding
    if args.model_parallel:
        for FC_devices in FC_devices_list:
            tf  = trace_path(args.num_channels, args.num_devices,
                             "model_parallel_embedding", args.model,
                             f"trace_{FC_devices}_FC_devices_seqlen_{seqlen}.txt")
            log = tf + ".log"
            if not os.path.exists(log) or detect_emtpy_file(log):
                cmds.append((f"{AIM_SIM} -f {AIM_CFG} -t {tf}", log))
    else:
        tf  = trace_path(args.num_channels, args.num_devices,
                         "pipeline_parallel_embedding", args.model,
                         f"trace_{channels_per_block}_channels_per_block_seqlen_{seqlen}.txt")
        log = tf + ".log"
        if not os.path.exists(log) or detect_emtpy_file(log):
            cmds.append((f"{AIM_SIM} -f {AIM_CFG} -t {tf}", log))

    for seqlen in seqlen_list:
        if args.model_parallel:
            for FC_devices in FC_devices_list:
                for mode in ["model_parallel", "model_parallel_FC"]:
                    tf  = trace_path(args.num_channels, args.num_devices, mode, args.model,
                                     f"trace_{FC_devices}_FC_devices_seqlen_{seqlen}.txt")
                    log = tf + ".log"
                    if not os.path.exists(log) or detect_emtpy_file(log):
                        cmds.append((f"{AIM_SIM} -f {AIM_CFG} -t {tf}", log))
        else:
            tf  = trace_path(args.num_channels, args.num_devices,
                             "pipeline_parallel", args.model,
                             f"trace_{channels_per_block}_channels_per_block_seqlen_{seqlen}.txt")
            log = tf + ".log"
            if not os.path.exists(log) or detect_emtpy_file(log):
                cmds.append((f"{AIM_SIM} -f {AIM_CFG} -t {tf}", log))

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.run_simulation_max_workers) as ex:
        futures = [ex.submit(run_command, cmd, log) for cmd, log in cmds]
        for f in concurrent.futures.as_completed(futures):
            f.result()


def process_results(args):
    print("Processing results...")
    mode_list = model_parallel_mode_list if args.model_parallel else pipeline_parallel_mode_list
    for mode in mode_list:
        compile_dir = trace_dir(args.num_channels, args.num_devices, mode, args.model)
        subprocess.run(["cp", os.path.join(ORIG_TRACE, "compile.sh"), compile_dir])
        subprocess.run(["cp", os.path.join(ORIG_TRACE, "compile.py"), compile_dir])
        result = subprocess.run(["bash", "compile.sh"], cwd=compile_dir,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        with open(os.path.join(compile_dir, "result.txt"), "w") as fh:
            fh.write(result.stdout)
            fh.write(result.stderr)
        result = subprocess.run(["python3", "compile.py", "./result.txt"], cwd=compile_dir,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        with open(os.path.join(compile_dir, "compiled_results.txt"), "w") as fh:
            fh.write(result.stdout)
            fh.write(result.stderr)


def calculate_acc_latency(args, seqlen):
    latency = {}
    GQA_factor = 1.00 + 1.00 / gqa_factor[args.model]
    latency["RMSNorm_latency"]  = embedding_size[args.model] / 16.00 / 16.00 / args.num_channels * ACCEL_CYCLE["VEC"]
    latency["RMSNorm_latency"] += SB_RD_CYCLE + SB_WR_CYCLE + 1.00
    latency["RMSNorm_latency"] += RV_RMSNorm_CYCLE
    latency["RMSNorm_latency"]  = float(2.00 * latency["RMSNorm_latency"]) / float(FREQ / KILO)
    latency["Softmax_latency"]  = seqlen * n_heads[args.model] / 16.00 / args.num_channels * ACCEL_CYCLE["EXP"]
    latency["Softmax_latency"] += seqlen * n_heads[args.model] / 16.00 / args.num_channels * ACCEL_CYCLE["VEC"]
    latency["Softmax_latency"] += n_heads[args.model] * 1.00 * SB_RD_CYCLE
    latency["Softmax_latency"] += n_heads[args.model] * RV_SFT_CYCLE_PIPELINE
    latency["Softmax_latency"]  = float(latency["Softmax_latency"]) / float(FREQ / KILO)
    latency["RotEmbed_latency"] = embedding_size[args.model] * RV_ROTEmbed_CYCLE
    latency["RotEmbed_latency"] = float(GQA_factor * latency["RotEmbed_latency"]) / float(FREQ / KILO)
    return latency


def load_data_point(args, seqlen, FC_devices, channels_per_block, PCIe_lanes_per_device,
                    blocks_per_device, embedding_latency, utilized_devices, pp, tp):
    if args.model_parallel:
        path = trace_path(args.num_channels, args.num_devices,
                          "model_parallel", args.model,
                          f"trace_{FC_devices}_FC_devices_seqlen_{seqlen}.txt.log")
    else:
        path = trace_path(args.num_channels, args.num_devices,
                          "pipeline_parallel", args.model,
                          f"trace_{channels_per_block}_channels_per_block_seqlen_{seqlen}.txt.log")
    stats = command_processor(path)
    pim_latency = stats["latency"]

    if args.model_parallel:
        if "Llama" in args.model:
            cxl_latency = llama_latency([embedding_size[args.model], ffn_size[args.model]],
                                        PCIe_lanes_per_device, FC_devices, args.num_devices)
        else:
            cxl_latency = gpt_latency([embedding_size[args.model], ffn_size[args.model]],
                                       PCIe_lanes_per_device, FC_devices, args.num_devices)
        embedding_latency_data = embedding_latency['model_parallel'][FC_devices]
    else:
        cxl_latency = vector_latency(embedding_size[args.model], PCIe_lanes_per_device)
        embedding_latency_data = embedding_latency['pipeline_parallel'][channels_per_block]

    acc_latency_dict = calculate_acc_latency(args, seqlen)
    acc_latency = (acc_latency_dict["RMSNorm_latency"] +
                   acc_latency_dict["Softmax_latency"] +
                   acc_latency_dict["RotEmbed_latency"]) * blocks_per_device
    transformer_block_latency = pim_latency + cxl_latency + acc_latency
    token_latency = transformer_block_latency * TransformerBlock_number[args.model] + embedding_latency_data + InOut_latency
    throughput = 1000 / token_latency * pp

    energy_token = {}
    power_alldv  = {}
    PCIE = (embedding_size[args.model] * 10 + ffn_size[args.model] * 2
            if args.model_parallel else embedding_size[args.model])
    energy_main, latency_main = power_calculator(stats, PCIE, n_heads[args.model],
                                                 embedding_size[args.model], seqlen,
                                                 gqa_factor[args.model])
    if args.model_parallel:
        pipeline_stages = args.num_devices // FC_devices
        fc_path = trace_path(args.num_channels, args.num_devices,
                             "model_parallel_FC", args.model,
                             f"trace_{FC_devices}_FC_devices_seqlen_{seqlen}.txt.log")
        stats_FC = command_processor(fc_path)
        energy_FC, latency_FC = power_calculator(stats_FC, PCIE, n_heads[args.model],
                                                 embedding_size[args.model], seqlen,
                                                 gqa_factor[args.model])
        for comp in energy_main.keys():
            energy_token[comp] = (energy_main[comp] + energy_FC[comp] * (FC_devices - 1)) * TransformerBlock_number[args.model]
            power_alldv[comp]  = (energy_main[comp] + energy_FC[comp] * (FC_devices - 1)) * pipeline_stages / stats["latency"]
    else:
        for comp in energy_main.keys():
            energy_token[comp] = energy_main[comp] * utilized_devices
            power_alldv[comp]  = energy_main[comp] * utilized_devices / stats["latency"]

    total_energy = sum(energy_token.values())
    total_power  = sum(power_alldv.values())
    device_utilization = 1.0 * utilized_devices / args.num_devices

    return pd.DataFrame([{
        'Model': args.model,
        'Device number': args.num_devices,
        'Pipeline parallelism': pp,
        'Tensor parallelism': tp,
        'Channels per device': args.num_channels,
        'Channels per block': channels_per_block,
        'Sequence length': seqlen,
        'PIM latency': pim_latency,
        'CXL latency': cxl_latency,
        'Acc latency': acc_latency,
        'TransformerBlock latency': transformer_block_latency,
        'Embedding latency': embedding_latency_data,
        'Token latency (ms)': token_latency,
        'Throughput (tokens/s)': throughput,
        'Token energy (mJ)': total_energy,
        'Total power (W)': total_power,
        'Device utilization': device_utilization,
    }])


def update_csv(args, seqlen_list):
    print("Updating simulation results to CSV...")

    if os.path.exists(args.simulation_result_path):
        results_df = pd.read_csv(args.simulation_result_path)
    else:
        columns = ['Model', 'Device number', 'Pipeline parallelism', 'Tensor parallelism',
                   'Channels per device', 'Channels per block', 'Sequence length',
                   'PIM latency', 'CXL latency', 'Acc latency', 'TransformerBlock latency',
                   'Embedding latency', 'Token latency (ms)', 'Throughput (tokens/s)',
                   'Token energy (mJ)', 'Total power (W)', 'Device utilization']
        results_df = pd.DataFrame(columns=columns)

    embedding_latency = {'pipeline_parallel': {}, 'model_parallel': {}}
    if args.model_parallel:
        FC_devices_list = fc_list(args)
        emb_dir = trace_dir(args.num_channels, args.num_devices,
                            "model_parallel_embedding", args.model)
        with open(os.path.join(emb_dir, "compiled_results.txt")) as fh:
            for line in fh:
                fname, lat = line.split()[0], line.split()[1]
                fc = int(fname.split('_')[1])
                embedding_latency["model_parallel"][fc] = float(lat)
    else:
        emb_dir = trace_dir(args.num_channels, args.num_devices,
                            "pipeline_parallel_embedding", args.model)
        with open(os.path.join(emb_dir, "compiled_results.txt")) as fh:
            for line in fh:
                fname, lat = line.split()[0], line.split()[1]
                cpb = int(fname.split('_')[1])
                embedding_latency["pipeline_parallel"][cpb] = float(lat)

    for seqlen in seqlen_list:
        PCIe_lanes_per_device = args.PCIE_lanes // args.total_devices

        if args.model_parallel:
            FC_devices_list = fc_list(args)
            blocks_per_device = 1
            channels_per_block = args.num_channels
            utilized_devices = args.num_devices
            for FC_devices in FC_devices_list:
                pp = args.num_devices // FC_devices
                tp = FC_devices
                new_df = load_data_point(args, seqlen, FC_devices, channels_per_block,
                                         PCIe_lanes_per_device, blocks_per_device,
                                         embedding_latency, utilized_devices, pp, tp)
                results_df = pd.concat([results_df, new_df], ignore_index=True)
        else:
            pp = TransformerBlock_number[args.model]
            tp = 1
            pp_per_device = (pp - 1) // args.num_devices + 1
            blocks_per_device = pp_per_device * (TransformerBlock_number[args.model] // pp)
            channels_per_block = args.num_channels // blocks_per_device
            if channels_per_block < minimal_channel_per_block[args.model]:
                continue
            utilized_devices = (TransformerBlock_number[args.model] - 1) // blocks_per_device + 1
            new_df = load_data_point(args, seqlen, 0, channels_per_block,
                                     PCIe_lanes_per_device, blocks_per_device,
                                     embedding_latency, utilized_devices, pp, tp)
            results_df = pd.concat([results_df, new_df], ignore_index=True)

    results_df = results_df.drop_duplicates(subset=[
        'Model', 'Device number', 'Pipeline parallelism', 'Tensor parallelism',
        'Channels per device', 'Channels per block', 'Sequence length'])
    results_df = results_df.sort_values(by=[
        'Model', 'Device number', 'Pipeline parallelism', 'Tensor parallelism',
        'Channels per device', 'Channels per block', 'Sequence length'])
    results_df.to_csv(args.simulation_result_path, index=False)


def process_throughputs(args):
    print("Processing throughputs...")

    if not os.path.exists(args.simulation_result_path):
        raise ValueError(f"{args.simulation_result_path} not found. Run update_csv first.")
    df_sim = pd.read_csv(args.simulation_result_path)

    if os.path.exists(args.processed_result_path):
        results_df = pd.read_csv(args.processed_result_path)
    else:
        columns = ['Model', 'Device number', 'Seqlen', 'Pipeline parallelism',
                   'Tensor parallelism', 'Phase', 'Total Latency (s)',
                   'Throughput (tokens/s)', 'Energy per Token (mJ)', 'Total power (W)']
        results_df = pd.DataFrame(columns=columns)

    if args.model_parallel:
        FC_devices_list = fc_list(args)
        for FC_devices in FC_devices_list:
            pp = args.num_devices // FC_devices
            tp = FC_devices
            df = df_sim[(df_sim['Model'] == args.model) &
                        (df_sim['Pipeline parallelism'] == pp) &
                        (df_sim['Tensor parallelism'] == tp)]
            if args.phase == "prefill":
                df = df[df['Sequence length'] <= args.prefill]
                seqlen = args.prefill
            elif args.phase == "decoding":
                df = df[(df['Sequence length'] > args.prefill) &
                        (df['Sequence length'] <= args.prefill + args.decoding)]
                seqlen = args.decoding
            else:
                df = df[df['Sequence length'] <= args.prefill + args.decoding]
                seqlen = args.prefill + args.decoding

            new_result = {
                'Model': args.model,
                'Device number': args.num_devices,
                'Seqlen': args.prefill + args.decoding,
                'Pipeline parallelism': pp,
                'Tensor parallelism': tp,
                'Phase': args.phase,
                'Total Latency (s)': df['Token latency (ms)'].mean() * seqlen / 1000,
                'Throughput (tokens/s)': df['Throughput (tokens/s)'].mean(),
                'Energy per Token (mJ)': df['Token energy (mJ)'].mean(),
                'Total power (W)': df['Total power (W)'].mean(),
            }
            results_df = pd.concat([results_df, pd.DataFrame([new_result])], ignore_index=True)
    else:
        df = df_sim[(df_sim['Model'] == args.model) &
                    (df_sim['Pipeline parallelism'] == TransformerBlock_number[args.model]) &
                    (df_sim['Tensor parallelism'] == 1)]
        if args.phase == "prefill":
            df = df[df['Sequence length'] <= args.prefill]
            seqlen = args.prefill
        elif args.phase == "decoding":
            df = df[(df['Sequence length'] > args.prefill) &
                    (df['Sequence length'] <= args.prefill + args.decoding)]
            seqlen = args.decoding
        else:
            df = df[df['Sequence length'] <= args.prefill + args.decoding]
            seqlen = args.prefill + args.decoding

        new_result = {
            'Model': args.model,
            'Device number': args.num_devices,
            'Seqlen': args.prefill + args.decoding,
            'Pipeline parallelism': TransformerBlock_number[args.model],
            'Tensor parallelism': 1,
            'Phase': args.phase,
            'Total Latency (s)': df['Token latency (ms)'].mean() * seqlen / 1000,
            'Throughput (tokens/s)': df['Throughput (tokens/s)'].mean(),
            'Energy per Token (mJ)': df['Token energy (mJ)'].mean(),
            'Total power (W)': df['Total power (W)'].mean(),
        }
        results_df = pd.concat([results_df, pd.DataFrame([new_result])], ignore_index=True)

    results_df = results_df.drop_duplicates()
    results_df = results_df.sort_values(by=[
        'Model', 'Device number', 'Seqlen',
        'Pipeline parallelism', 'Tensor parallelism', 'Phase'])
    results_df.to_csv(args.processed_result_path, index=False)


if __name__ == "__main__":
    args = get_args()

    if args.seqlen:
        seqlen_list = args.seqlen
    else:
        seqlen_list = [i * args.seqlen_gap
                       for i in range(1, (args.prefill + args.decoding) // args.seqlen_gap + 1)]

    # Create all trace subdirectories
    for mode in pipeline_parallel_mode_list + model_parallel_mode_list:
        os.makedirs(trace_dir(args.num_channels, args.num_devices, mode, args.model), exist_ok=True)
    os.makedirs(os.path.dirname(args.simulation_result_path), exist_ok=True)

    if args.generate_trace:
        generate_trace(args, seqlen_list)
    if args.simulate_trace:
        simulate_trace(args, seqlen_list)
    if args.process_results:
        process_results(args)
    if args.update_csv:
        update_csv(args, seqlen_list)
    if args.process_throughputs:
        process_throughputs(args)
