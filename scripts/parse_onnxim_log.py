import re
import sys
import pandas as pd

def parse_log(log_file_path, metrics_out="metrics.csv", layers_out="layers.csv"):
    metrics_data = []
    layers_data = []

    # Regex patterns
    # [INFO] Layer p0.layer0.attn.QKgen finish at 12345
    # or Layer p0.layer0.attn.QKgen finish at 12345
    layer_finish_re = re.compile(r"Layer\s+([\w\.\-]+)\s+(?:.*?finish at|finish at)\s+(\d+)")
    compute_time_re = re.compile(r"Total compute time\s+(\d+)")
    
    # Core [0] : Systolic Array Utilization(%) 50.00 (45.00% PE util), Vector Unit Utilization(%) 20.00, Total cycle: 1000
    util_re = re.compile(r"Core\s+\[\d+\]\s+:\s+Systolic Array Utilization\(%\)\s+([\d\.]+)\s+\(([\d\.]+)%\s+PE util\),\s+Vector Unit Utilization\(%\)\s+([\d\.]+),\s+Total cycle:\s+(\d+)")
    
    # Core [0] : Memory unit idle cycle 100 Systolic bubble cycle 50 Core idle cycle 200 
    idle_re = re.compile(r"Core\s+\[\d+\]\s+:\s+Memory unit idle cycle\s+(\d+)\s+Systolic bubble cycle\s+(\d+)\s+Core idle cycle\s+(\d+)")

    current_layer = None
    current_end = None
    
    # Assuming core_print_interval is used to calculate percentages for idle/bubble cycles
    # We might need to keep track of the interval size, but the user's graph shows percentages.
    # Let's extract raw numbers and calculate percentages later if needed.
    # Actually, the utilization prints are already percentages, except idle cycles which are raw counts.
    # Let's just collect everything.
    
    with open(log_file_path, 'r') as f:
        for line in f:
            # Match Layer Finish
            m_finish = layer_finish_re.search(line)
            if m_finish:
                current_layer = m_finish.group(1)
                current_end = int(m_finish.group(2))
                continue
            
            # Match Compute Time (follows layer finish)
            m_compute = compute_time_re.search(line)
            if m_compute and current_layer is not None:
                compute_time = int(m_compute.group(1))
                start_cycle = current_end - compute_time
                layers_data.append({
                    "Layer": current_layer,
                    "StartCycle": start_cycle,
                    "EndCycle": current_end
                })
                current_layer = None
                current_end = None
                continue
                
            # Match Utilization
            m_util = util_re.search(line)
            if m_util:
                sys_util = float(m_util.group(1))
                pe_util = float(m_util.group(2))
                vec_util = float(m_util.group(3))
                cycle = int(m_util.group(4))
                
                # Check if we have an incomplete metric entry and update it
                if len(metrics_data) > 0 and "Cycle" not in metrics_data[-1]:
                    metrics_data[-1].update({
                        "Cycle": cycle,
                        "SystolicArrayUtil(%)": sys_util,
                        "PEUtil(%)": pe_util,
                        "VectorUtil(%)": vec_util
                    })
                else:
                    metrics_data.append({
                        "Cycle": cycle,
                        "SystolicArrayUtil(%)": sys_util,
                        "PEUtil(%)": pe_util,
                        "VectorUtil(%)": vec_util
                    })
                continue
            
            # Match Idle Cycles
            m_idle = idle_re.search(line)
            if m_idle:
                mem_idle = int(m_idle.group(1))
                sys_bubble = int(m_idle.group(2))
                core_idle = int(m_idle.group(3))
                
                # We don't have the current cycle from this line, but it's usually printed 
                # right before or after the Utilization line. We'll store it and merge based on order.
                # Since idle is printed BEFORE util in Core::print_current_stats(), let's append it as a new row 
                # and let the util matcher fill in the cycle.
                metrics_data.append({
                    "MemoryIdleCycle": mem_idle,
                    "SystolicBubbleCycle": sys_bubble,
                    "CoreIdleCycle": core_idle
                })
                continue

    # Post-process metrics: Compute idle percentages based on cycle interval
    cleaned_metrics = []
    prev_cycle = 0
    for row in metrics_data:
        if "Cycle" in row:
            interval = row["Cycle"] - prev_cycle
            if interval > 0:
                row["MemoryIdle(%)"] = (row.get("MemoryIdleCycle", 0) / interval) * 100
                row["SystolicBubble(%)"] = (row.get("SystolicBubbleCycle", 0) / interval) * 100
                row["CoreIdle(%)"] = (row.get("CoreIdleCycle", 0) / interval) * 100
            prev_cycle = row["Cycle"]
            cleaned_metrics.append(row)
                
    df_metrics = pd.DataFrame(cleaned_metrics)
    df_layers = pd.DataFrame(layers_data)
    
    df_metrics.to_csv(metrics_out, index=False)
    df_layers.to_csv(layers_out, index=False)
    
    print(f"Parsed {len(df_layers)} layers and {len(df_metrics)} metric intervals.")
    print(f"Output written to {metrics_out} and {layers_out}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python parse_log.py <path_to_simulator_output.log>")
        sys.exit(1)
    
    log_path = sys.argv[1]
    parse_log(log_path)