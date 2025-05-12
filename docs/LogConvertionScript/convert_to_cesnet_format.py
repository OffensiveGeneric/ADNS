import pandas as pd
import numpy as np
from datetime import datetime
import os

# Load tshark log
df = pd.read_csv("traffic_log.csv", header=None, names=[
    "timestamp", "src_ip", "dst_ip", "length", "protocol",
    "tcp_src", "tcp_dst", "udp_src", "udp_dst", "layers"
])

# Convert epoch to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

# Assign hour bucket
df['minute'] = df['timestamp'].dt.floor('1T')  # 1 minute bins



# Normalize columns
df['length'] = pd.to_numeric(df['length'], errors='coerce')
df['tcp_src'] = pd.to_numeric(df['tcp_src'], errors='coerce')
df['tcp_dst'] = pd.to_numeric(df['tcp_dst'], errors='coerce')
df['udp_src'] = pd.to_numeric(df['udp_src'], errors='coerce')
df['udp_dst'] = pd.to_numeric(df['udp_dst'], errors='coerce')

# Unified port field
df['dst_port'] = df['tcp_dst'].fillna(df['udp_dst'])

# Direction (based on simple heuristic)
df['direction'] = np.where(df['src_ip'].str.startswith("192.168"), "out", "in")

# Aggregation
agg = df.groupby('minute').agg(

    n_flows=('src_ip', 'nunique'),
    n_packets=('src_ip', 'count'),
    n_bytes=('length', 'sum'),
    avg_duration=('timestamp', lambda x: (x.max() - x.min()).total_seconds()),
    n_dest_ports=('dst_port', lambda x: x.nunique()),
    n_dest_ip=('dst_ip', lambda x: x.nunique()),
    tcp_udp_ratio_packets=('protocol', lambda x: (x == 6).sum() / ((x == 17).sum() + 1)),
    dir_ratio_bytes=('direction', lambda x: (df.loc[x.index, 'length'][x == "out"].sum() + 1) /
                                         (df.loc[x.index, 'length'][x == "in"].sum() + 1)),
)

agg = agg.reset_index()
agg.to_csv("cesnet_style_output.csv", index=False)
print("✅ Converted file saved as cesnet_style_output.csv")
