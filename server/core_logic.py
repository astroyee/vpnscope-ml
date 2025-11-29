import numpy as np
import pandas as pd
from scapy.all import rdpcap, IP, TCP, UDP
from collections import defaultdict

FEATURE_COLUMNS = [
    "duration",
    "total_fiat", "total_biat",
    "min_fiat", "min_biat",
    "max_fiat", "max_biat",
    "mean_fiat", "mean_biat",
    "flowPktsPerSecond", "flowBytesPerSecond",
    "min_flowiat", "max_flowiat", "mean_flowiat", "std_flowiat",
    "min_active", "mean_active", "max_active", "std_active",
    "min_idle", "mean_idle", "max_idle", "std_idle"
]

def calculate_stats(values):
    if len(values) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    return np.min(values), np.max(values), np.mean(values), np.std(values), np.sum(values)

def extract_features(pcap_path, idle_timeout=5.0, flow_time_limit=15.0):
    try:
        packets = rdpcap(pcap_path)
    except Exception as e:
        return pd.DataFrame(), pd.DataFrame()

    flows = defaultdict(lambda: {
        'forward': [],
        'backward': [], 
        'all_packets': [],
        'start_time': 0.0,
        'last_time': 0.0,
        'protocol': ''
    })

    for pkt in packets:
        if IP not in pkt:
            continue
            
        timestamp = float(pkt.time)
        src_ip = pkt[IP].src
        dst_ip = pkt[IP].dst
        size = len(pkt)
        
        if TCP in pkt:
            proto = 6
            src_port = pkt[TCP].sport
            dst_port = pkt[TCP].dport
            proto_name = "TCP"
        elif UDP in pkt:
            proto = 17
            src_port = pkt[UDP].sport
            dst_port = pkt[UDP].dport
            proto_name = "UDP"
        else:
            continue

        fwd_key = (src_ip, src_port, dst_ip, dst_port, proto)
        bwd_key = (dst_ip, dst_port, src_ip, src_port, proto)

        if fwd_key in flows:
            if timestamp - flows[fwd_key]['start_time'] <= flow_time_limit:
                flows[fwd_key]['forward'].append((timestamp, size))
                flows[fwd_key]['all_packets'].append((timestamp, size, 1))
                flows[fwd_key]['last_time'] = timestamp
        elif bwd_key in flows:
            if timestamp - flows[bwd_key]['start_time'] <= flow_time_limit:
                flows[bwd_key]['backward'].append((timestamp, size))
                flows[bwd_key]['all_packets'].append((timestamp, size, -1))
                flows[bwd_key]['last_time'] = timestamp
        else:
            flows[fwd_key]['start_time'] = timestamp
            flows[fwd_key]['last_time'] = timestamp
            flows[fwd_key]['protocol'] = proto_name
            flows[fwd_key]['forward'].append((timestamp, size))
            flows[fwd_key]['all_packets'].append((timestamp, size, 1))

    feature_rows = []
    meta_rows = []

    for key, flow_data in flows.items():
        src_ip, _, dst_ip, _, _ = key
        
        all_pkts = sorted(flow_data['all_packets'], key=lambda x: x[0])
        timestamps = [p[0] for p in all_pkts]
        sizes = [p[1] for p in all_pkts]
        
        fwd_times = sorted([p[0] for p in flow_data['forward']])
        if len(fwd_times) > 1:
            fwd_iats = np.diff(fwd_times)
        else:
            fwd_iats = []
            
        bwd_times = sorted([p[0] for p in flow_data['backward']])
        if len(bwd_times) > 1:
            bwd_iats = np.diff(bwd_times)
        else:
            bwd_iats = []
            
        if len(timestamps) > 1:
            flow_iats = np.diff(timestamps)
        else:
            flow_iats = []

        active_times = []
        idle_times = []
        
        current_active_start = timestamps[0]
        
        for i in range(1, len(timestamps)):
            iat = timestamps[i] - timestamps[i-1]
            if iat > idle_timeout:
                idle_times.append(iat)
                active_duration = timestamps[i-1] - current_active_start
                if active_duration > 0:
                    active_times.append(active_duration)
                current_active_start = timestamps[i]
        
        last_active_duration = timestamps[-1] - current_active_start
        if last_active_duration >= 0:
             active_times.append(last_active_duration)
        
        duration = flow_data['last_time'] - flow_data['start_time']
        
        min_fiat, max_fiat, mean_fiat, _, total_fiat = calculate_stats(fwd_iats)
        min_biat, max_biat, mean_biat, _, total_biat = calculate_stats(bwd_iats)
        min_flowiat, max_flowiat, mean_flowiat, std_flowiat, _ = calculate_stats(flow_iats)
        min_active, max_active, mean_active, std_active, _ = calculate_stats(active_times)
        min_idle, max_idle, mean_idle, std_idle, _ = calculate_stats(idle_times)
        
        total_bytes = sum(sizes)
        total_packets = len(sizes)
        
        if duration > 0:
            flowPktsPerSecond = total_packets / duration
            flowBytesPerSecond = total_bytes / duration
        else:
            flowPktsPerSecond = 0.0
            flowBytesPerSecond = 0.0

        row = {
            "duration": duration * 1e6, 
            "total_fiat": total_fiat * 1e6,
            "total_biat": total_biat * 1e6,
            "min_fiat": min_fiat * 1e6,
            "min_biat": min_biat * 1e6,
            "max_fiat": max_fiat * 1e6,
            "max_biat": max_biat * 1e6,
            "mean_fiat": mean_fiat * 1e6,
            "mean_biat": mean_biat * 1e6,
            "flowPktsPerSecond": flowPktsPerSecond,
            "flowBytesPerSecond": flowBytesPerSecond,
            "min_flowiat": min_flowiat * 1e6,
            "max_flowiat": max_flowiat * 1e6,
            "mean_flowiat": mean_flowiat * 1e6,
            "std_flowiat": std_flowiat * 1e6,
            "min_active": min_active * 1e6,
            "mean_active": mean_active * 1e6,
            "max_active": max_active * 1e6,
            "std_active": std_active * 1e6,
            "min_idle": min_idle * 1e6,
            "mean_idle": mean_idle * 1e6,
            "max_idle": max_idle * 1e6,
            "std_idle": std_idle * 1e6
        }
        
        feature_rows.append(row)
        
        meta_rows.append({
            "Source IP": src_ip,
            "Dest IP": dst_ip,
            "Protocol": flow_data['protocol'],
            "Length": total_packets
        })

    df_features = pd.DataFrame(feature_rows)
    df_meta = pd.DataFrame(meta_rows)
    
    for col in FEATURE_COLUMNS:
        if col not in df_features.columns:
            df_features[col] = 0.0
            
    df_features = df_features[FEATURE_COLUMNS]
    
    return df_features, df_meta

def calculate_suspicion_score(df_meta, vpn_probs):
    weights = np.log1p(df_meta['Length'].values)
    
    proto_weights = np.where(df_meta['Protocol'].str.upper() == 'UDP', 1.2, 1.0)
    
    final_weights = weights * proto_weights
    
    if np.sum(final_weights) == 0:
        return 0.0
        
    weighted_prob = np.average(vpn_probs, weights=final_weights)
    
    return weighted_prob * 100