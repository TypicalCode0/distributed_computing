#!/usr/bin/env python3

"""
Usage:
  python3 plot_results.py --input results_task_1_....csv --outdir plots
"""
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', '-i', required=True, help='CSV file from run_experiments.sh')
    p.add_argument('--outdir', '-o', default='.', help='Output directory for PNGs')
    args = p.parse_args()

    df = pd.read_csv(args.input)
    # ensure numeric types
    df['time_sec'] = pd.to_numeric(df['time_sec'], errors='coerce')
    df['proc'] = pd.to_numeric(df['proc'], errors='coerce')
    df['samples'] = pd.to_numeric(df['samples'], errors='coerce')

    outdir = args.outdir
    ensure_dir(outdir)

    # For each sample size, average time over runs
    grouped = df.groupby(['samples', 'proc'])['time_sec'].agg(['mean','std','count']).reset_index()
    grouped.rename(columns={'mean':'time_mean','std':'time_std','count':'runs'}, inplace=True)

    # Plot time vs procs for each samples
    samples_list = sorted(grouped['samples'].unique())
    for s in samples_list:
        sub = grouped[grouped['samples'] == s].sort_values('proc')
        plt.figure()
        plt.errorbar(sub['proc'], sub['time_mean'], yerr=sub['time_std'], marker='o', linestyle='-')
        plt.xscale('linear')
        plt.xlabel('Number of processes')
        plt.ylabel('Execution time (s)')
        plt.title(f'Execution time vs processes (samples={int(s)})')
        plt.grid(True, which='both', ls='--', lw=0.5)
        fname = os.path.join(outdir, f"time_vs_procs_{int(s)}.png")
        plt.savefig(fname, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Wrote {fname}")

    # Compute speedup and efficiency:
    # speedup(proc) = T(1)/T(proc) for same samples.
    speedup_records = []
    for s in samples_list:
        sub = grouped[grouped['samples'] == s].set_index('proc')
        if 1 not in sub.index:
            print(f"Warning: no measurement for proc=1 for samples={s} - cannot compute speedup/efficiency for this samples")
            continue
        T1 = sub.loc[1, 'time_mean']
        for proc, row in sub.iterrows():
            Tproc = row['time_mean']
            if Tproc <= 0 or np.isnan(Tproc) or T1 <= 0 or np.isnan(T1):
                sp = np.nan
            else:
                sp = T1 / Tproc
            eff = sp / proc if not np.isnan(sp) else np.nan
            speedup_records.append({'samples': s, 'proc': proc, 'time': Tproc, 'speedup': sp, 'efficiency': eff})
    sp_df = pd.DataFrame(speedup_records)

    # Plot speedup for each samples
    plt.figure()
    for s in sorted(sp_df['samples'].unique()):
        sub = sp_df[sp_df['samples'] == s].sort_values('proc')
        plt.plot(sub['proc'], sub['speedup'], marker='o', label=f'samples={int(s)}')
    # also plot ideal line (linear speedup)
    max_proc = int(sp_df['proc'].max()) if not sp_df.empty else 1
    procs_line = np.array(sorted(sp_df['proc'].unique()))
    plt.plot(procs_line, procs_line, linestyle='--', label='ideal linear')
    plt.xscale('linear')
    plt.xlabel('Number of processes')
    plt.ylabel('Speedup (T1 / Tproc)')
    plt.title('Speedup vs processes')
    plt.legend()
    plt.grid(True, which='both', ls='--', lw=0.5)
    fname = os.path.join(outdir, "speedup.png")
    plt.savefig(fname, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Wrote {fname}")

    # Plot efficiency for each samples
    plt.figure()
    for s in sorted(sp_df['samples'].unique()):
        sub = sp_df[sp_df['samples'] == s].sort_values('proc')
        plt.plot(sub['proc'], sub['efficiency'], marker='o', label=f'samples={int(s)}')
    plt.xscale('linear')
    plt.xlabel('Number of processes')
    plt.ylabel('Efficiency (speedup / #proc)')
    plt.title('Parallel efficiency vs processes')
    plt.legend()
    plt.grid(True, which='both', ls='--', lw=0.5)
    fname = os.path.join(outdir, "efficiency.png")
    plt.savefig(fname, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Wrote {fname}")

    agg_fname = os.path.join(outdir, "aggregated_times.csv")
    grouped.to_csv(agg_fname, index=False)
    print(f"Wrote aggregated times to {agg_fname}")

if __name__ == "__main__":
    main()
