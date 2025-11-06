#!/usr/bin/env python3
import argparse
import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def normalize_df(df):
    # Make sure time_sec and proc exist
    if 'time_sec' not in df.columns:
        # try some alternatives
        for alt in ['time', 'time_s', 't']:
            if alt in df.columns:
                df = df.rename(columns={alt: 'time_sec'})
                break

    # proc
    if 'proc' not in df.columns and 'procs' in df.columns:
        df = df.rename(columns={'procs':'proc'})

    # unify sample/size column to 'samples'
    if 'samples' not in df.columns:
        for alt in ('rows', 'size', 'n', 'cols'):
            if alt in df.columns:
                df = df.rename(columns={alt: 'samples'})
                break

    # try numeric conversion
    if 'time_sec' in df.columns:
        df['time_sec'] = pd.to_numeric(df['time_sec'], errors='coerce')
    if 'proc' in df.columns:
        df['proc'] = pd.to_numeric(df['proc'], errors='coerce')
    if 'samples' in df.columns:
        df['samples'] = pd.to_numeric(df['samples'], errors='coerce')

    return df

def plot_time_vs_procs(grouped, samples_list, outdir, tag=""):
    for s in samples_list:
        sub = grouped[grouped['samples'] == s].sort_values('proc')
        if sub.empty:
            continue
        plt.figure()
        # if std available use errorbar
        if 'time_std' in sub.columns:
            yerr = sub['time_std'].fillna(0).values
            plt.errorbar(sub['proc'], sub['time_mean'], yerr=yerr, marker='o', linestyle='-')
        else:
            plt.plot(sub['proc'], sub['time_mean'], marker='o')
        plt.xlabel('Number of processes')
        plt.ylabel('Execution time (s)')
        title = f'Execution time vs processes (size={int(s)})'
        if tag:
            title = f'{tag} — ' + title
        plt.title(title)
        plt.grid(True, which='both', ls='--', lw=0.5)
        fname = os.path.join(outdir, f"time_vs_procs{('_'+tag) if tag else ''}_{int(s)}.png")
        plt.savefig(fname, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Wrote {fname}")

def compute_speedup_eff(grouped):
    records = []
    samples_list = sorted(grouped['samples'].unique())
    for s in samples_list:
        sub = grouped[grouped['samples'] == s].set_index('proc')
        if 1 not in sub.index:
            # cannot compute speedup without proc=1 baseline
            continue
        T1 = sub.loc[1, 'time_mean']
        if T1 <= 0 or np.isnan(T1):
            continue
        for proc, row in sub.iterrows():
            Tproc = row['time_mean']
            if Tproc <= 0 or np.isnan(Tproc):
                sp = np.nan
            else:
                sp = T1 / Tproc
            eff = sp / proc if proc != 0 and not np.isnan(sp) else np.nan
            records.append({'samples': s, 'proc': proc, 'time': Tproc, 'speedup': sp, 'efficiency': eff})
    return pd.DataFrame(records)

def plot_speedup_eff(sp_df, outdir, tag=""):
    if sp_df.empty:
        return
    # speedup
    plt.figure()
    for s in sorted(sp_df['samples'].unique()):
        sub = sp_df[sp_df['samples'] == s].sort_values('proc')
        plt.plot(sub['proc'], sub['speedup'], marker='o', label=f'size={int(s)}')
    procs_line = np.array(sorted(sp_df['proc'].unique()))
    if procs_line.size:
        plt.plot(procs_line, procs_line, linestyle='--', label='ideal linear')
    plt.xlabel('Number of processes')
    plt.ylabel('Speedup (T1 / Tproc)')
    title = 'Speedup vs processes'
    if tag:
        title = f'{tag} — ' + title
    plt.title(title)
    plt.legend()
    plt.grid(True, which='both', ls='--', lw=0.5)
    fname = os.path.join(outdir, f"speedup{('_'+tag) if tag else ''}.png")
    plt.savefig(fname, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Wrote {fname}")

    # efficiency
    plt.figure()
    for s in sorted(sp_df['samples'].unique()):
        sub = sp_df[sp_df['samples'] == s].sort_values('proc')
        plt.plot(sub['proc'], sub['efficiency'], marker='o', label=f'size={int(s)}')
    plt.xlabel('Number of processes')
    plt.ylabel('Efficiency (speedup / #proc)')
    title = 'Parallel efficiency vs processes'
    if tag:
        title = f'{tag} — ' + title
    plt.title(title)
    plt.legend()
    plt.grid(True, which='both', ls='--', lw=0.5)
    fname = os.path.join(outdir, f"efficiency{('_'+tag) if tag else ''}.png")
    plt.savefig(fname, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Wrote {fname}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', '-i', required=True, help='CSV file from run_experiments.sh')
    p.add_argument('--outdir', '-o', default='.', help='Output directory for PNGs')
    args = p.parse_args()

    outdir = args.outdir
    ensure_dir(outdir)

    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        print("Failed to read CSV:", e, file=sys.stderr)
        sys.exit(1)

    df = normalize_df(df)

    # If mode exists, process per-mode
    if 'mode' in df.columns:
        modes = sorted(df['mode'].dropna().unique())
        agg_list = []
        for mode in modes:
            subdf = df[df['mode'] == mode].copy()
            subdf = normalize_df(subdf)  # ensure numeric again
            # aggregate mean/std
            grouped = subdf.groupby(['samples', 'proc'])['time_sec'].agg(['mean','std','count']).reset_index()
            grouped = grouped.rename(columns={'mean':'time_mean','std':'time_std','count':'runs'})
            agg_list.append(grouped.assign(mode=mode))
            samples_list = sorted(grouped['samples'].dropna().unique())
            tag = str(mode)
            plot_time_vs_procs(grouped, samples_list, outdir, tag=tag)
            sp_df = compute_speedup_eff(grouped)
            plot_speedup_eff(sp_df, outdir, tag=tag)
        if agg_list:
            pd.concat(agg_list).to_csv(os.path.join(outdir, "aggregated_times_by_mode.csv"), index=False)
    else:
        # old-style: just use samples and proc
        if 'samples' not in df.columns:
            print("No 'samples' column and no 'mode' column found in CSV.", file=sys.stderr)
            sys.exit(1)
        grouped = df.groupby(['samples', 'proc'])['time_sec'].agg(['mean','std','count']).reset_index()
        grouped = grouped.rename(columns={'mean':'time_mean','std':'time_std','count':'runs'})
        samples_list = sorted(grouped['samples'].dropna().unique())
        plot_time_vs_procs(grouped, samples_list, outdir, tag="")
        sp_df = compute_speedup_eff(grouped)
        plot_speedup_eff(sp_df, outdir, tag="")
        grouped.to_csv(os.path.join(outdir, "aggregated_times.csv"), index=False)

if __name__ == "__main__":
    main()
