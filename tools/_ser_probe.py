import json, glob, os, sys

pattern = sys.argv[1] if len(sys.argv) > 1 else 'campaign/minos_baseline_vp3_expind_pre_clearseq_shuf*'
dirs = sorted(glob.glob(pattern))
for d in dirs:
    f = os.path.join(d, 'aggregated_results_Halo+Bleed+HotColumn+HotPixel_1e.json')
    if not os.path.exists(f):
        continue
    j = json.load(open(f))
    print("===", os.path.basename(d))
    print(f"{'exp':>6} {'SER_trap':>12} {'SER_notr':>12} {'excess':>12} {'ratio':>8}")
    for e in ['0', '4', '6', '10', '20']:
        ct = j['total_counts_traps'][e]; pt = j['total_pix_traps'][e]
        cn = j['total_counts_notraps'][e]; pn = j['total_pix_notraps'][e]
        st = ct / pt; sn = cn / pn; ex = st - sn
        print(f"{e:>6} {st:12.4e} {sn:12.4e} {ex:12.4e} {st/sn:8.4f}")
