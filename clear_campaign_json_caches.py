#!/usr/bin/env python
"""Delete the aggregated-results JSON caches inside campaign/ subdirectories.

The figures notebook (charge_trap_figures.ipynb) caches the aggregated per-run
results for each scenario in `aggregated_results_<mask>_<event>.json` files,
keyed off the pickles in that run directory. When new simulation pickles are
added to a scenario the cache is stale, so deleting it forces the notebook to
re-aggregate from the current set of pickles on the next load.

Usage:
    python clear_campaign_json_caches.py            # delete the caches
    python clear_campaign_json_caches.py --dry-run  # list what would be deleted
    python clear_campaign_json_caches.py --root some/other/campaign_dir
"""

import argparse
import os
import sys

CACHE_GLOB = "aggregated_results_*.json"


def find_caches(root):
    """Yield every aggregated_results_*.json under root (recursively)."""
    import fnmatch

    for dirpath, _dirnames, filenames in os.walk(root):
        for name in fnmatch.filter(filenames, CACHE_GLOB):
            yield os.path.join(dirpath, name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="campaign",
                        help="Directory to search (default: campaign)")
    parser.add_argument("--dry-run", action="store_true",
                        help="List the files that would be deleted, but don't delete.")
    args = parser.parse_args(argv)

    if not os.path.isdir(args.root):
        print(f"error: '{args.root}' is not a directory", file=sys.stderr)
        return 1

    caches = sorted(find_caches(args.root))
    if not caches:
        print(f"No '{CACHE_GLOB}' caches found under '{args.root}'.")
        return 0

    verb = "Would delete" if args.dry_run else "Deleting"
    for path in caches:
        print(f"{verb}: {path}")
        if not args.dry_run:
            os.remove(path)

    action = "would be deleted" if args.dry_run else "deleted"
    print(f"\n{len(caches)} cache file(s) {action}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
