#!/usr/bin/env python3
"""Download the NITF sample files the corpus tests run against.

Two public sets are fetched into ``samples/``:

jitc21
    The JITC NITF 2.1 / NSIF 1.0 conformance samples, mirrored in the
    codice/imaging-nitf repository. Each file comes with one reference PNG per
    image segment, which is what makes pixel-exact testing possible.
gdal
    GDAL's NITF driver test data: NITF 2.0 and 1.1 files, odd extension
    layouts, and deliberately corrupt inputs for error handling.

Neither set is redistributed with this package. Both are US government or
open-source test data; the originals are available from JITC on request.
"""

import argparse
import concurrent.futures
import json
import os
import sys
import urllib.error
import urllib.request

SETS = {
    "jitc21": (
        "https://api.github.com/repos/codice/imaging-nitf/contents/"
        "shared-test-resources/src/main/resources/JitcNitf21Samples"
    ),
    "gdal": (
        "https://api.github.com/repos/OSGeo/gdal/contents/"
        "autotest/gdrivers/data/nitf"
    ),
}

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "samples")


def listing(url):
    request = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        # Only needed if the unauthenticated rate limit has been hit.
        request.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def fetch(entry, target):
    path = os.path.join(target, entry["name"])
    if os.path.exists(path) and os.path.getsize(path) == entry["size"]:
        return "cached", entry["name"]
    try:
        urllib.request.urlretrieve(entry["download_url"], path)
    except (urllib.error.URLError, OSError) as exc:
        return "failed: %s" % exc, entry["name"]
    return "downloaded", entry["name"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "sets", nargs="*", choices=sorted(SETS),
        help="which sample sets to fetch (default: all)",
    )
    parser.add_argument("--jobs", type=int, default=8, help="parallel downloads")
    args = parser.parse_args(argv)
    chosen = args.sets or sorted(SETS)

    failures = 0
    for name in chosen:
        target = os.path.join(ROOT, name)
        os.makedirs(target, exist_ok=True)
        print("fetching %s -> %s" % (name, target))
        try:
            entries = [e for e in listing(SETS[name]) if e.get("type") == "file"]
        except urllib.error.URLError as exc:
            print("  could not list the set: %s" % exc, file=sys.stderr)
            failures += 1
            continue
        with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
            results = list(pool.map(lambda e: fetch(e, target), entries))
        downloaded = sum(1 for status, _ in results if status == "downloaded")
        cached = sum(1 for status, _ in results if status == "cached")
        bad = [(status, n) for status, n in results if status.startswith("failed")]
        total = sum(
            os.path.getsize(os.path.join(target, e["name"]))
            for e in entries
            if os.path.exists(os.path.join(target, e["name"]))
        )
        print(
            "  %d downloaded, %d already present, %d failed, %.1f MB on disk"
            % (downloaded, cached, len(bad), total / 1048576.0)
        )
        for status, filename in bad:
            print("    %s: %s" % (filename, status), file=sys.stderr)
        failures += len(bad)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
