"""Command line interface for PlateForge."""

import argparse
import os
import sys

from plateforge import __version__


def build_parser():
    parser = argparse.ArgumentParser(
        prog="plateforge",
        description="Read number plates from CCTV video.",
    )
    parser.add_argument("--version", action="version", version=f"PlateForge {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    info = sub.add_parser("info", help="Show video metadata")
    info.add_argument("video")

    detect = sub.add_parser("detect", help="Detect plate regions in a video")
    detect.add_argument("video")
    detect.add_argument("--every-seconds", type=float, default=1.0)
    detect.add_argument("--output-dir", default="detections")

    read = sub.add_parser("read", help="Detect and read plates from a video")
    read.add_argument("video")
    read.add_argument("--every-seconds", type=float, default=1.0)
    read.add_argument("--min-confidence", type=float, default=0.8)
    read.add_argument("--output-dir", default="readings")

    log_cmd = sub.add_parser("log", help="Show the plates in a log file")
    log_cmd.add_argument("log_file")

    search = sub.add_parser("search", help="Search a log by plate or time")
    search.add_argument("log_file")
    search.add_argument("--plate", default=None)
    search.add_argument("--start", type=float, default=None, dest="start_time")
    search.add_argument("--end", type=float, default=None, dest="end_time")

    batch = sub.add_parser("batch", help="Process many videos from a list file")
    batch.add_argument("list_file")
    batch.add_argument("--output-dir", default="results")
    batch.add_argument("--every-seconds", type=float, default=1.0)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "info":
        from plateforge.video import get_video_info

        info = get_video_info(args.video)
        print(f"Video: {info.path}")
        print(f"Duration: {info.duration_seconds:.2f}s")
        print(f"FPS: {info.fps}")
        print(f"Frames: {info.frame_count}")
        print(f"Resolution: {info.width}x{info.height}")
        print(f"Codec: {info.codec}")
        return 0

    if args.command == "detect":
        from plateforge.detect import filter_candidates, find_plate_candidates
        from plateforge.extract import extract_by_interval

        total = 0
        for index, timestamp, frame in extract_by_interval(args.video, every_seconds=args.every_seconds):
            found = filter_candidates(find_plate_candidates(frame), frame.shape)
            total += len(found)
            for cand in found:
                print(f"{timestamp:.2f}s frame {index}: plate at {cand.box} score {cand.score}")
        print(f"Plate regions found: {total}")
        return 0

    if args.command == "read":
        from plateforge.batch import process_video

        entries = process_video(args.video, args.output_dir, args.every_seconds, args.min_confidence)
        for entry in entries:
            print(f"{entry.video_time:.2f}s {entry.plate} ({entry.confidence:.2f})")
        print(f"Plates logged: {len(entries)} in {args.output_dir}")
        return 0

    if args.command == "log":
        from plateforge.log import load_csv, load_json

        path = args.log_file
        entries = load_json(path) if path.endswith(".json") else load_csv(path)
        for entry in entries:
            print(f"{entry.video_time:.2f}s {entry.plate} ({entry.confidence:.2f})")
        print(f"Entries: {len(entries)}")
        return 0

    if args.command == "search":
        from plateforge.log import load_csv, load_json, search_log

        path = args.log_file
        entries = load_json(path) if path.endswith(".json") else load_csv(path)
        found = search_log(entries, plate=args.plate, start_time=args.start_time, end_time=args.end_time)
        for entry in found:
            print(f"{entry.video_time:.2f}s {entry.plate} ({entry.confidence:.2f})")
        print(f"Matches: {len(found)}")
        return 0

    if args.command == "batch":
        from plateforge.batch import process_batch_file

        results = process_batch_file(args.list_file, args.output_dir, args.every_seconds)
        total = sum(len(v) for v in results.values())
        for video, entries in results.items():
            print(f"{video}: {len(entries)} plates")
        print(f"Total plates: {total}")
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
