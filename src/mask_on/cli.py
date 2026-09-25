"""Command-line interface; model runtimes are imported only on execution."""

import argparse
import json

from .download import catalog


def parser():
    p = argparse.ArgumentParser(prog="mask-on")
    s = p.add_subparsers(dest="command", required=True)
    s.add_parser("list", help="List pinned models, datasets and protocols")
    d = s.add_parser("download", help="Download model pairs and prepare benchmark records")
    d.add_argument("--models", nargs="*", choices=catalog()["models"], default=[])
    d.add_argument("--tasks", nargs="*", choices=catalog()["datasets"], default=[])
    d.add_argument("--dry-run", action="store_true")
    d.add_argument(
        "--metadata-only",
        action="store_true",
        help="Download only code/tokenizer/config, not weights",
    )
    n = s.add_parser("native-source", help="Fetch a pinned native decoder source checkout")
    n.add_argument("model", choices=["sdar", "wedlm"])
    panel = s.add_parser("calibration-panel", help="Prepare a disjoint Nemotron 128/32 panel")
    panel.add_argument("--output", required=True)
    panel.add_argument("--seed", type=int, default=20260908)
    c = s.add_parser(
        "collect", help="Save endpoint completions then static-clean projection inputs"
    )
    c.add_argument("--model", choices=catalog()["models"], required=True)
    c.add_argument("--panel", required=True)
    c.add_argument("--output", required=True)
    c.add_argument("--device", default="cuda:0")
    f = s.add_parser("fit", help="Fit Mask-On scales, selecting alpha using fit error only")
    f.add_argument("--model", choices=catalog()["models"], required=True)
    f.add_argument("--collection", required=True)
    f.add_argument("--output", required=True)
    f.add_argument("--alpha", type=float, nargs="+", help="Default: .05, .10, ..., .95")
    f.add_argument("--device", default="cpu")
    m = s.add_parser("prepare", help="Create a hash-bound merged checkpoint")
    m.add_argument("--model", choices=catalog()["models"], required=True)
    m.add_argument(
        "--method",
        required=True,
        choices=[
            "task_arithmetic",
            "ties",
            "dare",
            "t_switch",
            "mask_on",
            "adamerging",
            "adamerging_plus_plus",
            "aim",
            "regmean",
            "bitdelta",
            "delta_come",
        ],
    )
    m.add_argument("--output", required=True)
    m.add_argument(
        "--state", help="Verified fitted-state directory; required by calibrated methods"
    )
    m.add_argument("--beta", type=float)
    m.add_argument("--alpha", type=float, default=0.5)
    m.add_argument("--drop", type=float, default=0.5)
    m.add_argument("--vocabulary", choices=["full", "endpoint"])
    a = s.add_parser("ar-view", help="Materialize a native AR config/weight view of a candidate")
    a.add_argument("--model", choices=catalog()["models"], required=True)
    a.add_argument("--checkpoint", required=True)
    a.add_argument("--output", required=True)
    e = s.add_parser("evaluate", help="Run one atomic benchmark in native diffusion or AR mode")
    e.add_argument("--model", choices=catalog()["models"], required=True)
    e.add_argument("--task", choices=catalog()["datasets"], required=True)
    e.add_argument("--mode", choices=["ar", "diffusion"], default="diffusion")
    e.add_argument("--backend", choices=["native", "vllm"], default="native")
    e.add_argument("--checkpoint")
    e.add_argument("--output", required=True)
    e.add_argument("--limit", type=int)
    e.add_argument("--device", default="cuda:0")
    sc = s.add_parser("score", help="CPU scoring of a completed generation bundle")
    sc.add_argument("root")
    r = s.add_parser("results", help="Print metrics and optionally export CSV")
    r.add_argument("roots", nargs="+")
    r.add_argument("--csv")
    dr = s.add_parser("draft-review", help="GSM8K Fast draft followed by strict native AR review")
    dr.add_argument("--checkpoint", required=True)
    dr.add_argument("--output", required=True)
    dr.add_argument("--threshold", type=float, choices=[0.8, 0.9, 1.0], default=0.9)
    dr.add_argument("--limit", type=int)
    dr.add_argument("--device", default="cuda:0")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    kw = vars(args).copy()
    command = kw.pop("command")
    if command == "list":
        result = catalog()
    elif command == "download":
        from .download import download

        result = download(**kw)
    elif command == "native-source":
        from .download import native_source

        result = str(native_source(args.model, fetch=True))
    elif command == "calibration-panel":
        from .calibration import build_panel

        result = build_panel(**kw)
    elif command == "collect":
        from .calibration import collect

        result = collect(**kw)
    elif command == "fit":
        from .calibration import fit, GRID

        kw["alphas"] = kw.pop("alpha") or GRID
        result = fit(**kw)
    elif command == "prepare":
        from .prepare import prepare

        result = prepare(**kw)
    elif command == "ar-view":
        from .checkpoints import ar_view

        result = ar_view(**kw)
    elif command == "evaluate":
        from .evaluate import evaluate

        result = evaluate(**kw)
    elif command == "score":
        from .evaluate import score

        result = score(**kw)
    elif command == "results":
        from .evaluate import results

        result = results(args.roots, args.csv)
    elif command == "draft-review":
        from .draft_review import run

        result = run(**kw)
    else:
        raise ValueError(f"Unknown command: {command}")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
