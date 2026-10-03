#!/usr/bin/env python3
import argparse
from pathlib import Path

from src.pipelines.vanilla_pipeline import VanillaPipeline
from src.pipelines.pls_pipeline import PLSPipeline
from src.pipelines.fabo_pipeline import FABOPipeline
from src.pipelines.pca_pipeline import PCAPipeline
from src.pipelines.opls_pipeline import OPLSPipeline
from baselines.outlier_search import OutlierSearchPipeline
from baselines.random_search import RandomSearchPipeline
from src.simulation import EnergySimulator


def parse_list(arg_str, type_fn=str):
    return [type_fn(x) for x in arg_str.split(",") if x]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run Bayesian Optimization with various kernels, acquisitions, and seeds"
    )
    parser.add_argument(
        "--input", "-i", required=True, help="CSV with SMILES + descriptors"
    )
    parser.add_argument(
        "--output-dir", "-o", default="results", help="Base results directory"
    )
    parser.add_argument(
        "--cache", "-c", required=True, help="Path to energy cache JSON"
    )
    parser.add_argument(
        "--n-initial", type=int, required=True, help="Number of initial random samples"
    )
    parser.add_argument(
        "--n-iter", type=int, required=True, help="Number of BO iterations per run"
    )
    parser.add_argument("--seed", type=int, required=True, help="Starting random seed")
    parser.add_argument(
        "--repeats", type=int, default=1, help="How many successive seeds to run"
    )
    parser.add_argument(
        "--mode",
        choices=["vanilla", "pls", "fabo", "pca", "opls", "random", "outlier"],
        required=True,
        help="Which method to run (vanilla = plain GP on all features; "
        "random / outlier = model-free baselines)",
    )
    parser.add_argument(
        "--kernels",
        "-k",
        type=str,
        default="Matern",
        help="Comma-separated GP kernels, e.g. Matern,RBF",
    )
    parser.add_argument(
        "--acquisitions",
        "-a",
        type=str,
        default="UCB",
        help="Comma-separated acquisition functions, e.g. UCB,EI",
    )
    parser.add_argument(
        "--betas",
        "-b",
        type=str,
        default="1.0",
        help="Comma-separated β values for UCB (ignored by EI)",
    )
    # FABO-specific parameters
    parser.add_argument(
        "--fabo-threshold",
        type=float,
        default=None,
        help="Spearman correlation cutoff for FABO; if omitted, chosen automatically",
    )
    parser.add_argument(
        "--fabo-k",
        type=int,
        default=None,
        help="Number of top features for FABO; if omitted, k will be chosen automatically",
    )
    parser.add_argument(
        "--pls-n-components",
        type=int,
        default=None,
        help="Fix the number of PLS components (skip CV tuning if set)",
    )
    parser.add_argument(
        "--pca-n-components",
        type=float,
        default=None,
        help="If set, number of PCA components (int) or fraction of variance (0–1) to keep.",
    )
    parser.add_argument(
        "--pca-whiten",
        action="store_true",
        help="Whiten PCA components (default: False)",
    )

    parser.add_argument(
        "--opls-n-components",
        type=int,
        default=None,
        help="Max predictive components for OPLS; if omitted, chosen adaptively (up to 10)",
    )
    parser.add_argument(
        "--opls-orthogonal",
        type=int,
        default=None,
        help="Fixed number of orthogonal components for OPLS; if omitted, "
        "chosen adaptively between 1 and 3",
    )
    parser.add_argument(
        "--opls-scale",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Standardise X before OPLS (default: on; --no-opls-scale to disable)",
    )
    parser.add_argument(
        "--prune-correlated",
        type=float,
        default=None,
        metavar="THRESHOLD",
        help="Drop constant columns, then greedily drop columns with |Pearson r| > "
        "THRESHOLD against an already kept column (computed on the whole X table, "
        "no targets; default: off)",
    )
    return parser


def build_config(args, kernel, acq, beta, seed):
    return {
        "data": {
            "n_initial": args.n_initial,
            "test_frac": 0.1,
            "prune_correlated": args.prune_correlated,
        },
        "optimization": {
            "n_iter": args.n_iter,
            "kernel": kernel,
            "acquisition": acq,
            "beta": beta,
            "batch_size": 1,
        },
        "random_state": seed,
        "results_dir": args.output_dir,
        "feature_selection": {
            "pls_components": args.pls_n_components,
            "pca_components": args.pca_n_components,
            "pca_whiten": args.pca_whiten,
            "opls_components": args.opls_n_components,
            "opls_orthogonal": args.opls_orthogonal,
            "opls_scale": args.opls_scale,
            "fabo_threshold": args.fabo_threshold,
            "fabo_k": args.fabo_k,
        },
    }


def main():
    args = build_parser().parse_args()

    # parse lists
    kernels = parse_list(args.kernels, str)
    acquisitions = parse_list(args.acquisitions, str)
    betas = parse_list(args.betas, float)

    # choose pipeline class
    pipeline_map = {
        "vanilla": VanillaPipeline,
        "pls": PLSPipeline,
        "fabo": FABOPipeline,
        "pca": PCAPipeline,
        "opls": OPLSPipeline,
        "random": RandomSearchPipeline,
        "outlier": OutlierSearchPipeline,
    }
    PipelineClass = pipeline_map[args.mode]

    # instantiate simulator once
    simulator = EnergySimulator()

    # Model-free baselines have a different interface (no kernel/acquisition loops)
    if args.mode in ("random", "outlier"):
        for run_idx in range(args.repeats):
            seed = args.seed + run_idx
            config = {
                "n_initial": args.n_initial,
                "n_iter": args.n_iter,
                "random_state": seed,
                "results_dir": args.output_dir,
                "test_frac": 0.1,
            }
            print(f"=== RUN: mode={args.mode}, seed={seed} ===")
            pipeline = PipelineClass(config)
            pipeline.run(
                input_csv=args.input,
                cache_path=Path(args.cache),
                simulator=simulator,
            )
    else:
        # loop over all combinations for BO methods
        for kernel in kernels:
            for acq in acquisitions:
                for beta in betas:
                    for run_idx in range(args.repeats):
                        seed = args.seed + run_idx
                        config = build_config(args, kernel, acq, beta, seed)
                        print(
                            f"=== RUN: mode={args.mode}, kernel={kernel}, acq={acq}, beta={beta}, seed={seed} ==="
                        )
                        pipeline = PipelineClass(config)
                        pipeline.run(
                            input_csv=args.input,
                            cache_path=Path(args.cache),
                            simulator=simulator,
                        )


if __name__ == "__main__":
    main()
