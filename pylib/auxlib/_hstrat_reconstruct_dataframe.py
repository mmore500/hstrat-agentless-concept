import subprocess
import tempfile

import numpy as np
import pandas as pd


def hstrat_reconstruct_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    with tempfile.NamedTemporaryFile(suffix=".pqt") as genomes_f:
        with tempfile.NamedTemporaryFile(suffix=".pqt") as phylo_f:
            df.to_parquet(genomes_f.name, index=False)

            subprocess.run(
                [
                    "singularity",
                    "exec",
                    "docker://ghcr.io/mmore500/hstrat:v1.18.2",
                    "python3",
                    "-m",
                    "hstrat.dataframe.surface_build_tree",
                    "--no-delete-trunk",
                    "--trie-postprocessor",
                    "hstrat.AssignOriginTimeNodeRankTriePostprocessor(t0='dstream_S')",
                    phylo_f.name,
                ],
                capture_output=True,
                check=True,
                input=genomes_f.name.encode(),
            )

            res = pd.read_parquet(phylo_f.name)

            # fix wraparound of negative origin_time
            if pd.api.types.is_unsigned_integer_dtype(res["origin_time"]):
                res["origin_time"] = (
                    res["origin_time"]
                    .to_numpy()
                    .astype(
                        np.dtype(
                            res["origin_time"].dtype.name.replace("u", ""),
                        ),
                        casting="unsafe",
                    )
                )

            return res
