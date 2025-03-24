import subprocess
import tempfile

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
                    phylo_f.name,
                ],
                check=True,
                input=genomes_f.name.encode(),
            )

            return pd.read_parquet(phylo_f.name)
