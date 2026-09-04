# uv lock dosyalari: image build `uv sync --frozen` ile kullanir; commit'li kalir.

- isaac-sonic/  Python 3.11 (Isaac Sim Kit interpreter) + Torch 2.7.0 cu128 + Isaac Lab 2.3.2 + SONIC [training]
- sonic-sim/    Python 3.11 + MuJoCo + SONIC [sim]
- groot-n17     upstream Isaac-GR00T pyproject.toml + uv.lock (pinned source tree'den kurulur, buraya kopyalanmaz)

Yeniden uretim: env'in pyproject.toml'unu duzenle -> uv lock -> diff'i review et -> ikisini birlikte commit'le.
