git clone https://github.com/snimu/modded-nanogpt.git
cd modded-nanogpt
git switch mot
pip install uv
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
# uv pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126 --upgrade
python data/cached_fineweb10B.py
