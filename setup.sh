git clone https://github.com/snimu/modded-nanogpt.git
cd modded-nanogpt
git switch f-of-v-exp
git clone https://github.com/KellerJordan/modded-nanogpt.git && cd modded-nanogpt
pip install -r requirements.txt
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126 --upgrade
# downloads only the first 800M training tokens to save time
python data/cached_fineweb10B.py 8
