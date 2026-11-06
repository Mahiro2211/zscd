"""Download the three CLIP ViT-L/14 checkpoints — mainland-China friendly.

Sources (all verified reachable from CN networks):
  remote / geors : HuggingFace via hf-mirror.com (domestic mirror; default
                   HF_ENDPOINT, override with a reachable endpoint if needed)
  openai         : openaipublic.azureedge.net (Azure public CDN, directly
                   reachable in CN — no mirror required)

Usage:
  python scripts/dl_hf.py remote
  python scripts/dl_hf.py geors
  python scripts/dl_hf.py openai
"""
import os
import shutil
import socket
import sys

# WSL2/部分内网: IPv6 解析正常但不可达,强制 IPv4
_orig = socket.getaddrinfo
def _ipv4(host, port, family=0, type=0, proto=0, flags=0):
    return _orig(host, port, socket.AF_INET, type, proto, flags)
socket.getaddrinfo = _ipv4

# 国内默认走 hf-mirror(云端在海外可 export HF_ENDPOINT=https://huggingface.co)
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

from huggingface_hub import hf_hub_download  # noqa: E402

CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checkpoints')

JOBS = {
    'remote': dict(repo='chendelong/RemoteCLIP', fn='RemoteCLIP-ViT-L-14.pt',
                   dst=os.path.join(CKPT_DIR, 'RemoteCLIP-ViT-L-14.pt')),
    'geors': dict(repo='Zilun/GeoRSCLIP', fn='ckpt/RS5M_ViT-L-14-336.pt',
                  dst=os.path.join(CKPT_DIR, 'GeoRSCLIP-ViT-L-14-336.pt')),
}
OPENAI_URL = ('https://openaipublic.azureedge.net/clip/models/'
              'b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt')


def dl_openai():
    import urllib.request
    dst_dir = os.path.expanduser('~/.cache/clip')
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, 'ViT-L-14.pt')
    if os.path.exists(dst) and os.path.getsize(dst) == 932767534:
        print('openai already present:', dst)
        return
    print('downloading OpenAI ViT-L/14 (Azure CDN, ~890MB) ...')
    urllib.request.urlretrieve(OPENAI_URL, dst)
    print('openai OK ->', dst)


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'remote'
    if which == 'openai':
        dl_openai()
    else:
        job = JOBS[which]
        p = hf_hub_download(job['repo'], job['fn'], cache_dir=os.path.join(CKPT_DIR, 'hf_cache'))
        shutil.copy(p, job['dst'])
        print(which, 'OK ->', job['dst'])
