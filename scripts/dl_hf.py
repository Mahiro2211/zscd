"""Download HF checkpoints with IPv4 forced (WSL2: IPv6 resolves but is unreachable)."""
import socket
import sys

_orig = socket.getaddrinfo
def _ipv4(host, port, family=0, type=0, proto=0, flags=0):
    return _orig(host, port, socket.AF_INET, type, proto, flags)
socket.getaddrinfo = _ipv4

from huggingface_hub import hf_hub_download
import shutil

jobs = {
    'remote': ('chendelong/RemoteCLIP', 'RemoteCLIP-ViT-L-14.pt',
               '/home/dhm/zero_shot_dg/zscd/checkpoints/RemoteCLIP-ViT-L-14.pt'),
    'geors': ('Zilun/GeoRSCLIP', 'ckpt/RS5M_ViT-L-14-336.pt',
              '/home/dhm/zero_shot_dg/zscd/checkpoints/GeoRSCLIP-ViT-L-14-336.pt'),
}
which = sys.argv[1] if len(sys.argv) > 1 else 'remote'
repo, fn, dst = jobs[which]
p = hf_hub_download(repo, fn, cache_dir='/home/dhm/zero_shot_dg/zscd/checkpoints/hf_cache')
shutil.copy(p, dst)
print(which, 'OK ->', dst)
