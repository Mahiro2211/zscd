"""Pre-download the DINOv2 timm weights.

Primary source: ModelScope (modelscope.cn) — hosts the timm/ namespace repos
in the exact timm format, fast on CN networks. Fallback: huggingface.co via
timm itself (populates the default HF cache). Files land in checkpoints/ as
DINOv2-ViT-L-14-reg4.safetensors / DINOv2-ViT-L-14.safetensors — DinoTower
prefers these local files, so runs afterwards are fully offline.

Usage:
  python scripts/dl_dino.py           # both variants (~1.2GB each)
  python scripts/dl_dino.py dinov2-reg
"""
import os
import socket
import subprocess
import sys

# WSL2/部分内网: IPv6 解析正常但不可达,强制 IPv4
_orig = socket.getaddrinfo
def _ipv4(host, port, family=0, type=0, proto=0, flags=0):
    return _orig(host, port, socket.AF_INET, type, proto, flags)
socket.getaddrinfo = _ipv4

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from zscd.models.dino_tower import DINO_MODELS, LOCAL_CKPTS  # noqa: E402

MS_URL = 'https://modelscope.cn/models/timm/{name}/resolve/master/model.safetensors'


def dl_modelscope(timm_name, dst):
    url = MS_URL.format(name=timm_name)
    head = subprocess.run(['curl', '-4', '-sI', '--max-time', '15', url],
                          capture_output=True, text=True, check=True).stdout
    size = next((int(l.split(':', 1)[1]) for l in head.splitlines()
                 if l.lower().startswith('content-length')), 0)
    tmp = dst + '.dl'
    subprocess.run(['curl', '-4', '-sL', '--retry', '3', '--retry-delay', '2',
                    '-o', tmp, url], check=True)
    got = os.path.getsize(tmp)
    if size and got != size:
        raise RuntimeError(f'size mismatch: got {got}, expect {size}')
    os.replace(tmp, dst)


def dl_hf(timm_name):
    # 这台 WSL2 直连 huggingface.co 可达(须强制 IPv4);hf-mirror 的 308 跳转会破坏
    # huggingface_hub>=1.x 的 HEAD 元数据校验,不作默认。
    os.environ.setdefault('HF_ENDPOINT', 'https://huggingface.co')
    import timm
    timm.create_model(timm_name, pretrained=True, num_classes=0)


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    names = list(DINO_MODELS) if which == 'all' else [which]
    for n in names:
        dst = LOCAL_CKPTS[n]
        if os.path.exists(dst) and os.path.getsize(dst) > 1e9:
            print(n, 'already present:', dst)
            continue
        try:
            dl_modelscope(DINO_MODELS[n], dst)
            print(n, 'OK via ModelScope ->', dst)
        except Exception as e:
            print(f'{n}: ModelScope failed ({type(e).__name__}: {e}); falling back to huggingface.co')
            dl_hf(DINO_MODELS[n])
            print(n, 'OK via huggingface.co (HF cache)')
