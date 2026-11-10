"""CN-mirror-first checkpoint fetching — shared CLI / load-time entry.

Sources: ModelScope (modelscope.cn) hosts the timm/ namespace repos in the
exact timm format; hf-mirror.com mirrors everything else (SatDINO).
huggingface.co direct is last resort only — on CN networks it usually times
out. scripts/dl_dino.py is the bulk-download CLI; DinoTower calls ensure_*()
on first use so even a bare `run_24g.py` on a fresh clone never touches
timm's HF-hub channel.

Importing this module forces IPv4 DNS (WSL2/some intranets resolve IPv6 but
cannot route it).
"""
import os
import socket
import subprocess

_orig = socket.getaddrinfo
def _ipv4(host, port, family=0, type=0, proto=0, flags=0):
    return _orig(host, port, socket.AF_INET, type, proto, flags)
socket.getaddrinfo = _ipv4

MS_URL = 'https://modelscope.cn/models/timm/{name}/resolve/master/model.safetensors'
# strakaj/SatDINO (arXiv 2508.21402): fMoW-RGB self-supervised DINO — the
# RS-pretrained DINO-family comparison point (facebook-DINO 'teacher' ckpt).
# ModelScope hosts the repo page but no files; hf-mirror serves them.
SATDINO_URLS = [
    'https://hf-mirror.com/strakajk/satdino-vit_base-16/resolve/main/satdino-vit_base-16.pth',
    'https://huggingface.co/strakajk/satdino-vit_base-16/resolve/main/satdino-vit_base-16.pth',
]


def dl_url(url, dst):
    # -L: hf-mirror 302-redirects to the CDN; without it the HEAD content-length
    # is the redirect page's (~1KB), not the file's, and the size check misfires
    head = subprocess.run(['curl', '-4', '-sIL', '--max-time', '15', url],
                          capture_output=True, text=True, check=True).stdout
    size = next((int(l.split(':', 1)[1]) for l in reversed(head.splitlines())
                 if l.lower().startswith('content-length')), 0)
    tmp = dst + '.dl'
    subprocess.run(['curl', '-4', '-sL', '--retry', '3', '--retry-delay', '2',
                    '-o', tmp, url], check=True)
    got = os.path.getsize(tmp)
    if size and got != size:
        raise RuntimeError(f'size mismatch: got {got}, expect {size}')
    os.replace(tmp, dst)


def dl_first(urls, dst):
    """Try each URL in order (CN mirrors first); return the one that worked."""
    err = None
    for u in urls:
        try:
            dl_url(u, dst)
            return u
        except Exception as e:
            err = e
    raise err


def dl_hf(timm_name):
    # 直连 huggingface.co 需 IPv4;hf-mirror 的 308 跳转会破坏
    # huggingface_hub>=1.x 的 HEAD 元数据校验,不作默认。
    os.environ.setdefault('HF_ENDPOINT', 'https://huggingface.co')
    import timm
    timm.create_model(timm_name, pretrained=True, num_classes=0)


def ensure_timm_ckpt(dst, timm_name):
    """Make dst exist: ModelScope first, huggingface.co timm-hub fallback."""
    if os.path.exists(dst) and os.path.getsize(dst) > 1e8:
        return
    try:
        dl_url(MS_URL.format(name=timm_name), dst)
        print(f'[weights_dl] {timm_name} via ModelScope -> {dst}')
    except Exception as e:
        print(f'[weights_dl] ModelScope failed ({type(e).__name__}: {e}); '
              f'falling back to huggingface.co timm-hub')
        dl_hf(timm_name)


def ensure_satdino_ckpt(dst):
    """Make dst exist: hf-mirror first, huggingface.co fallback. No timm-hub
    fallback — that would fetch plain OpenAI weights, not SatDINO."""
    if os.path.exists(dst) and os.path.getsize(dst) > 1e8:
        return
    u = dl_first(SATDINO_URLS, dst)
    print(f'[weights_dl] satdino via {u.split("/")[2]} -> {dst}')
