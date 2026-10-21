import argparse
import json
import math
import os
import queue
import shutil
import threading
import time
import urllib.request

UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'


def make_req(url, extra=None, method=None):
    headers = {'User-Agent': UA}
    if extra:
        headers.update(extra)
    return urllib.request.Request(url, headers=headers, method=method)


def probe_total(url):
    try:
        with urllib.request.urlopen(make_req(url, method='HEAD'), timeout=30) as r:
            n = r.headers.get('Content-Length')
            return int(n) if n else None
    except Exception:
        return None


def probe_range(url):
    try:
        with urllib.request.urlopen(make_req(url, {'Range': 'bytes=0-0'}), timeout=30) as r:
            cr = r.headers.get('Content-Range')
            if r.status == 206 and cr and '/' in cr:
                return int(cr.rsplit('/', 1)[1])
    except Exception:
        return None
    return None


class Status:
    def __init__(self, out, total):
        self.lock = threading.Lock()
        self.done = 0
        self.total = total
        self.t0 = time.time()
        self.last_t = self.t0
        self.last_d = 0
        self.last_write = 0.0
        self.path = out + '.status.json'

    def add(self, n):
        with self.lock:
            self.done += n

    def snapshot(self, force=False):
        now = time.time()
        with self.lock:
            d = self.done
        inst = (d - self.last_d) / max(now - self.last_t, 1e-9)
        self.last_t, self.last_d = now, d
        if not force and now - self.last_write < 2:
            return None
        self.last_write = now
        pct = 100.0 * d / self.total if self.total else 0.0
        info = {'done': d, 'total': self.total, 'pct': round(pct, 2),
                'inst_kbps': int(inst / 1024), 'ts': time.strftime('%H:%M:%S')}
        tmp = self.path + '.tmp'
        try:
            with open(tmp, 'w') as f:
                json.dump(info, f)
            os.replace(tmp, self.path)
        except Exception:
            pass
        return info


def run_single(url, out, status):
    tmp = out + '.tmp'
    while True:
        total = 0
        got = 0
        try:
            with urllib.request.urlopen(make_req(url), timeout=90) as r:
                total = int(r.headers.get('Content-Length') or 0)
                if total:
                    status.total = total
                with open(tmp, 'wb') as f:
                    while True:
                        b = r.read(262144)
                        if not b:
                            break
                        f.write(b)
                        got += len(b)
                        status.add(len(b))
                        status.snapshot()
            if total and got == total:
                os.replace(tmp, out)
                return True
            print('single: incomplete %d/%d, retry' % (got, total), flush=True)
        except Exception as e:
            print('single error:', type(e).__name__, e, flush=True)
        status.add(-got)
        time.sleep(10)


def run_parallel(url, out, conns, chunk_mb, total, status):
    parts = out + '.parts'
    os.makedirs(parts, exist_ok=True)
    chunk = int(chunk_mb * 1024 * 1024)
    nchunks = math.ceil(total / chunk)
    q = queue.Queue()
    have = 0
    for i in range(nchunks):
        start = i * chunk
        end = min(start + chunk, total) - 1
        p = os.path.join(parts, '%05d' % i)
        want = end - start + 1
        if os.path.exists(p) and os.path.getsize(p) == want:
            have += want
            continue
        q.put((i, start, end))
    status.done = have
    print('parallel: %d chunks, %d pending, %d conns' % (nchunks, q.qsize(), conns), flush=True)

    def worker():
        while True:
            try:
                i, start, end = q.get_nowait()
            except queue.Empty:
                return
            p = os.path.join(parts, '%05d' % i)
            want = end - start + 1
            for _ in range(100000):
                try:
                    with urllib.request.urlopen(make_req(url, {'Range': 'bytes=%d-%d' % (start, end)}), timeout=90) as r:
                        got = 0
                        with open(p, 'wb') as f:
                            while got < want:
                                b = r.read(min(262144, want - got))
                                if not b:
                                    break
                                f.write(b)
                                got += len(b)
                                status.add(len(b))
                                status.snapshot()
                        if got < want:
                            status.add(-got)
                    if os.path.getsize(p) == want:
                        break
                except Exception as e:
                    print('chunk %d error: %s %s' % (i, type(e).__name__, e), flush=True)
                    try:
                        if os.path.exists(p):
                            wasted = os.path.getsize(p)
                            status.add(-wasted)
                            os.remove(p)
                    except Exception:
                        pass
                time.sleep(5)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(conns)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for i in range(nchunks):
        p = os.path.join(parts, '%05d' % i)
        want = min((i + 1) * chunk, total) - i * chunk
        if not os.path.exists(p) or os.path.getsize(p) != want:
            print('chunk %d missing, abort merge' % i, flush=True)
            return False
    with open(out, 'wb') as w:
        for i in range(nchunks):
            p = os.path.join(parts, '%05d' % i)
            with open(p, 'rb') as f:
                shutil.copyfileobj(f, w, 4 * 1024 * 1024)
    if os.path.getsize(out) != total:
        print('merge size mismatch', flush=True)
        return False
    shutil.rmtree(parts, ignore_errors=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--conns', type=int, default=12)
    ap.add_argument('--chunk-mb', type=float, default=4.0)
    ap.add_argument('--force-single', action='store_true')
    a = ap.parse_args()
    total = probe_total(a.url)
    rangable = probe_range(a.url) if not a.force_single else None
    if total is None:
        total = rangable
    if total:
        print('total size: %.1f MB' % (total / 1e6), flush=True)
    if os.path.exists(a.out) and total and os.path.getsize(a.out) == total:
        print('already complete:', a.out, flush=True)
        return
    status = Status(a.out, total or 1)
    status.snapshot(force=True)
    ok = False
    if rangable and total and total > 8 * 1024 * 1024:
        ok = run_parallel(a.url, a.out, a.conns, a.chunk_mb, total, status)
    else:
        ok = run_single(a.url, a.out, status)
    st = status.snapshot(force=True) or {}
    if ok:
        print('DONE %s %d bytes in %.1f min' % (a.out, st.get('done', -1), (time.time() - status.t0) / 60), flush=True)
    else:
        print('FAILED', flush=True)


if __name__ == '__main__':
    main()
