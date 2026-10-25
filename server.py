"""Trainingsdienst für die Brett-Erkennung (CNN aus ebook-board-training) auf der Spark-GPU.

POST /train   (Authorization: Bearer <TRAIN_TOKEN>, Körper = np.savez_compressed mit)
    X  uint8 [N,32,32]  Trainingsfelder        Y  int64 [N]  Klassen 0..12
    T  uint8 [M,32,32]  Testfelder (vorher/nachher bewerten)
    P  uint8 [K,32,32]  Felder, die nach dem Training gelesen werden
    epochs (optional, Standard 2)
  Antwort = np.savez_compressed mit Tb, Ta (Klasse je Testfeld vor/nach dem Training), pred (Klasse je P-Feld),
  conf (höchste Wahrscheinlichkeit je P-Feld), model (state_dict als Bytes), secs, device.
GET /health → {"device": …}

Immer nur ein Auftrag gleichzeitig; Ausgangsgewichte sind final32.pt (wie train_batch.py).
"""
import os, io, json, time, threading, numpy as np, torch, torch.nn.functional as F
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from net import Net, augment

TOKEN = os.environ['TRAIN_TOKEN']
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'final32.pt')
LOCK = threading.Lock()


def run(job):
    X, Y, T, P = (torch.from_numpy(job[k]) for k in ('X', 'Y', 'T', 'P'))
    epochs = int(job['epochs']) if 'epochs' in job else 2
    net = Net().to(DEV); net.load_state_dict(torch.load(BASE, map_location=DEV))

    def predict(C):
        net.eval(); cls, conf = [], []
        with torch.no_grad():
            for i in range(0, len(C), 8192):
                p = F.softmax(net(C[i:i + 8192].to(DEV).float().unsqueeze(1) / 255), 1)
                m = p.max(1); cls.append(m.indices.to(torch.uint8).cpu()); conf.append(m.values.half().cpu())
        return (torch.cat(cls).numpy(), torch.cat(conf).numpy()) if cls else (np.zeros(0, np.uint8), np.zeros(0, np.float16))

    Tb, _ = predict(T)
    opt = torch.optim.AdamW(net.parameters(), 5e-4, weight_decay=1e-4); t0 = time.time()
    Xg, Yg = X.to(DEV), Y.to(DEV)
    for _ in range(epochs):
        net.train(); perm = torch.randperm(len(Yg), device=DEV)
        for i in range(0, len(Yg), 256):   # gleiche Losgröße wie auf der CPU, damit die Ergebnisse vergleichbar bleiben
            j = perm[i:i + 256]; x = augment(Xg[j].float().unsqueeze(1) / 255)
            loss = F.cross_entropy(net(x), Yg[j], label_smoothing=0.05); opt.zero_grad(); loss.backward(); opt.step()
    secs = time.time() - t0
    Ta, _ = predict(T); pred, conf = predict(P)
    buf = io.BytesIO(); torch.save({k: v.cpu() for k, v in net.state_dict().items()}, buf)
    out = io.BytesIO()
    np.savez_compressed(out, Tb=Tb, Ta=Ta, pred=pred, conf=conf, model=np.frombuffer(buf.getvalue(), np.uint8),
                        secs=np.float32(secs), device=np.array(DEV))
    return out.getvalue()


class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype='application/json'):
        self.send_response(code); self.send_header('Content-Type', ctype); self.send_header('Content-Length', str(len(body)))
        self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        if self.path == '/health':
            name = torch.cuda.get_device_name(0) if DEV == 'cuda' else 'cpu'
            return self._send(200, json.dumps({'device': DEV, 'name': name, 'busy': LOCK.locked()}).encode())
        self._send(404, b'{}')

    def do_POST(self):
        if self.headers.get('Authorization') != f'Bearer {TOKEN}': return self._send(401, b'{"error":"token"}')
        if self.path != '/train': return self._send(404, b'{}')
        n = int(self.headers.get('Content-Length', 0))
        data = self.rfile.read(n)
        if not LOCK.acquire(blocking=False): return self._send(409, b'{"error":"busy"}')
        try:
            body = run(dict(np.load(io.BytesIO(data))))
            self._send(200, body, 'application/octet-stream')
        except Exception as e:
            self._send(500, json.dumps({'error': repr(e)}).encode())
        finally:
            LOCK.release()


if __name__ == '__main__':
    ThreadingHTTPServer(('0.0.0.0', int(os.environ.get('PORT', '8765'))), H).serve_forever()
