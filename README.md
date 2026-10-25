# board-train – Trainingsdienst für die Brett-Erkennung auf der Spark

Der Opus-Stapel-Ablauf (`ebook-board-training/fs/train_batch.py`) schickt die Trainingsdaten eines Stapels hierher,
die GPU trainiert und liest alle Bretter neu, die Antwort geht zurück. Ist der Dienst nicht erreichbar, trainiert der
Server wie bisher auf der CPU (nichts geht verloren).

## Einrichten auf der Spark (einmalig)
1. Repo auf die Spark klonen und `.env` anlegen: `TRAIN_TOKEN=<Token>` (das Token steht auf dem Heimserver in
   `~/.config/spark/train.env` als `SPARK_TRAIN_TOKEN`, oder liegt im Paket `spark-train.tar.gz`). `.env` ist nicht im Repo.
2. `docker compose up -d --build`  (zieht `nvcr.io/nvidia/pytorch:25.09-py3`, ca. 10 GB beim ersten Mal)
3. Prüfen: `curl http://localhost:8765/health` → `"device": "cuda"`
4. Port 8765 muss vom Heimserver aus erreichbar sein. Adresse steht in `~/.config/spark/train.env`
   (`SPARK_TRAIN_URL`, derzeit `http://spark.cp-austria.at:8765`) – bei anderer IP/Port dort anpassen.

## Schnittstelle
- `GET /health` → Gerät, Auslastung
- `POST /train` (Bearer-Token) → sofort eine Auftrags-Id; `GET /job/<id>` liefert 202 solange es läuft, dann das Ergebnis
  (asynchron, weil ein Reverse-Proxy davor nach ~60 s abbricht). Datenformat siehe Kopf von `server.py`.

Netz (`net.py`) und Ausgangsgewichte (`final32.pt`) sind Kopien aus `ebook-board-training/hr/` (Stand 25.10.).
Ändert sich dort das Netz, Ordner neu bauen.
