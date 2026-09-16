"""Persistent caption slots with exact embedding deduplication and bounded reads."""

import hashlib
from contextlib import closing
import json
import os
from pathlib import Path
import random
import sqlite3
from dataclasses import asdict, replace

import torch
from torch.nn.utils.rnn import pad_sequence

from .caption import build_caption


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=lambda x: sorted(x)).encode()
    ).hexdigest()


def _encoder_fingerprint_payload(cfg):
    root = Path(cfg.train.model_path)
    files = []
    overrides = any(getattr(cfg.train, k, None) for k in ("text_encoder_path", "tokenizer_path"))
    if overrides:
        from ..modeling.loader import ASSETS, text_sources

        sources = list(text_sources(root, getattr(cfg.train, "text_encoder_path", None),
                                    getattr(cfg.train, "tokenizer_path", None)))
        if sources[0].is_file():
            sources.append(ASSETS / "qwen3vl4b/config.json")
    else:
        # Preserve existing cache keys for the original directory loading method.
        sources = [root / folder for folder in ("text_encoder", "tokenizer")]
    for source in sources:
        for p in ([source] if source.is_file() else sorted(source.rglob("*"))):
            if p.is_file():
                st = p.stat()
                files.append((str(p.resolve()), st.st_size, st.st_mtime_ns))
    from ..modeling.batched import (
        PROMPT_TEMPLATE_ENCODE,
        PROMPT_TEMPLATE_ENCODE_START_IDX,
    )
    from importlib.metadata import version

    versions = {name: version(name) for name in ("torch", "transformers", "sdnq")}
    from ..training.quant import text_encoder_quant_config

    quant = text_encoder_quant_config(cfg.quant)
    quant_payload = asdict(quant) if quant is not None else None
    if quant_payload is not None:
        # Selector metadata is not encoder behavior; preserve existing cache keys.
        quant_payload.pop("text_encoder_weights_dtype", None)
    return dict(
        version=1,
        versions=versions,
        files=files,
        dtype=cfg.train.dtype,
        tokens=cfg.train.max_text_tokens,
        template=(PROMPT_TEMPLATE_ENCODE, PROMPT_TEMPLATE_ENCODE_START_IDX),
        quant=quant_payload,
    )


def encoder_fingerprint(cfg, cache_path=None):
    """Key actual encoder behavior; reuse compatible v1 namespaces without copying blobs."""
    payload = _encoder_fingerprint_payload(cfg)
    canonical = digest(payload)
    if cache_path is None or not Path(cache_path).is_file() or payload["quant"] is None:
        return canonical
    # Older keys included transformer-only options which encoder loading overrides.
    # Enumerate only those overrides: precision, weights, tokens, model identity and
    # all other effective encoder options must still match exactly.
    from itertools import product
    from ..training.quant import SKIP_POLICIES

    compatible = {canonical}
    for mode, qmm, skip, extra in product(
        ("frozen", "training"),
        (False, True, "auto"),
        SKIP_POLICIES,
        ([], ["lm_head"], cfg.quant.extra_skip),
    ):
        old_quant = asdict(
            replace(
                cfg.quant,
                mode=mode,
                use_quantized_matmul=qmm,
                skip_policy=skip,
                extra_skip=extra,
            )
        )
        old_quant.pop("text_encoder_weights_dtype", None)
        # An explicit encoder override must not reuse the transformer's dtype.
        old_quant["weights_dtype"] = payload["quant"]["weights_dtype"]
        compatible.add(digest({**payload, "quant": old_quant}))
    with closing(
        sqlite3.connect(Path(cache_path).resolve().as_uri() + "?mode=ro", uri=True)
    ) as db:
        if not db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='embeddings'"
        ).fetchone():
            return canonical
        namespaces = db.execute(
            "SELECT encoder,COUNT(*) FROM embeddings GROUP BY encoder"
        ).fetchall()
    matches = [
        (count, key == canonical, key) for key, count in namespaces if key in compatible
    ]
    return max(matches)[2] if matches else canonical


class CaptionVariationCache:
    def __init__(self, path, encoder_key):
        self.path = str(Path(path).resolve())
        self.encoder_key = encoder_key
        self._db = None
        self._pid = None

    def close(self):
        if self._db is not None:
            self._db.close()
        self._db = None

    def __getstate__(self):
        return {**self.__dict__, "_db": None, "_pid": None}

    def reader(self):
        if self._db is None or self._pid != os.getpid():
            self.close()
            self._db = sqlite3.connect(Path(self.path).as_uri() + "?mode=ro", uri=True)
            self._db.execute("PRAGMA cache_size=-8192")
            self._pid = os.getpid()
        return self._db

    def writer(self):
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path, timeout=120)
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA cache_size=-8192")
        db.executescript("""
          CREATE TABLE IF NOT EXISTS captions(id TEXT PRIMARY KEY, text TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS slots(pool TEXT, slot INTEGER, caption TEXT NOT NULL,
            PRIMARY KEY(pool,slot));
          CREATE TABLE IF NOT EXISTS embeddings(encoder TEXT, caption TEXT, rows INTEGER,
            cols INTEGER, dtype TEXT, data BLOB, PRIMARY KEY(encoder,caption));
        """)
        return db

    def prepare(self, entries, caption_cfg, seed, count, write=True):
        """Keep N slots, including duplicates. Increasing N preserves the old prefix."""
        self.count = count
        self.seed = seed
        self.dropout = caption_cfg.caption_dropout_percent
        aug = replace(caption_cfg, caption_dropout_percent=0)
        self.indices = []
        occurrence, totals = {}, {}
        unique = {}
        for entry in entries:
            # Repeated entries and resolution variants sharing captions/source stem use one pool.
            from .cache import parse_cache_filename

            parsed = parse_cache_filename(entry.path)
            stem = parsed[0] if parsed else entry.path.stem
            identity = (str(entry.path.parent.resolve()), stem, entry.tags, entry.nl)
            pool = digest((identity, asdict(aug), seed, "caption-slots-v1"))
            ordinal = occurrence.get(pool, 0)
            occurrence[pool] = ordinal + 1
            self.indices.append((pool, ordinal))
            totals[pool] = ordinal + 1
            unique[pool] = entry
        self.totals = totals
        if not write:
            return
        with self.writer() as db:
            db.execute("INSERT OR IGNORE INTO captions VALUES (?,?)", (digest(""), ""))
            from tqdm import tqdm

            for pool, entry in tqdm(unique.items(), desc="Planning caption slots"):
                existing = db.execute(
                    "SELECT COALESCE(MAX(slot)+1,0) FROM slots WHERE pool=?", (pool,)
                ).fetchone()[0]
                for slot in range(existing, count):
                    text = build_caption(
                        entry.tags, entry.nl, aug, random.Random(digest((pool, slot)))
                    )
                    key = digest(text)
                    db.execute(
                        "INSERT OR IGNORE INTO captions VALUES (?,?)", (key, text)
                    )
                    db.execute("INSERT INTO slots VALUES (?,?,?)", (pool, slot, key))
                db.commit()
            db.execute("CREATE TEMP TABLE active(pool TEXT PRIMARY KEY)")
            db.executemany("INSERT INTO active VALUES (?)", ((p,) for p in unique))
            # Disk-backed pending list: do not hold millions of captions/embeddings in RAM.
            db.execute(
                "CREATE TABLE IF NOT EXISTS pending(encoder TEXT, caption TEXT, PRIMARY KEY(encoder,caption))"
            )
            db.execute("DELETE FROM pending WHERE encoder=?", (self.encoder_key,))
            db.execute(
                """INSERT OR IGNORE INTO pending SELECT ?, s.caption FROM slots s
                JOIN active a ON a.pool=s.pool LEFT JOIN embeddings e ON e.encoder=? AND e.caption=s.caption
                WHERE s.slot < ? AND e.caption IS NULL""",
                (self.encoder_key, self.encoder_key, count),
            )
            db.execute(
                """INSERT OR IGNORE INTO pending SELECT ?, ? WHERE NOT EXISTS
                (SELECT 1 FROM embeddings WHERE encoder=? AND caption=?)""",
                (self.encoder_key, digest(""), self.encoder_key, digest("")),
            )
            slots, unique_captions = db.execute(
                """SELECT COUNT(*),COUNT(DISTINCT s.caption) FROM slots s
                JOIN active a ON a.pool=s.pool WHERE s.slot<?""",
                (count,),
            ).fetchone()
            pending_count = db.execute(
                "SELECT COUNT(*) FROM pending WHERE encoder=?", (self.encoder_key,)
            ).fetchone()[0]
            self.statistics = dict(
                images=len(unique),
                slots=slots,
                unique_captions=unique_captions,
                pending_embeddings=pending_count,
            )
        db.close()

    def pending(self, rank=0, world_size=1):
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("Invalid caption-cache rank/world_size")
        # Keyset pagination avoids both an unbounded list and a long read transaction during writes.
        last = ""
        while True:
            rows = (
                self.reader()
                .execute(
                    """SELECT c.id,c.text FROM pending p JOIN captions c ON c.id=p.caption
                WHERE p.encoder=? AND p.caption>? ORDER BY p.caption LIMIT 64""",
                    (self.encoder_key, last),
                )
                .fetchall()
            )
            if not rows:
                break
            for key, text in rows:
                # Stable ownership even while other ranks delete completed rows.
                # Offset/modulo on row numbers would skip work as the table shrinks.
                if int(key[:16], 16) % world_size == rank:
                    yield key, text
            last = rows[-1][0]

    def add(self, key, hidden, mask, db=None):
        value = hidden[0, mask[0].bool()].detach().cpu().contiguous()
        own = db is None
        db = self.writer() if own else db
        db.execute(
            "INSERT OR REPLACE INTO embeddings VALUES (?,?,?,?,?,?)",
            (
                self.encoder_key,
                key,
                *value.shape,
                str(value.dtype).split(".")[-1],
                value.view(torch.uint8).numpy().tobytes(),
            ),
        )
        db.execute(
            "DELETE FROM pending WHERE encoder=? AND caption=?", (self.encoder_key, key)
        )
        if own:
            db.commit()
            db.close()

    def caption(self, index, epoch):
        pool, ordinal = self.indices[index]
        visit = epoch * self.totals[pool] + ordinal
        cycle, offset = divmod(visit, self.count)
        # Affine permutation is O(1) space/time and visits each slot once per cycle.
        import math

        rng = random.Random(digest((self.seed, pool, cycle)))
        a = rng.randrange(1, self.count + 1)
        while math.gcd(a, self.count) != 1:
            a = a % self.count + 1
        slot = (a * offset + rng.randrange(self.count)) % self.count
        if (
            random.Random(digest((pool, epoch, ordinal, "dropout"))).random()
            < self.dropout
        ):
            return ""
        row = (
            self.reader()
            .execute(
                """SELECT c.text FROM slots s JOIN captions c ON c.id=s.caption
            WHERE s.pool=? AND s.slot=?""",
                (pool, slot),
            )
            .fetchone()
        )
        if row is None:
            raise RuntimeError("Caption variation cache is incomplete")
        return row[0]

    def get(self, captions, device, dtype):
        values = []
        for text in captions:
            row = (
                self.reader()
                .execute(
                    "SELECT rows,cols,dtype,data FROM embeddings WHERE encoder=? AND caption=?",
                    (self.encoder_key, digest(text)),
                )
                .fetchone()
            )
            if row is None:
                raise RuntimeError(
                    "Missing cached caption embedding; rebuild the variation cache"
                )
            rows, cols, stored_dtype, blob = row
            values.append(
                torch.frombuffer(
                    bytearray(blob), dtype=getattr(torch, stored_dtype)
                ).reshape(rows, cols)
            )
        hidden = pad_sequence(values, batch_first=True).to(device=device, dtype=dtype)
        lengths = torch.tensor([v.shape[0] for v in values], device=device)
        mask = torch.arange(hidden.shape[1], device=device)[None] < lengths[:, None]
        return hidden, mask
