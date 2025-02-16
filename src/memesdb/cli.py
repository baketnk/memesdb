# cli.py
import sys
try:
    import pyvips
except OSError:
    print("⚠️ libvips not found! please install it:")
    print("  macOS: brew install vips")
    print("  Ubuntu/Debian: apt-get install libvips")
    print("  Fedora/RHEL: dnf install vips")
    print("  Arch: pacman -S libvips")
    sys.exit(1)

import typer
from rich import print
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.console import Console
from rich.logging import RichHandler
import logging
from pathlib import Path
import sqlite3
import sqlite_vec
from PIL import Image
import moondream as md
from sentence_transformers import SentenceTransformer
import struct
import os
import json
from concurrent.futures import ThreadPoolExecutor
from fzf import Fzf
import platform
import subprocess
import base64
import imagehash

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)
log = logging.getLogger("memesdb")

console = Console()
app = typer.Typer()

DB_PATH_STR = os.getenv("MEMESDB_PATH", "~/.local/share/memesdb/memes.db")
DB_PATH = Path(DB_PATH_STR).expanduser()

# Detect OS for clipboard ops
SYSTEM = platform.system()

EMBED_DIM = 384

def copy_to_clipboard(text):
    """Cross-platform clipboard support"""
    if SYSTEM == "Darwin":  # macOS
        subprocess.run("pbcopy", text=True, input=text)
    elif SYSTEM == "Linux":
        subprocess.run(["xclip", "-selection", "clipboard"], text=True, input=text)
    elif SYSTEM == "Windows":
        subprocess.run(["clip"], text=True, input=text)

def preview_in_terminal(image_path):
    """Show image preview if supported"""
    if SYSTEM == "Darwin":  # iTerm2
        try:
            b64_image = base64.b64encode(open(image_path, 'rb').read()).decode('utf-8')
            print(f'\033]1337;File=inline=1:{b64_image}\a')
        except:
            log.debug("iTerm2 image preview failed")

class MemeDB:
    def __init__(self):
        self.conn = None
        self.init_db()

    def init_db(self):
        """Initialize database with vector support"""
        try:
            DB_PATH.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(DB_PATH)
            self.conn.enable_load_extension(True)
            sqlite_vec.load(self.conn)
            
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS memes (
                    id INTEGER PRIMARY KEY,
                    path TEXT UNIQUE,
                    meta JSON,
                    short_caption TEXT,
                    long_caption TEXT,
                    auto_tags TEXT,
                    user_tags TEXT,
                    hash TEXT
                )""")
            
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_path ON memes(path);
            """)

            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_hash ON memes(hash);
            """)
            
            self.conn.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_memes 
                USING vec0(embedding float[{EMBED_DIM}])""")
            
        except Exception as e:
            log.error(f"Failed to initialize database: {e}")
            raise


MODEL_PATH = Path("./downloads/moondream-2b-int8.mf.gz")

class MemeScanner:
    def __init__(self):
        try:
            console.print("[bold green]Loading AI models...")
            self.caption_model = md.vl(model=str(MODEL_PATH))
            self.embed_model = SentenceTransformer('BAAI/bge-small-en-v1.5')
        except Exception as e:
            log.error(f"Failed to load AI models: {e}")
            raise

    def process_batch(self, batch):
        """Process a batch of images"""
        results = []
        for img_path in batch:
            try:
                results.append((img_path, self.probe_image(img_path)))
            except Exception as e:
                log.error(f"Error processing {img_path}: {e}")
        return results

    def probe_image(self, img_path):
        """Extract information from a single image"""
        img = Image.open(img_path)
        meta = {
            "format": img.format,
            "size": img.size,
            "mode": img.mode,
            "path": str(img_path)
        }
        
        # Encode image once and reuse
        encoded_img = self.caption_model.encode_image(img)
        
        return {
            "meta": meta,
            "short": self.caption_model.caption(encoded_img)["caption"],
            "long": self.caption_model.query(encoded_img, "Describe this image in detail")["caption"],
            "tags": self.caption_model.query(encoded_img, "List comma-separated tags for this image")["answer"]
        }

@app.command()
def index(dir_path: Path, batch_size: int = 4):
    """Scan directory for memes with progress bar"""
    scanner = MemeScanner()
    db = MemeDB()
    
    image_paths = [p for p in dir_path.rglob("*") 
                  if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.gif'}]
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TextColumn("•"),
        TextColumn("[blue]{task.fields[filename]}"),
        TextColumn("•"),
        TextColumn("[yellow]{task.completed}/{task.total}"),
        TextColumn("•"),
        TextColumn("[green]{task.fields[current_op]}"),
        console=console
    ) as progress:
        task = progress.add_task(
            "[cyan]Indexing memes...", 
            total=len(image_paths),
            filename="",
            current_op="starting..."
        )
        
        for img_path in image_paths:
            try:
                progress.update(task, filename=img_path.name)
                # check for existing path before any heavy lifting
                existing = db.conn.execute(
                    "SELECT path FROM memes WHERE path = ?", 
                    [str(img_path)]
                ).fetchone()
                
                if existing:
                    progress.update(task, current_op="skipping (already indexed)")
                    progress.advance(task)
                    continue
                
                progress.update(task, current_op="loading image")
                img = Image.open(img_path)

                img_hash = str(imagehash.average_hash(img))

                progress.update(task, current_op="encoding image")
                encoded_img = scanner.caption_model.encode_image(img)

                progress.update(task, current_op="generating short caption")

                short_caption = scanner.caption_model.caption(encoded_img, length="short")["caption"]

                progress.update(task, current_op="generating long caption")
                long_caption  = scanner.caption_model.caption(encoded_img, length="normal")["caption"]
                progress.update(task, current_op="generating tags")

                tags_text =  scanner.caption_model.query(encoded_img, "List comma-separated tags for this image")["answer"] 
                data = {
                    "meta": {
                        "format": img.format,
                        "size": tuple(img.size),  # ensure it's a tuple
                        "mode": img.mode,
                        "path": str(img_path)  # ensure path is string
                    },
                    "short": short_caption,
                    "long": long_caption,
                    "tags": tags_text,
                }
                
                progress.update(task, current_op="computing embeddings")
                embedding = scanner.embed_model.encode(
                    f"{data['short']} {data['long']} {data['tags']}"
                )
                
                progress.update(task, current_op="saving to db")

                log.debug("""
                    INSERT INTO memes 
                        (path, meta, short_caption, long_caption, auto_tags, hash)
                    VALUES 
                        (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(path) DO UPDATE SET
                        meta = excluded.meta,
                        short_caption = excluded.short_caption,
                        long_caption = excluded.long_caption,
                        auto_tags = excluded.auto_tags,
                        hash = excluded.hash""")
                log.debug([
                    str(img_path),
                    json.dumps(data["meta"]),
                    data["short"], 
                    data["long"],
                    data["tags"],
                    img_hash
                ])
                        
                cursor = db.conn.execute("""
                    INSERT INTO memes 
                        (path, meta, short_caption, long_caption, auto_tags, hash)
                    VALUES 
                        (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(path) DO UPDATE SET
                        meta = excluded.meta,
                        short_caption = excluded.short_caption,
                        long_caption = excluded.long_caption,
                        auto_tags = excluded.auto_tags,
                        hash = excluded.hash""", [
                    str(img_path),
                    json.dumps(data["meta"]),
                    data["short"], 
                    data["long"],
                    data["tags"],
                    img_hash
                ])
                meme_id = cursor.lastrowid
                
                blob = struct.pack(f"{len(embedding)}f", *embedding)
                db.conn.execute("DELETE FROM vec_memes WHERE rowid = ?", [meme_id])
                db.conn.execute("INSERT INTO vec_memes (rowid, embedding) VALUES (?,?)", 
                    [meme_id, blob])
                
                db.conn.commit()
                progress.advance(task)
                
            except Exception as e:
                console.print()
                log.exception(e)
                log.error(f"Failed to process {img_path}: {e}")
                
    
    print("[green]✨ Indexing complete![/]")

@app.command()
def tag(query: str = ""):
    """Tag memes with custom labels"""
    db = MemeDB()
    
    # Get memes matching query if provided
    if query:
        scanner = MemeScanner()
        vec = scanner.embed_model.encode(query)
        blob = struct.pack(f"{len(vec)}f", *vec)
        results = db.conn.execute("""
            SELECT m.id, m.path, m.short_caption, m.user_tags
            FROM vec_memes v
            JOIN memes m ON v.rowid = m.id
            WHERE v.embedding MATCH ?
            ORDER BY distance
            LIMIT 20
        """, [blob]).fetchall()
    else:
        results = db.conn.execute("""
            SELECT id, path, short_caption, user_tags
            FROM memes
            LIMIT 20
        """).fetchall()
    
    choices = [f"{path}\n  {caption}\n  Tags: {tags or 'None'}" 
              for _, path, caption, tags in results]
    
    selected = Fzf().prompt(choices)
    if selected:
        meme_id = results[choices.index(selected)][0]
        new_tags = typer.prompt("Enter comma-separated tags")
        
        db.conn.execute("""
            UPDATE memes SET user_tags = ? WHERE id = ?
        """, [new_tags, meme_id])
        db.conn.commit()
        print(f"[green]Updated tags for {selected.split()[0]}[/]")

@app.command()
def search(query: str):
    """Find memes with semantic search"""
    db = MemeDB()
    scanner = MemeScanner()
    
    vec = scanner.embed_model.encode(query)
    blob = struct.pack(f"{len(vec)}f", *vec)
    
    results = db.conn.execute("""
        SELECT m.path, m.short_caption, m.long_caption, m.auto_tags, m.user_tags
        FROM vec_memes v
        JOIN memes m ON v.rowid = m.id
        WHERE v.embedding MATCH ?
        AND k = 20
        ORDER BY distance
    """, [blob]).fetchall()
    
    choices = []
    for path, short, long, auto_tags, user_tags in results:
        preview_in_terminal(path)
        choices.append(
            f"{path}\n  {short}\n  {long}\n  Auto: {auto_tags}\n  User: {user_tags or 'None'}"
        )
    
    selected = Fzf().prompt(choices)
    if selected:
        path = selected.split('\n')[0]
        print(f"[cyan]Selected:[/] {path}")
        copy_to_clipboard(path)
        print("[green]Path copied to clipboard![/]")

@app.command()
def stats():
    """show quick db stats"""
    db = MemeDB()
    
    # get basic counts
    counts = db.conn.execute("""
        SELECT COUNT(*) as total,
               SUM(LENGTH(meta) + LENGTH(short_caption) + LENGTH(long_caption) + LENGTH(COALESCE(auto_tags,'')) + LENGTH(COALESCE(user_tags,''))) as text_size,
               COUNT(DISTINCT hash) as unique_images
        FROM memes
    """).fetchone()
    
    # get db file size
    db_size = DB_PATH.stat().st_size

    console.print("\n[bold cyan]📊 memesdb stats[/]\n")
    console.print(f"📑 total entries: [yellow]{counts[0]:,}[/]")
    console.print(f"🖼️  unique images: [yellow]{counts[2]:,}[/]")
    console.print(f"💾 database size: [yellow]{db_size / 1024 / 1024:.1f}MB[/]")
    console.print(f"📝 text data size: [yellow]{counts[1] / 1024:.1f}KB[/]")
    console.print(f"🗄️  database location: [dim]{DB_PATH}[/]\n")

if __name__ == "__main__":
    app()
