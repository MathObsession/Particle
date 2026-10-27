#!/usr/bin/env python3
"""Development-only update host: signed feeds/archives on loopback, never app data."""
import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote,urlsplit

class UpdateHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        name=unquote(urlsplit(self.path).path).lstrip('/')
        if '/' in name or '\\' in name or name not in self.allowed_files():
            self.send_error(404);return
        super().do_GET()
    def do_HEAD(self):
        name=unquote(urlsplit(self.path).path).lstrip('/')
        if '/' in name or '\\' in name or name not in self.allowed_files():
            self.send_error(404);return
        super().do_HEAD()
    def allowed_files(self):
        root=Path(self.directory)
        return {p.name for p in root.iterdir() if p.is_file() and not p.is_symlink() and (p.name=='appcast.xml' or (p.name.startswith(('Particle-','ParticleOne-')) and p.suffix=='.zip'))}
    def list_directory(self,path):
        self.send_error(404);return None
    def end_headers(self):
        self.send_header('Cache-Control','no-store');super().end_headers()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);p.add_argument('--port',type=int,default=49271);a=p.parse_args()
    root=a.directory.resolve(strict=True)
    ThreadingHTTPServer(('127.0.0.1',a.port),lambda *args,**kwargs:UpdateHandler(*args,directory=str(root),**kwargs)).serve_forever()
