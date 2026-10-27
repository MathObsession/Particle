#!/usr/bin/env python3
"""Model-free checks for the update feed and loopback file boundary."""
import importlib.util,sys,threading,tempfile,unittest,urllib.request,urllib.error,xml.etree.ElementTree as ET
from http.server import ThreadingHTTPServer
from pathlib import Path
from functools import partial
sys.path.insert(0,str(Path(__file__).parent))
from publish_update import validate_appcast,NS
from serve_updates import UpdateHandler
from cleanup_releases import prune_archives,older

class UpdateTests(unittest.TestCase):
    def test_feed_must_match_verified_archive_and_build(self):
        release={'version':'0.2.3','build':'2026102601'}
        data=b'<rss xmlns:sparkle="http://www.andymatuschak.org/xml-namespaces/sparkle"><channel><item><sparkle:version>2026102601</sparkle:version><sparkle:shortVersionString>0.2.3</sparkle:shortVersionString><enclosure url="https://example.com/app.zip" length="123" sparkle:edSignature="signed-data" /></item></channel></rss>'
        self.assertEqual(validate_appcast(data,release,'https://example.com/app.zip',123),'signed-data')
        for candidate,url,size in [({'version':'0.2.3','build':'../../escape'},'https://example.com/app.zip',123),({'version':'0.2.3','build':'2'},'https://example.com/app.zip',123),(release,'https://other.example/app.zip',123),(release,'https://example.com/app.zip',124)]:
            with self.assertRaises(ValueError):validate_appcast(data,candidate,url,size)
    def test_retention_deletes_only_older_particle_archives(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            old=['Particle-0.3.0-arm64.zip','ParticleOne-0.2.3-2026102601-arm64.zip','Particle-0.3.1-2026102701-arm64.zip','a'*64+'--Particle-0.3.0-arm64.zip']
            keep=['Particle-0.3.1-arm64.zip','Particle-0.3.1-2026102702-arm64.zip','Particle-0.3.2-arm64.zip','other.zip','churo-chat.particle']
            for name in old+keep:(root/name).write_bytes(b'fixture')
            link=root/('b'*64+'--Particle-0.3.0-arm64.zip');link.symlink_to(root/'other.zip')
            removed=prune_archives([root],'0.3.1','2026102702')
            self.assertEqual(set(Path(p).name for p in removed),set(old))
            self.assertTrue(all((root/name).exists() for name in keep));self.assertTrue(link.is_symlink())
    def test_numeric_versions_and_same_version_builds(self):
        self.assertTrue(older('0.3.2',None,'0.3.10','100'))
        self.assertFalse(older('0.3.10',None,'0.3.2','100'))
        self.assertTrue(older('0.3.1','99','0.3.1.0','100'))
        self.assertFalse(older('0.3.1',None,'0.3.1','100'))
    def test_host_exposes_only_feed_and_non_symlink_archives(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);(root/'appcast.xml').write_text('signed-feed');(root/'ParticleOne-1.zip').write_bytes(b'archive')
            (root/'private-key.txt').write_text('not public');(root/'ParticleOne-link.zip').symlink_to(root/'private-key.txt')
            server=ThreadingHTTPServer(('127.0.0.1',0),partial(UpdateHandler,directory=name));worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
            base='http://127.0.0.1:'+str(server.server_port)
            try:
                self.assertEqual(urllib.request.urlopen(base+'/appcast.xml').read(),b'signed-feed')
                self.assertEqual(urllib.request.urlopen(base+'/ParticleOne-1.zip').read(),b'archive')
                for path in ['/','/private-key.txt','/ParticleOne-link.zip','/%2e%2e/private-key.txt','/sub/appcast.xml']:
                    with self.assertRaises(urllib.error.HTTPError) as error:urllib.request.urlopen(base+path)
                    self.assertEqual(error.exception.code,404)
            finally:server.shutdown();server.server_close();worker.join()
if __name__=='__main__':unittest.main()
