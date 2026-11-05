import {readFile, writeFile, mkdir, cp, rm} from 'node:fs/promises';
const root = new URL('./', import.meta.url);
const out = new URL('./dist/', root);
const escape = value => String(value).replace(/[&<>"']/g, character => ({
  '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
}[character]));
const models = JSON.parse(await readFile(new URL('models.json', root), 'utf8'));
if (models.length !== 14 || models.filter(model => model.group === 'default').length !== 1) {
  throw Error('Unexpected model collection');
}
const rows = models.filter(model => model.group !== 'default').map(model => `
<a class="model-row" href="${escape(model.url)}" target="_blank" rel="noopener noreferrer">
  <span class="model-name">${escape(model.name)}</span>
  <span class="model-source">${escape(model.quantization)}${model.group === 'fast' ? ' · Fast' : ''}</span>
  <span aria-hidden="true">↗</span>
</a>`).join('');
let html = (await readFile(new URL('index.html', root), 'utf8')).replace('<!-- MODEL_ROWS -->', rows);
const host = process.env.VERCEL_PROJECT_PRODUCTION_URL || process.env.VERCEL_URL;
if (host && /^[a-zA-Z0-9.-]+$/.test(host)) {
  const origin = 'https://' + host;
  html = html.replace('content="/assets/app-light.webp"', `content="${origin}/assets/app-light.webp"`)
    .replace('</head>', `<link rel="canonical" href="${origin}/"><meta property="og:url" content="${origin}/"></head>`);
}
await rm(out, {recursive: true, force: true});
await mkdir(out, {recursive: true});
for (const file of ['app.js', 'style.css', 'models.json', 'favicon.ico', 'robots.txt', '404.html']) {
  await cp(new URL(file, root), new URL(file, out));
}
for (const folder of ['assets', 'fonts']) {
  await cp(new URL(folder, root), new URL(folder + '/', out), {
    recursive: true,
    filter: source => !source.endsWith('.png') || source.endsWith('logo.png')
  });
}
await writeFile(new URL('index.html', out), html);
console.log('Built Particle website: system theme, fourteen models, self-hosted assets.');
