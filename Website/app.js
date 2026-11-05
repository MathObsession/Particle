const $=(s,r=document)=>r.querySelector(s),$$=(s,r=document)=>[...r.querySelectorAll(s)];
const announce=message=>{$('#announcement').textContent=message};
const themeButtons=$$('.theme-toggle');
function updateThemeLabel(){const dark=document.documentElement.dataset.theme==='dark';themeButtons.forEach(button=>button.setAttribute('aria-label',`Switch to ${dark?'light':'dark'} appearance`));$('meta[name="theme-color"]')?.setAttribute('content',dark?'#20241f':'#f7f8f2')}
updateThemeLabel();
themeButtons.forEach(button=>button.addEventListener('click',()=>{const dark=document.documentElement.dataset.theme!=='dark';document.documentElement.dataset.theme=dark?'dark':'light';try{localStorage.setItem('particle-theme',dark?'dark':'light')}catch{}updateThemeLabel()}));
const menuButton=$('.mobile-menu'),nav=$('#primary-nav');
function closeMenu(){menuButton.setAttribute('aria-expanded','false');nav.removeAttribute('data-open')}
menuButton?.addEventListener('click',()=>{const open=menuButton.getAttribute('aria-expanded')!=='true';menuButton.setAttribute('aria-expanded',String(open));nav.toggleAttribute('data-open',open)});
$$('a',nav).forEach(link=>link.addEventListener('click',closeMenu));
document.addEventListener('keydown',event=>{if(event.key==='Escape'){closeMenu();if(document.activeElement?.closest('#primary-nav'))menuButton.focus()}});
document.addEventListener('click',event=>{if(!event.target.closest('.site-header'))closeMenu()});
$$('[data-demo]').forEach(button=>button.addEventListener('click',()=>{const mode=button.dataset.demo;$$('[data-demo]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));$('#demo-chat').hidden=mode!=='chat';$('#demo-code').hidden=mode!=='code';$('#composer-text').textContent=mode==='chat'?'A question, a thought, a place to begin…':'Describe what you want to build…'}));
let filter='all';
function filterModels(){const query=$('#model-search').value.trim().toLowerCase();let count=0;$$('[data-model]').forEach(card=>{const visible=(filter==='all'||card.dataset.group===filter)&&card.dataset.search.includes(query);card.hidden=!visible;if(visible)count++});$('#model-empty').hidden=count!==0;$('#model-count').textContent=`${count} ${count===1?'model':'models'} in this view`}
$$('[data-filter]').forEach(button=>button.addEventListener('click',()=>{filter=button.dataset.filter;$$('[data-filter]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));filterModels()}));
$('#model-search')?.addEventListener('input',filterModels);
$$('[data-copy]').forEach(button=>button.addEventListener('click',async()=>{const code=document.getElementById(button.dataset.copy),text=code.textContent.trim();let copied=false;try{await navigator.clipboard.writeText(text);copied=true}catch{const range=document.createRange();range.selectNodeContents(code);const selection=window.getSelection();selection.removeAllRanges();selection.addRange(range);try{copied=document.execCommand('copy')}catch{}}
const label=$('span',button);label.textContent=copied?'Copied':'Select & copy';announce(copied?'Installation command copied.':'The command is selected. Copy it using your keyboard.');setTimeout(()=>label.textContent='Copy',2200)}));
