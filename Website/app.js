const announcement = document.getElementById('announcement');
for (const button of document.querySelectorAll('[data-copy]')) {
  button.addEventListener('click', async () => {
    const code = document.getElementById(button.dataset.copy);
    const text = code.textContent.trim();
    let copied = false;
    try {
      await navigator.clipboard.writeText(text);
      copied = true;
    } catch {
      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(code);
      selection.removeAllRanges();
      selection.addRange(range);
      try { copied = document.execCommand('copy'); } catch {}
    }
    button.textContent = copied ? 'Copied' : 'Select & copy';
    announcement.textContent = copied
      ? 'Installation command copied.'
      : 'The command is selected. Copy it using your keyboard.';
    setTimeout(() => { button.textContent = 'Copy'; }, 2200);
  });
}

const modelTabs = [...document.querySelectorAll('[data-model-tab]')];
const modelPanels = [...document.querySelectorAll('[data-model-panel]')];
if (modelTabs.length && modelTabs.length === modelPanels.length) {
  const tabList = document.querySelector('.model-tabs');
  tabList.setAttribute('role', 'tablist');
  function selectModel(id, focus = false) {
    for (const tab of modelTabs) {
      const selected = tab.dataset.modelTab === id;
      tab.setAttribute('aria-selected', String(selected));
      tab.tabIndex = selected ? 0 : -1;
      if (selected && focus) tab.focus();
    }
    for (const panel of modelPanels) panel.hidden = panel.dataset.modelPanel !== id;
  }
  modelTabs.forEach((tab, index) => {
    tab.setAttribute('role', 'tab');
    tab.setAttribute('aria-controls', 'model-' + tab.dataset.modelTab);
    tab.addEventListener('click', () => selectModel(tab.dataset.modelTab));
    tab.addEventListener('keydown', event => {
      if (event.altKey || event.ctrlKey || event.metaKey) return;
      let next;
      if (event.key === 'ArrowRight') next = (index + 1) % modelTabs.length;
      else if (event.key === 'ArrowLeft') next = (index + modelTabs.length - 1) % modelTabs.length;
      else if (event.key === 'Home') next = 0;
      else if (event.key === 'End') next = modelTabs.length - 1;
      else return;
      event.preventDefault();
      selectModel(modelTabs[next].dataset.modelTab, true);
    });
  });
  for (const panel of modelPanels) {
    panel.setAttribute('role', 'tabpanel');
    panel.setAttribute('aria-labelledby', 'tab-' + panel.dataset.modelPanel);
    panel.tabIndex = 0;
  }
  selectModel(modelTabs[0].dataset.modelTab);
  tabList.hidden = false;
}
