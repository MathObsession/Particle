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
