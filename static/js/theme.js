/* ============================================================
   theme.js — Sign.Detect · Dark / Light toggle
   Incluir antes de </body> en cada HTML
   ============================================================ */
(function () {
  const root = document.documentElement;
  const KEY  = 'sd-theme';

  /* Aplica el tema ANTES de pintar — evita flash */
  if (localStorage.getItem(KEY) === 'light') root.classList.add('light');

  document.addEventListener('DOMContentLoaded', () => {
    const btn = document.getElementById('theme-toggle');
    if (!btn) return;

    function syncLabel() {
      btn.dataset.label = root.classList.contains('light') ? 'light' : 'dark';
      btn.title = root.classList.contains('light')
        ? 'Cambiar a modo oscuro'
        : 'Cambiar a modo claro';
      /* Texto del botón */
      const span = btn.querySelector('span') || document.createElement('span');
      span.textContent = root.classList.contains('light') ? 'modo claro' : 'modo oscuro';
      if (!btn.querySelector('span')) btn.appendChild(span);
    }

    syncLabel();

    btn.addEventListener('click', () => {
      root.classList.toggle('light');
      localStorage.setItem(KEY, root.classList.contains('light') ? 'light' : 'dark');
      syncLabel();
    });
  });
})();