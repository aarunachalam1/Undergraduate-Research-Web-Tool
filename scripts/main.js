// Run after DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    const yearEl = document.getElementById('year');
    if (yearEl) {
        yearEl.textContent = new Date().getFullYear();
    }

    // Smooth scroll for internal links
    document.querySelectorAll('a[href^="#"]').forEach(a => {
        a.addEventListener('click', e => {
            const id = a.getAttribute('href').slice(1);
            const el = document.getElementById(id);
            if (el) { e.preventDefault(); el.scrollIntoView({ behavior: 'smooth', block: 'start' }); }
        });
    });

    // --- Smoke tests (console) ---
    console.group('Smoke tests');
    console.assert(!!yearEl, 'Footer year element #year exists');
    ['download', 'tools', 'publications'].forEach(id => {
        console.assert(!!document.getElementById(id), `Section #${id} exists`);
    });
    const navLinks = Array.from(document.querySelectorAll('.nav-links a'));
    console.assert(navLinks.length >= 3, 'Primary nav has at least 3 links');
    navLinks.forEach(a => console.assert(a.getAttribute('href').startsWith('#'), 'Nav link uses in‑page anchor'));
    console.assert(document.querySelectorAll('details.accordion').length >= 1, 'At least one accordion exists');
    console.groupEnd();
});