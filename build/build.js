// Assembles the page from its parts.
//   ../index.html     the public site (share tags, general closing line)
//   ../bereshit.html  the personal copy (not committed)
//   argv[2]           optional: an artifact variant without doctype/html/head/body
const fs = require('fs'), path = require('path');
const here = __dirname, rd = f => fs.readFileSync(path.join(here, f), 'utf8');
const fix = s => s.split('[' + '.' + ']').join('.');
const PERSONAL = 'אמרת שאתה מאמין בלתת לבינה להיות בינה. ככה זה נראה מקרוב.';
const PUBLIC = 'ככה זה נראה מקרוב, כשנותנים לבינה להיות בינה.';
const SITE = 'https://davidpiterman.github.io/bereshit/';
const DESC = 'מוח שנולד בדפדפן שלך, קורא את התורה ולומד עברית מאפס מול העיניים. זה לא סרטון. זה קורה עכשיו, אצלך.';
const og = `<meta name="description" content="${DESC}">
<meta property="og:type" content="website">
<meta property="og:title" content="בראשית">
<meta property="og:description" content="${DESC}">
<meta property="og:url" content="${SITE}">
<meta property="og:image" content="${SITE}og.jpg">
<meta name="twitter:card" content="summary_large_image">
`;
const css = fix(rd('app.css')), bodyRaw = fix(rd('body.html')), engine = fix(rd('engine.js')), app = fix(rd('app.js')), corpus = rd('torah.txt').trim();
const head = `<title>בראשית</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Frank+Ruhl+Libre:wght@400;500;700;900&family=Heebo:wght@300;400;500&display=swap">
<style>
${css}</style>`;
const tail = `<script type="text/plain" id="corpus">
${corpus}
</script>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js"></script>
<script>
${engine}
</script>
<script>
${app}
</script>`;
const bodyP = bodyRaw.replace('<!--CLOSE3-->', PERSONAL), bodyG = bodyRaw.replace('<!--CLOSE3-->', PUBLIC);
const mk = (body, extra) => `<!doctype html>
<html lang="he" dir="rtl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
${extra}<style>body{margin:0}[hidden]{display:none!important}</style>
${head}
</head>
<body>
${body}
${tail}
</body>
</html>
`;
fs.writeFileSync(path.join(here, '..', 'bereshit.html'), mk(bodyP, ''));
fs.writeFileSync(path.join(here, '..', 'index.html'), mk(bodyG, og));
const out = process.argv[2];
if (out) fs.writeFileSync(out, `${head}\n${bodyP}\n${tail}\n`);
console.log('built', out || '');
