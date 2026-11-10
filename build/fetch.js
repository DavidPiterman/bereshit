const fs=require('fs');
const books=[['Genesis',50],['Exodus',40],['Leviticus',27],['Numbers',36],['Deuteronomy',34]];
(async()=>{
  const verses=[];
  for(const [b,n] of books){
    for(let s=1;s<=n;s+=10){
      const e=Math.min(n,s+9);
      const url=`https://www.sefaria.org/api/texts/${b}.${s}-${e}?context=0&commentary=0&pad=0`;
      let j;
      for(let k=0;k<4;k++){try{const r=await fetch(url);j=await r.json();if(j.he)break;}catch(err){await new Promise(r=>setTimeout(r,1500));}}
      let he=j.he; if(typeof he[0]==='string') he=[he];
      for(const ch of he) for(const v of ch) verses.push(v);
      console.error(b,s,e,he.length);
    }
  }
  const clean=verses.map(v=>{
    v=v.replace(/<[^>]*>/g,'').replace(/&[a-z]+;/g,' ').replace(/\{[^}]*\}/g,' ').replace(/\([^)]*\)/g,' ').replace(/\[[^\]]*\]/g,' ');
    v=v.replace(/־/g,' ').replace(/[^א-ת ]/g,'').replace(/ +/g,' ').trim();
    v=v.replace(/יהוה/g,"ה'");
    return v;
  }).filter(v=>v.length>0);
  fs.writeFileSync('torah.txt',clean.join('\n'),'utf8');
  console.log(clean.length,'verses',clean.join(' ').length,'chars');
  console.log(clean.slice(0,5).join('\n'));
})();
