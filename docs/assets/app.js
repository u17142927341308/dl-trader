async function loadCSV(url){
  const t = await fetch(url, {cache:"no-store"});
  const s = await t.text();
  const [head, ...rows] = s.trim().split(/\r?\n/);
  const cols = head.split(",");
  return rows.map(r=>{
    const vals = r.split(",");
    const o={};
    cols.forEach((c,i)=>o[c]=vals[i]);
    return o;
  });
}

async function loadJSON(url){
  const t = await fetch(url, {cache:"no-store"});
  return await t.json();
}

function fmtTs(ts){ try { return new Date(ts).toLocaleString(); } catch { return ts; } }

function card(signal){
  const side = (signal.side||"").toUpperCase();
  const badge = side==="BUY" ? "buy":"sell";
  const w = Number(signal.weight||0);
  const wstr = (w>=0?"+":"") + w.toFixed(2);
  const p = Number(signal.price||0).toFixed(2);
  return `
    <div class="card">
      <h3>${signal.symbol}</h3>
      <div class="meta">${fmtTs(signal.ts)}</div>
      <div class="meta">Kurs: ${p}</div>
      <div class="meta">Gewicht: ${wstr}</div>
      <div class="badge ${badge}">${side}</div>
    </div>`;
}

async function main(){
  const signals = await loadCSV("data/signals.csv").catch(()=>[]);
  const metrics = await loadJSON("data/metrics.json").catch(()=>({history:[], per_symbol:{}}));
  const metaEl = document.getElementById("meta");
  const sigEl = document.getElementById("signals");

  // letzte pro Symbol
  const lastBySym = {};
  for (const r of signals){
    lastBySym[r.symbol] = r;
  }
  sigEl.innerHTML = Object.values(lastBySym).map(card).join("");

  // Kopfzeile + nächster Lauf
  const lastTs = signals.length ? signals[signals.length-1].ts : null;
  metaEl.textContent = lastTs ? `Letztes Update: ${fmtTs(lastTs)}` : "Noch keine Daten";
  const nextRun = document.getElementById("nextRun");
  nextRun.textContent = "Geplante Läufe: stündlich per GitHub Actions";

  // Trainingschart (Loss über Zeit)
  const ctx = document.getElementById("trainChart");
  const labels = metrics.history.map(x=>fmtTs(x.ts));
  const losses = metrics.history.map(x=>x.loss);
  new Chart(ctx, {
    type:"line",
    data:{ labels, datasets:[{ label:"Train Loss", data:losses }] },
    options:{ responsive:true, plugins:{legend:{display:true}} }
  });

  // Symbol-Chart (gewicht / Vorhersage)
  const sel = document.getElementById("symbolSelect");
  const symbols = Object.keys(metrics.per_symbol||{});
  symbols.forEach(s=>{
    const o = document.createElement("option");
    o.value=s; o.textContent=s; sel.appendChild(o);
  });
  const symCtx = document.getElementById("symbolChart");
  let symChart=null;
  function renderSymbol(sym){
    const arr = (metrics.per_symbol[sym]||[]);
    const labels = arr.map(x=>fmtTs(x.ts));
    const w = arr.map(x=>x.weight);
    if (symChart) symChart.destroy();
    symChart = new Chart(symCtx, {
      type:"line",
      data:{ labels, datasets:[{ label:`${sym} Gewicht`, data:w }]},
      options:{ responsive:true }
    });
  }
  if (symbols.length){ renderSymbol(symbols[0]); sel.value=symbols[0]; sel.addEventListener("change",()=>renderSymbol(sel.value)); }
}
main();
