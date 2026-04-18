import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  AreaChart, Area, LineChart, Line,
  XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, PieChart, Pie, Legend
} from 'recharts';

// ── CSS injected directly ─────────────────────────────────────────────────────
const GLOBAL_CSS = `
  @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@0,300;0,400;0,500;1,300;1,400&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,400&display=swap');

  :root {
    --sage:        #7C9E87;
    --sage-light:  #B8CFC0;
    --sage-dark:   #4A7A5A;
    --peach:       #F2A07B;
    --peach-light: #FAD4BC;
    --cream:       #FDF8F2;
    --warm-white:  #FFFCF8;
    --charcoal:    #2C2C2C;
    --muted:       #9A9A9A;
    --border:      #EAE2D8;
    --shadow-sm:   0 2px 10px rgba(44,44,44,0.06);
    --shadow-md:   0 6px 24px rgba(44,44,44,0.10);
    --radius:      22px;
    --radius-sm:   14px;
    --radius-xs:   8px;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: 'DM Sans', sans-serif;
    background: #F0EBE3;
    color: var(--charcoal);
    min-height: 100vh;
  }

  #root {
    max-width: 430px;
    margin: 0 auto;
    background: var(--cream);
    min-height: 100vh;
    position: relative;
    box-shadow: 0 0 80px rgba(0,0,0,0.12);
  }

  h1,h2,h3,h4 { font-family: 'Fraunces', serif; font-weight: 400; }
  ::selection  { background: var(--peach-light); }
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-thumb { background: var(--sage-light); border-radius: 2px; }

  @keyframes spin     { to { transform: rotate(360deg); }}
  @keyframes fadeUp   { from { opacity:0; transform:translateY(12px); } to { opacity:1; transform:translateY(0); }}
  @keyframes pulse    { 0%,100% { transform: scale(1); } 50% { transform: scale(1.04); }}
  @keyframes shimmer  { 0% { background-position: -200% 0; } 100% { background-position: 200% 0; }}

  .fade-up  { animation: fadeUp 0.4s ease forwards; }
  .pulse    { animation: pulse 2s ease-in-out infinite; }

  textarea:focus { outline: none; }
  button:active  { transform: scale(0.97); }
`;

// ── API ───────────────────────────────────────────────────────────────────────
const API = 'http://localhost:8000';
async function api(path, opts = {}) {
  const res = await fetch(`${API}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Request failed');
  }
  return res.json();
}

// ── State metadata ────────────────────────────────────────────────────────────
const SM = {
  Normal              : { color:'#7C9E87', bg:'#EDF4EF', emoji:'😊', label:'Feeling Well'      },
  Anxiety             : { color:'#E8845A', bg:'#FDF0EA', emoji:'😰', label:'Feeling Anxious'   },
  Stress              : { color:'#D4A843', bg:'#FBF5E6', emoji:'😤', label:'Under Stress'       },
  Depression          : { color:'#7B8EC8', bg:'#EEF0F8', emoji:'😔', label:'Feeling Low'        },
  Bipolar             : { color:'#C17BC0', bg:'#F5EEF5', emoji:'🔄', label:'Mixed Emotions'     },
  Suicidal            : { color:'#C05A5A', bg:'#F5EDED', emoji:'🆘', label:'Please Seek Help'   },
  'Personality Disorder':{ color:'#E09060',bg:'#FBF1E9', emoji:'🌀', label:'Emotionally Intense'},
};

// ════════════════════════════════════════════════════════════════════════════
// APP ROOT
// ════════════════════════════════════════════════════════════════════════════
export default function App() {
  const [page, setPage] = useState('home');
  const [toast, setToast] = useState(null);

  useEffect(() => {
    const el = document.createElement('style');
    el.textContent = GLOBAL_CSS;
    document.head.appendChild(el);
  }, []);

  const showToast = (msg, type = 'success') => {
    setToast({ msg, type });
    setTimeout(() => setToast(null), 3000);
  };

  return (
    <>
      <TopBar />
      <main style={{ paddingBottom: 76 }}>
        {page === 'home'    && <HomePage    showToast={showToast} />}
        {page === 'diary'   && <DiaryPage   showToast={showToast} />}
        {page === 'tracker' && <TrackerPage showToast={showToast} />}
        {page === 'profile' && <ProfilePage showToast={showToast} />}
      </main>
      <BottomNav current={page} onChange={setPage} />
      {toast && <Toast msg={toast.msg} type={toast.type} />}
    </>
  );
}

// ── Top bar ───────────────────────────────────────────────────────────────────
function TopBar() {
  const today = new Date().toLocaleDateString('en-US',
    { weekday: 'long', month: 'long', day: 'numeric' });
  return (
    <header style={{
      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      padding: '18px 22px 14px',
      background: 'var(--warm-white)',
      borderBottom: '1px solid var(--border)',
      position: 'sticky', top: 0, zIndex: 200,
    }}>
      <div>
        <div style={{ display:'flex', alignItems:'center', gap:8 }}>
          <span style={{ fontSize:20 }}>🌿</span>
          <h1 style={{ fontSize:20, letterSpacing:'-0.3px', color:'var(--sage-dark)' }}>
            Serenity
          </h1>
        </div>
        <p style={{ fontSize:11, color:'var(--muted)', marginTop:2 }}>{today}</p>
      </div>
      <div style={{
        width:36, height:36, borderRadius:'50%',
        background:'linear-gradient(135deg,var(--sage-light),var(--peach-light))',
        display:'flex', alignItems:'center', justifyContent:'center', fontSize:16,
        boxShadow:'var(--shadow-sm)',
      }}>🌸</div>
    </header>
  );
}

// ── Bottom nav ────────────────────────────────────────────────────────────────
function BottomNav({ current, onChange }) {
  const tabs = [
    { id:'home',    icon:'🏠', label:'Home'    },
    { id:'diary',   icon:'📓', label:'Diary'   },
    { id:'tracker', icon:'📈', label:'Tracker' },
    { id:'profile', icon:'🌱', label:'Profile' },
  ];
  return (
    <nav style={{
      position:'fixed', bottom:0, left:'50%', transform:'translateX(-50%)',
      width:'100%', maxWidth:430,
      background:'var(--warm-white)', borderTop:'1px solid var(--border)',
      display:'flex', justifyContent:'space-around',
      padding:'8px 0 14px', zIndex:200,
    }}>
      {tabs.map(t => (
        <button key={t.id} onClick={() => onChange(t.id)} style={{
          display:'flex', flexDirection:'column', alignItems:'center', gap:3,
          background:'none', border:'none', cursor:'pointer',
          color: current===t.id ? 'var(--sage-dark)' : 'var(--muted)',
          transition:'all 0.2s', padding:'2px 14px',
        }}>
          <span style={{
            fontSize:22,
            filter: current===t.id ? 'none' : 'grayscale(30%)',
            transform: current===t.id ? 'translateY(-2px)' : 'none',
            transition:'transform 0.2s',
          }}>{t.icon}</span>
          <span style={{ fontSize:10, fontWeight: current===t.id ? 600 : 400 }}>
            {t.label}
          </span>
          {current===t.id && (
            <div style={{
              width:4, height:4, borderRadius:'50%',
              background:'var(--sage-dark)', marginTop:1,
            }}/>
          )}
        </button>
      ))}
    </nav>
  );
}

// ── Toast ─────────────────────────────────────────────────────────────────────
function Toast({ msg, type }) {
  const bg = type === 'error' ? '#C05A5A' : type === 'info' ? '#7B8EC8' : '#4A7A5A';
  return (
    <div style={{
      position:'fixed', bottom:90, left:'50%', transform:'translateX(-50%)',
      background:bg, color:'#fff', borderRadius:30, padding:'10px 20px',
      fontSize:13, zIndex:999, boxShadow:'var(--shadow-md)',
      animation:'fadeUp 0.3s ease',
    }}>{msg}</div>
  );
}

// ════════════════════════════════════════════════════════════════════════════
// HOME PAGE
// ════════════════════════════════════════════════════════════════════════════
function HomePage({ showToast }) {
  const [data, setData]    = useState(null);
  const [loading, setLoad] = useState(true);

  useEffect(() => {
    Promise.all([
      api('/api/dashboard'),
      api('/api/content/today').catch(() => null),
    ]).then(([dash, content]) => {
      setData({ ...dash, content });
      setLoad(false);
    }).catch(() => setLoad(false));
  }, []);

  if (loading) return <Loader msg="Loading your day..." />;

  return (
    <div style={{ padding:'18px 18px 0' }}>
      <GreetingBanner streak={data?.streak || 0} hasDiary={data?.has_diary} />
      {data?.prediction
        ? <TodayStateCard p={data.prediction} />
        : <WritePromptCard />
      }
      {data?.usage && <ActivityRings usage={data.usage} />}
      {data?.content && <ContentCard content={data.content} showToast={showToast} />}
      {data?.usage && <StatsRow usage={data.usage} />}
    </div>
  );
}

function GreetingBanner({ streak, hasDiary }) {
  const h = new Date().getHours();
  const greet = h<12 ? 'Good morning' : h<17 ? 'Good afternoon' : 'Good evening';
  return (
    <div style={{
      background:'linear-gradient(135deg,#4A7A5A 0%,#7C9E87 60%,#8FB89A 100%)',
      borderRadius:'var(--radius)', padding:'22px 22px 18px',
      color:'#fff', marginBottom:14, overflow:'hidden', position:'relative',
    }}>
      <div style={{ position:'absolute',right:-20,top:-20,width:100,height:100,
        borderRadius:'50%',background:'rgba(255,255,255,0.07)' }}/>
      <div style={{ position:'absolute',right:30,bottom:-30,width:70,height:70,
        borderRadius:'50%',background:'rgba(255,255,255,0.05)' }}/>
      <p style={{ fontSize:12,opacity:0.8,letterSpacing:'0.8px',textTransform:'uppercase' }}>
        {greet}
      </p>
      <h2 style={{ fontSize:24,marginTop:4,fontWeight:400 }}>
        {hasDiary ? 'Welcome back 🌱' : 'How are you feeling?'}
      </h2>
      {streak > 0 && (
        <div style={{
          marginTop:12,display:'inline-flex',alignItems:'center',gap:6,
          background:'rgba(255,255,255,0.2)',borderRadius:20,padding:'5px 14px',
          fontSize:12,fontWeight:500,
        }}>🔥 {streak} day streak — keep it up!</div>
      )}
    </div>
  );
}

function TodayStateCard({ p }) {
  const m = SM[p.predicted_state] || SM.Normal;
  return (
    <div className="fade-up" style={{
      background:m.bg,
      border:`1.5px solid ${m.color}35`,
      borderRadius:'var(--radius)', padding:'18px 20px',
      marginBottom:14, display:'flex', alignItems:'center', gap:14,
    }}>
      <div style={{
        width:52,height:52,borderRadius:'50%',
        background:m.color+'22',
        display:'flex',alignItems:'center',justifyContent:'center',
        fontSize:24,flexShrink:0,
      }} className="pulse">{m.emoji}</div>
      <div style={{ flex:1 }}>
        <p style={{ fontSize:11,color:m.color,fontWeight:600,
          textTransform:'uppercase',letterSpacing:'0.6px' }}>Today's State</p>
        <h3 style={{ fontSize:20,color:'var(--charcoal)',marginTop:2 }}>
          {p.predicted_state}
        </h3>
        <p style={{ fontSize:12,color:'var(--muted)',marginTop:3 }}>{m.label}</p>
      </div>
      <div style={{
        background:m.color, color:'#fff',
        borderRadius:20, padding:'5px 12px',
        fontSize:13, fontWeight:600,
      }}>
        {((p.confidence||0)*100).toFixed(0)}%
      </div>
    </div>
  );
}

function WritePromptCard() {
  return (
    <div style={{
      background:'var(--warm-white)',
      border:'2px dashed var(--border)',
      borderRadius:'var(--radius)', padding:'22px',
      marginBottom:14, textAlign:'center',
    }}>
      <p style={{ fontSize:32,marginBottom:8 }}>📓</p>
      <h3 style={{ fontSize:17,color:'var(--charcoal)' }}>Write today's entry</h3>
      <p style={{ fontSize:13,color:'var(--muted)',marginTop:6,lineHeight:1.5 }}>
        Open the Diary tab, write how you're feeling,<br/>then tap Analyse to see your state.
      </p>
    </div>
  );
}

function ActivityRings({ usage }) {
  const rings = [
    { label:'Screen', v:usage.screen_time_mins||0, max:720, color:'#7B8EC8', icon:'💻' },
    { label:'Social', v:usage.social_media_mins||0, max:300, color:'#E8845A', icon:'📱' },
    { label:'Active', v:usage.active_mins||0,       max:600, color:'#7C9E87', icon:'⚡' },
  ];
  const fmt = (m) => {
    const h = Math.floor(m/60), mn = Math.round(m%60);
    return h>0 ? `${h}h ${mn}m` : `${mn}m`;
  };
  return (
    <div style={{
      background:'var(--warm-white)',borderRadius:'var(--radius)',
      padding:'16px 20px',marginBottom:14,boxShadow:'var(--shadow-sm)',
    }}>
      <p style={{ fontSize:11,color:'var(--muted)',textTransform:'uppercase',
        letterSpacing:'0.6px',marginBottom:16 }}>Today's Activity</p>
      <div style={{ display:'flex',justifyContent:'space-around' }}>
        {rings.map(r => {
          const pct = Math.min(r.v/r.max,1);
          const sz=68, sw=6, rad=(sz-sw)/2, circ=2*Math.PI*rad;
          return (
            <div key={r.label} style={{ textAlign:'center' }}>
              <div style={{ position:'relative',width:sz,height:sz,margin:'0 auto 6px' }}>
                <svg width={sz} height={sz} style={{ transform:'rotate(-90deg)' }}>
                  <circle cx={sz/2} cy={sz/2} r={rad} fill="none"
                    stroke={r.color+'20'} strokeWidth={sw}/>
                  <circle cx={sz/2} cy={sz/2} r={rad} fill="none"
                    stroke={r.color} strokeWidth={sw} strokeLinecap="round"
                    strokeDasharray={circ}
                    strokeDashoffset={circ*(1-pct)}
                    style={{ transition:'stroke-dashoffset 1.2s ease' }}/>
                </svg>
                <div style={{ position:'absolute',inset:0,display:'flex',
                  alignItems:'center',justifyContent:'center',fontSize:20 }}>
                  {r.icon}
                </div>
              </div>
              <p style={{ fontSize:12,fontWeight:600,color:'var(--charcoal)' }}>
                {fmt(r.v)}
              </p>
              <p style={{ fontSize:10,color:'var(--muted)' }}>{r.label}</p>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function ContentCard({ content, showToast }) {
  const [rated, setRated] = useState(false);
  const isExercise = content.type === 'exercise';
  const isCrisis   = content.type === 'crisis';
  const accent = isCrisis ? '#C05A5A' : isExercise ? '#7C9E87' : '#D4A843';

  return (
    <div style={{
      borderRadius:'var(--radius)', padding:'18px 20px', marginBottom:14,
      background: isCrisis ? '#FEF5F5' : isExercise ? '#F2F8F4' : '#FBF8EE',
      border:`1.5px solid ${accent}28`,
    }}>
      <div style={{ display:'flex',alignItems:'center',gap:8,marginBottom:10 }}>
        <span style={{ fontSize:18 }}>
          {isCrisis ? '🆘' : isExercise ? '🌿' : '✨'}
        </span>
        <p style={{ fontSize:11,fontWeight:600,color:accent,
          textTransform:'uppercase',letterSpacing:'0.6px' }}>
          {isCrisis ? 'Important' : isExercise ? 'Try This Today' : "Today's Thought"}
        </p>
      </div>
      <p style={{
        fontSize:15,lineHeight:1.7,color:'var(--charcoal)',
        fontFamily: isExercise ? 'DM Sans,sans-serif' : 'Fraunces,serif',
        fontStyle:  isExercise ? 'normal' : 'italic',
      }}>{content.text}</p>

      {!rated && content.id && (
        <div style={{ display:'flex',alignItems:'center',gap:8,marginTop:12 }}>
          <p style={{ fontSize:12,color:'var(--muted)' }}>Was this helpful?</p>
          {['👍','👎'].map((e,i) => (
            <button key={e} onClick={async () => {
              await api('/api/content/rate',{
                method:'POST',
                body:JSON.stringify({content_id:content.id,was_helpful:i===0})
              }).catch(()=>{});
              setRated(true);
              showToast(i===0 ? 'Thanks! We\'ll show more like this.' : 'Got it, we\'ll adjust!');
            }} style={{
              background:'rgba(255,255,255,0.8)',
              border:`1px solid ${accent}30`,
              borderRadius:20,padding:'4px 14px',cursor:'pointer',fontSize:15,
            }}>{e}</button>
          ))}
        </div>
      )}
      {rated && <p style={{ fontSize:12,color:accent,marginTop:10 }}>
        Thanks for your feedback! 🌱
      </p>}
    </div>
  );
}

function StatsRow({ usage }) {
  const items = [
    { icon:'⌨️', label:'Keystrokes', val:(usage.keystrokes||0).toLocaleString() },
    { icon:'☕', label:'Breaks',      val:`${usage.break_count||0} taken`        },
    { icon:'🌙', label:'Late Night',  val:`${Math.round((usage.late_night_mins||usage.screen_time_mins||0)*0)||0}m` },
  ];
  return (
    <div style={{ display:'grid',gridTemplateColumns:'1fr 1fr 1fr',gap:10,marginBottom:18 }}>
      {items.map(s => (
        <div key={s.label} style={{
          background:'var(--warm-white)',borderRadius:'var(--radius-sm)',
          padding:'14px 10px',textAlign:'center',boxShadow:'var(--shadow-sm)',
        }}>
          <p style={{ fontSize:22,marginBottom:4 }}>{s.icon}</p>
          <p style={{ fontSize:13,fontWeight:600 }}>{s.val}</p>
          <p style={{ fontSize:10,color:'var(--muted)',marginTop:2 }}>{s.label}</p>
        </div>
      ))}
    </div>
  );
}

// ════════════════════════════════════════════════════════════════════════════
// DIARY PAGE
// ════════════════════════════════════════════════════════════════════════════
function DiaryPage({ showToast }) {
  const today     = new Date().toISOString().split('T')[0];
  const [text,    setText]   = useState('');
  const [wc,      setWc]     = useState(0);
  const [saving,  setSaving] = useState(false);
  const [saved,   setSaved]  = useState(false);
  const [pred,    setPred]   = useState(null);
  const [busy,    setBusy]   = useState(false);
  const [err,     setErr]    = useState(null);
  const [prompts, setPrompts]= useState(false);

  useEffect(() => {
    api(`/api/diary/${today}`).then(d => {
      if (d.exists) { setText(d.text); setWc(d.word_count); setSaved(true); }
    }).catch(()=>{});
    api(`/api/prediction/${today}`).then(p => {
      if (p.exists) setPred(p);
    }).catch(()=>{});
  }, [today]);

  const handleChange = e => {
    setText(e.target.value);
    setWc(e.target.value.trim().split(/\s+/).filter(Boolean).length);
    setSaved(false); setErr(null);
  };

  const save = async () => {
    if (text.trim().length < 10) { setErr('Please write at least 10 characters.'); return; }
    setSaving(true);
    try {
      await api('/api/diary',{ method:'POST', body:JSON.stringify({text,date:today}) });
      setSaved(true); showToast('Diary saved ✓');
    } catch(e) { setErr(e.message); showToast(e.message,'error'); }
    finally { setSaving(false); }
  };

  const analyse = async () => {
    if (!saved) await save();
    setBusy(true); setErr(null);
    try {
      const r = await api('/api/predict',{ method:'POST', body:JSON.stringify({date:today}) });
      setPred(r); showToast(`Prediction: ${r.predicted_state} ${SM[r.predicted_state]?.emoji||''}`);
    } catch(e) { setErr(e.message); showToast(e.message,'error'); }
    finally { setBusy(false); }
  };

  const PROMPTS = [
    "Today I felt...", "The hardest part of today was...",
    "Something I'm grateful for is...", "Right now my body feels...",
    "One thing that made me smile today...", "Tomorrow I want to...",
  ];

  return (
    <div style={{ padding:'18px 18px 0' }}>
      <div style={{ marginBottom:18 }}>
        <h2 style={{ fontSize:24 }}>My Diary</h2>
        <p style={{ fontSize:13,color:'var(--muted)',marginTop:4 }}>
          Write freely — this space is only for you.
        </p>
      </div>

      {/* Editor */}
      <div style={{
        background:'var(--warm-white)',borderRadius:'var(--radius)',
        border:'1.5px solid var(--border)',
        boxShadow:'var(--shadow-sm)',marginBottom:12,overflow:'hidden',
      }}>
        <textarea value={text} onChange={handleChange}
          placeholder="Today I felt..."
          style={{
            width:'100%',minHeight:220,padding:'20px 20px 12px',
            border:'none',resize:'vertical',
            fontFamily:'Fraunces,serif',fontSize:16,lineHeight:1.75,
            color:'var(--charcoal)',background:'transparent',
            fontStyle: text ? 'normal' : 'italic',
          }}/>
        <div style={{
          display:'flex',justifyContent:'space-between',alignItems:'center',
          padding:'10px 20px',borderTop:'1px solid var(--border)',
          background:'rgba(253,248,242,0.6)',
        }}>
          <span style={{ fontSize:12,color:'var(--muted)' }}>{wc} words</span>
          <div style={{ display:'flex',alignItems:'center',gap:8 }}>
            {saved && <span style={{ fontSize:12,color:'var(--sage)' }}>✓ Saved</span>}
            <button onClick={() => setPrompts(p=>!p)} style={{
              background:'none',border:'none',cursor:'pointer',
              fontSize:12,color:'var(--muted)',display:'flex',alignItems:'center',gap:4,
            }}>💡 Prompts</button>
          </div>
        </div>
      </div>

      {/* Prompts */}
      {prompts && (
        <div className="fade-up" style={{
          display:'flex',flexWrap:'wrap',gap:8,marginBottom:14,
        }}>
          {PROMPTS.map(p => (
            <button key={p} onClick={() => {
              setText(t => t+(t?'\n\n':'')+p);
              setWc(w => w + p.split(' ').length);
              setSaved(false); setPrompts(false);
            }} style={{
              background:'var(--warm-white)',border:'1px solid var(--border)',
              borderRadius:20,padding:'7px 14px',cursor:'pointer',
              fontSize:13,color:'var(--charcoal)',
              fontFamily:'Fraunces,serif',fontStyle:'italic',
            }}>{p}</button>
          ))}
        </div>
      )}

      {err && (
        <div style={{
          background:'#FEF2F2',border:'1px solid #FECACA',
          borderRadius:'var(--radius-xs)',padding:'10px 14px',
          marginBottom:12,fontSize:13,color:'#B91C1C',
        }}>{err}</div>
      )}

      {/* Action buttons */}
      <div style={{ display:'flex',gap:10,marginBottom:20 }}>
        <button onClick={save} disabled={saving||!text.trim()} style={{
          padding:'13px 20px',borderRadius:30,border:'1.5px solid var(--border)',
          background:'var(--warm-white)',fontSize:14,cursor:'pointer',
          color: saving||!text.trim() ? 'var(--muted)' : 'var(--charcoal)',
          opacity: saving||!text.trim() ? 0.6 : 1,
        }}>{saving ? 'Saving...' : 'Save'}</button>
        <button onClick={analyse} disabled={busy||!text.trim()} style={{
          flex:1,padding:'13px 20px',borderRadius:30,border:'none',
          background: busy||!text.trim() ? 'var(--border)' : 'var(--sage-dark)',
          color: busy||!text.trim() ? 'var(--muted)' : '#fff',
          fontSize:14,fontWeight:500,cursor:'pointer',letterSpacing:'0.2px',
        }}>{busy ? 'Analysing...' : '✦ Analyse My Day'}</button>
      </div>

      {/* Prediction result */}
      {pred && <PredResult pred={pred} />}
    </div>
  );
}

function PredResult({ pred }) {
  const m = SM[pred.predicted_state] || SM.Normal;
  return (
    <div className="fade-up" style={{
      background:m.bg,border:`1.5px solid ${m.color}40`,
      borderRadius:'var(--radius)',padding:'20px',marginBottom:20,
    }}>
      <div style={{ display:'flex',alignItems:'center',gap:14,marginBottom:18 }}>
        <div style={{
          width:54,height:54,borderRadius:'50%',background:m.color+'25',
          display:'flex',alignItems:'center',justifyContent:'center',fontSize:26,
        }} className="pulse">{m.emoji}</div>
        <div style={{ flex:1 }}>
          <p style={{ fontSize:11,color:m.color,fontWeight:600,
            textTransform:'uppercase',letterSpacing:'0.6px' }}>AI Analysis</p>
          <h3 style={{ fontSize:22,color:'var(--charcoal)',marginTop:2 }}>
            {pred.predicted_state}
          </h3>
        </div>
        <div style={{ textAlign:'right' }}>
          <p style={{ fontSize:24,fontWeight:700,color:m.color }}>
            {((pred.confidence||0)*100).toFixed(0)}%
          </p>
          <p style={{ fontSize:10,color:'var(--muted)' }}>confidence</p>
        </div>
      </div>

      {/* Score bars */}
      {pred.score_list && (
        <div style={{ display:'flex',flexDirection:'column',gap:8,marginBottom:14 }}>
          {[...pred.score_list].sort((a,b)=>b.score-a.score).slice(0,4).map(s => (
            <div key={s.label}>
              <div style={{ display:'flex',justifyContent:'space-between',
                marginBottom:4,fontSize:12 }}>
                <span style={{ color:s.is_top?s.color:'var(--muted)',
                  fontWeight:s.is_top?600:400 }}>
                  {s.emoji} {s.label}
                </span>
                <span style={{ color:'var(--muted)' }}>
                  {(s.score*100).toFixed(1)}%
                </span>
              </div>
              <div style={{ height:6,background:s.color+'18',borderRadius:3,overflow:'hidden' }}>
                <div style={{
                  height:'100%',width:`${s.score*100}%`,
                  background:s.color,borderRadius:3,
                  transition:'width 1.2s ease',
                }}/>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Influence */}
      <div style={{
        display:'flex',gap:12,fontSize:12,color:'var(--muted)',
        paddingTop:12,borderTop:`1px solid ${m.color}20`,
      }}>
        <span>📝 Text: {((pred.text_weight||0.5)*100).toFixed(0)}%</span>
        <span>•</span>
        <span>💻 Usage: {((pred.num_weight||0.5)*100).toFixed(0)}%</span>
      </div>
    </div>
  );
}

// ════════════════════════════════════════════════════════════════════════════
// TRACKER PAGE
// ════════════════════════════════════════════════════════════════════════════
function TrackerPage({ showToast }) {
  const [data, setData]    = useState(null);
  const [loading,setLoad]  = useState(true);
  const [tab, setTab]      = useState('mood');
  const [dlPdf, setDlPdf]  = useState(false);

  useEffect(() => {
    api('/api/tracker?days=7').then(d => { setData(d); setLoad(false); })
      .catch(() => setLoad(false));
  }, []);

  const downloadPDF = async (period) => {
    setDlPdf(true);
    try {
      const res = await fetch(`${API}/api/report?period=${period}`);
      if (!res.ok) { const e = await res.json(); throw new Error(e.detail); }
      const blob = await res.blob();
      const url  = URL.createObjectURL(blob);
      const a    = document.createElement('a');
      a.href     = url;
      a.download = `serenity_report_${period}_${new Date().toISOString().split('T')[0]}.pdf`;
      a.click();
      URL.revokeObjectURL(url);
      showToast('Report downloaded!');
    } catch(e) {
      showToast(e.message || 'Download failed','error');
    } finally { setDlPdf(false); }
  };

  if (loading) return <Loader msg="Loading your week..." />;

  return (
    <div style={{ padding:'18px 18px 0' }}>
      <div style={{ display:'flex',alignItems:'flex-start',justifyContent:'space-between',marginBottom:18 }}>
        <div>
          <h2 style={{ fontSize:24 }}>Weekly Tracker</h2>
          <p style={{ fontSize:13,color:'var(--muted)',marginTop:4 }}>Your emotional journey</p>
        </div>
        <button onClick={() => downloadPDF('week')} disabled={dlPdf} style={{
          background:dlPdf?'var(--border)':'var(--sage-dark)',
          color:dlPdf?'var(--muted)':'#fff',
          border:'none',borderRadius:20,padding:'8px 14px',
          fontSize:12,cursor:'pointer',display:'flex',alignItems:'center',gap:6,
          flexShrink:0,
        }}>
          {dlPdf ? '...' : '⬇️ PDF'}
        </button>
      </div>

      {/* Tab switcher */}
      <div style={{
        display:'flex',background:'var(--warm-white)',
        borderRadius:30,padding:4,marginBottom:18,
        border:'1px solid var(--border)',
      }}>
        {[['mood','😊 Mood'],['usage','💻 Usage'],['letter','📬 Letter']].map(([id,lbl]) => (
          <button key={id} onClick={() => setTab(id)} style={{
            flex:1,padding:'8px 0',borderRadius:26,border:'none',cursor:'pointer',
            background: tab===id ? 'var(--sage-dark)' : 'transparent',
            color: tab===id ? '#fff' : 'var(--muted)',
            fontSize:13,fontWeight: tab===id?500:400,transition:'all 0.25s',
          }}>{lbl}</button>
        ))}
      </div>

      {tab==='mood'   && <MoodTab   data={data} />}
      {tab==='usage'  && <UsageTab  data={data} />}
      {tab==='letter' && <LetterTab data={data} />}
    </div>
  );
}

function MoodTab({ data }) {
  const history = data?.mood_history || [];
  if (!history.length) return (
    <Empty icon="😊" title="No mood data yet"
      sub="Analyse a few diary entries to see your mood history" />
  );

  const chartData = history.map(h => ({
    day  : new Date(h.date).toLocaleDateString('en-US',{weekday:'short'}),
    conf : Math.round((h.confidence||0)*100),
    state: h.predicted_state,
    color: h.color,
  }));

  return (
    <div>
      {/* Emoji timeline */}
      <div style={{
        background:'var(--warm-white)',borderRadius:'var(--radius)',
        padding:'18px 20px',marginBottom:14,boxShadow:'var(--shadow-sm)',
      }}>
        <p style={{ fontSize:11,color:'var(--muted)',textTransform:'uppercase',
          letterSpacing:'0.6px',marginBottom:16 }}>This Week</p>
        <div style={{ display:'flex',justifyContent:'space-between',alignItems:'center' }}>
          {history.map((h,i) => {
            const m = SM[h.predicted_state]||SM.Normal;
            return (
              <div key={i} style={{ textAlign:'center' }}>
                <div style={{
                  width:42,height:42,borderRadius:'50%',
                  background:m.bg,border:`2px solid ${m.color}`,
                  display:'flex',alignItems:'center',justifyContent:'center',fontSize:18,
                  margin:'0 auto 6px',
                }}>{m.emoji}</div>
                <p style={{ fontSize:9,color:'var(--muted)' }}>
                  {new Date(h.date).toLocaleDateString('en-US',{weekday:'short'})}
                </p>
              </div>
            );
          })}
        </div>
      </div>

      {/* Area chart */}
      <div style={{
        background:'var(--warm-white)',borderRadius:'var(--radius)',
        padding:'18px 20px',boxShadow:'var(--shadow-sm)',
      }}>
        <p style={{ fontSize:11,color:'var(--muted)',textTransform:'uppercase',
          letterSpacing:'0.6px',marginBottom:14 }}>Confidence Trend</p>
        <ResponsiveContainer width="100%" height={150}>
          <AreaChart data={chartData} margin={{top:5,right:5,bottom:0,left:-25}}>
            <defs>
              <linearGradient id="cg" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%"  stopColor="#7C9E87" stopOpacity={0.35}/>
                <stop offset="95%" stopColor="#7C9E87" stopOpacity={0}/>
              </linearGradient>
            </defs>
            <XAxis dataKey="day" tick={{fontSize:10,fill:'#9A9A9A'}} axisLine={false} tickLine={false}/>
            <YAxis tick={{fontSize:10,fill:'#9A9A9A'}} axisLine={false} tickLine={false} domain={[0,100]}/>
            <Tooltip contentStyle={{borderRadius:12,border:'none',boxShadow:'0 4px 20px rgba(0,0,0,0.1)',fontSize:11}}/>
            <Area type="monotone" dataKey="conf" stroke="#7C9E87" strokeWidth={2.5}
              fill="url(#cg)" dot={{fill:'#7C9E87',r:4,strokeWidth:0}}/>
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function UsageTab({ data }) {
  const trend = data?.usage_trend || [];
  if (!trend.length) return (
    <Empty icon="💻" title="No usage data yet"
      sub="The collector needs a few days to gather data" />
  );

  const chartData = trend.map(u => ({
    day    : new Date(u.date).toLocaleDateString('en-US',{weekday:'short'}),
    Screen : Math.round(u.screen_time_mins||0),
    Social : Math.round(u.social_media_mins||0),
    Active : Math.round(u.active_mins||0),
    Late   : Math.round(u.late_night_mins||0),
  }));

  const COLORS = {Screen:'#7B8EC8',Social:'#E8845A',Active:'#7C9E87',Late:'#C17BC0'};

  return (
    <div>
      <div style={{
        background:'var(--warm-white)',borderRadius:'var(--radius)',
        padding:'18px 20px',marginBottom:14,boxShadow:'var(--shadow-sm)',
      }}>
        <p style={{ fontSize:11,color:'var(--muted)',textTransform:'uppercase',
          letterSpacing:'0.6px',marginBottom:14 }}>Daily Usage (mins)</p>
        <ResponsiveContainer width="100%" height={180}>
          <LineChart data={chartData} margin={{top:5,right:5,bottom:0,left:-25}}>
            <XAxis dataKey="day" tick={{fontSize:10,fill:'#9A9A9A'}} axisLine={false} tickLine={false}/>
            <YAxis tick={{fontSize:10,fill:'#9A9A9A'}} axisLine={false} tickLine={false}/>
            <Tooltip contentStyle={{borderRadius:12,border:'none',boxShadow:'0 4px 20px rgba(0,0,0,0.1)',fontSize:11}}/>
            {Object.entries(COLORS).map(([k,c]) => (
              <Line key={k} type="monotone" dataKey={k} stroke={c} strokeWidth={2}
                dot={{r:3,strokeWidth:0,fill:c}} activeDot={{r:5}}/>
            ))}
          </LineChart>
        </ResponsiveContainer>
        <div style={{ display:'flex',gap:14,marginTop:12,flexWrap:'wrap' }}>
          {Object.entries(COLORS).map(([k,c]) => (
            <div key={k} style={{ display:'flex',alignItems:'center',gap:5 }}>
              <div style={{ width:8,height:8,borderRadius:'50%',background:c }}/>
              <span style={{ fontSize:10,color:'var(--muted)' }}>{k}</span>
            </div>
          ))}
        </div>
      </div>

      {data?.weekly?.available && (
        <div style={{
          background:'var(--warm-white)',borderRadius:'var(--radius)',
          padding:'18px 20px',boxShadow:'var(--shadow-sm)',
        }}>
          <p style={{ fontSize:11,color:'var(--muted)',textTransform:'uppercase',
            letterSpacing:'0.6px',marginBottom:14 }}>Weekly Averages</p>
          <div style={{ display:'grid',gridTemplateColumns:'1fr 1fr',gap:10 }}>
            {[
              { icon:'💻',label:'Screen/day',  val:`${Math.round(data.weekly.avg_screen_time_mins||0)}m` },
              { icon:'📱',label:'Social/day',  val:`${Math.round(data.weekly.avg_social_media_mins||0)}m` },
              { icon:'📈',label:'Trend',        val: data.weekly.trend||'stable' },
              { icon:'📓',label:'Days logged',  val:`${data.weekly.days_analysed||0}` },
            ].map(s => (
              <div key={s.label} style={{
                background:'var(--cream)',borderRadius:'var(--radius-sm)',
                padding:'12px 14px',
              }}>
                <p style={{ fontSize:20,marginBottom:4 }}>{s.icon}</p>
                <p style={{ fontSize:15,fontWeight:600 }}>{s.val}</p>
                <p style={{ fontSize:10,color:'var(--muted)',marginTop:2 }}>{s.label}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function LetterTab({ data }) {
  const w = data?.weekly;
  if (!w?.available) return (
    <Empty icon="📬" title="Letter not ready yet"
      sub="Keep journaling for 3+ days to receive your weekly letter" />
  );
  return (
    <div style={{
      background:'linear-gradient(160deg,#FDF8F2,#EDF4EF)',
      border:'1.5px solid var(--border)',borderRadius:'var(--radius)',
      padding:'24px 22px',boxShadow:'var(--shadow-sm)',
    }}>
      <div style={{ display:'flex',alignItems:'center',gap:10,marginBottom:18 }}>
        <span style={{ fontSize:28 }}>📬</span>
        <div>
          <p style={{ fontSize:11,color:'var(--muted)',textTransform:'uppercase',letterSpacing:'0.5px' }}>
            Weekly Letter
          </p>
          <p style={{ fontSize:14,color:'var(--sage-dark)',fontWeight:500 }}>
            From your Serenity
          </p>
        </div>
      </div>
      <p style={{
        fontSize:15,lineHeight:1.85,color:'var(--charcoal)',
        fontFamily:'Fraunces,serif',whiteSpace:'pre-line',
      }}>{w.weekly_letter}</p>
      <div style={{
        marginTop:16,paddingTop:14,borderTop:'1px solid var(--border)',
        display:'flex',gap:14,fontSize:12,color:'var(--muted)',flexWrap:'wrap',
      }}>
        <span>Dominant: {w.dominant_state}</span>
        <span>•</span>
        <span>Trend: {w.trend}</span>
        <span>•</span>
        <span>{w.days_analysed} days</span>
      </div>
    </div>
  );
}

// ════════════════════════════════════════════════════════════════════════════
// PROFILE PAGE
// ════════════════════════════════════════════════════════════════════════════
function ProfilePage({ showToast }) {
  const [fl,     setFl]     = useState(null);
  const [health, setHealth] = useState(null);

  useEffect(() => {
    api('/api/fl/status').then(setFl).catch(()=>{});
    api('/health').then(setHealth).catch(()=>{});
  }, []);

  return (
    <div style={{ padding:'18px 18px 0' }}>
      <h2 style={{ fontSize:24,marginBottom:4 }}>Profile</h2>
      <p style={{ fontSize:13,color:'var(--muted)',marginBottom:18 }}>
        Your privacy, your data, your way.
      </p>

      {/* Privacy card */}
      <div style={{
        background:'linear-gradient(135deg,#4A7A5A,#7C9E87)',
        borderRadius:'var(--radius)',padding:'22px',color:'#fff',marginBottom:14,
        position:'relative',overflow:'hidden',
      }}>
        <div style={{ position:'absolute',right:-20,top:-20,width:90,height:90,
          borderRadius:'50%',background:'rgba(255,255,255,0.08)' }}/>
        <p style={{ fontSize:26,marginBottom:10 }}>🔒</p>
        <h3 style={{ fontSize:18,fontWeight:400,marginBottom:8 }}>
          Your Data is 100% Private
        </h3>
        <p style={{ fontSize:13,opacity:0.85,lineHeight:1.65 }}>
          Diary entries, app usage, and predictions are stored only on your
          laptop's MySQL database. Only anonymous weight updates are ever
          shared — never your content.
        </p>
      </div>

      {/* FL Status */}
      <Section title="Federated Learning">
        {fl ? (
          <>
            <StatusRow icon="📤" label="Uploaded today"
              val={fl.uploaded_today?'Yes':'Not yet'} ok={fl.uploaded_today}/>
            <StatusRow icon="📡" label="Can upload now"
              val={fl.can_upload_now?'Ready':'Waiting'} ok={fl.can_upload_now}/>
            <StatusRow icon="📊" label="Enough data"
              val={fl.enough_data?'Yes':'Need more days'} ok={fl.enough_data}/>
            <p style={{ fontSize:12,color:'var(--muted)',marginTop:8,lineHeight:1.5 }}>
              {fl.reason}
            </p>
          </>
        ) : <p style={{ fontSize:13,color:'var(--muted)' }}>Loading...</p>}
      </Section>

      {/* System health */}
      <Section title="System Status">
        {health ? (
          <>
            <StatusRow icon="⚡" label="API"      val={health.api}      ok={health.api==='ok'}/>
            <StatusRow icon="🗄️" label="Database" val={health.database} ok={health.database==='ok'}/>
            <StatusRow icon="🧠" label="Model"    val={health.model}    ok={health.model?.startsWith('ok')}/>
          </>
        ) : <p style={{ fontSize:13,color:'var(--muted)' }}>Checking...</p>}
      </Section>

      {/* About */}
      <Section title="About Serenity">
        {[
          ['🧠','Model',    'GRU + Fusion (text + usage)'],
          ['🔐','Privacy',  'Federated Learning on-device'],
          ['💾','Storage',  'Local MySQL 8.0'],
          ['🏷️','Version',  '1.0.0'],
        ].map(([icon,label,val]) => (
          <div key={label} style={{
            display:'flex',justifyContent:'space-between',
            padding:'8px 0',borderBottom:'1px solid var(--border)',fontSize:13,
          }}>
            <span style={{ color:'var(--muted)' }}>{icon} {label}</span>
            <span style={{ color:'var(--charcoal)' }}>{val}</span>
          </div>
        ))}
      </Section>

      <div style={{ height:10 }}/>
    </div>
  );
}

function Section({ title, children }) {
  return (
    <div style={{
      background:'var(--warm-white)',borderRadius:'var(--radius)',
      padding:'18px 20px',marginBottom:14,boxShadow:'var(--shadow-sm)',
    }}>
      <p style={{ fontSize:11,color:'var(--muted)',textTransform:'uppercase',
        letterSpacing:'0.6px',marginBottom:14 }}>{title}</p>
      {children}
    </div>
  );
}

function StatusRow({ icon, label, val, ok }) {
  return (
    <div style={{ display:'flex',alignItems:'center',
      justifyContent:'space-between',marginBottom:10 }}>
      <span style={{ fontSize:13,color:'var(--charcoal)' }}>{icon} {label}</span>
      <span style={{
        fontSize:12,padding:'3px 12px',borderRadius:20,fontWeight:500,
        background: ok ? '#EDF4EF' : '#FBF5E6',
        color:      ok ? '#4A7A5A' : '#92611A',
      }}>{val}</span>
    </div>
  );
}

// ── Shared ────────────────────────────────────────────────────────────────────
function Loader({ msg }) {
  return (
    <div style={{ display:'flex',flexDirection:'column',alignItems:'center',
      justifyContent:'center',minHeight:'60vh',gap:16 }}>
      <div style={{
        width:44,height:44,borderRadius:'50%',
        border:'3px solid var(--border)',borderTopColor:'var(--sage)',
        animation:'spin 0.8s linear infinite',
      }}/>
      <p style={{ fontSize:14,color:'var(--muted)',fontFamily:'Fraunces,serif',fontStyle:'italic' }}>
        {msg}
      </p>
    </div>
  );
}

function Empty({ icon, title, sub }) {
  return (
    <div style={{
      textAlign:'center',padding:'40px 20px',
      background:'var(--warm-white)',borderRadius:'var(--radius)',
      border:'2px dashed var(--border)',
    }}>
      <p style={{ fontSize:40,marginBottom:12 }}>{icon}</p>
      <h3 style={{ fontSize:17,color:'var(--charcoal)' }}>{title}</h3>
      <p style={{ fontSize:13,color:'var(--muted)',marginTop:6,lineHeight:1.5 }}>{sub}</p>
    </div>
  );
}
