import { useState } from 'react'
import { useQuery } from 'react-query'
import { Network, FileText, Tag, GitBranch, Hash, Layers } from 'lucide-react'
import { graphAPI } from '../api/client'

const STATS = [
  { key:'total_papers',        label:'Papers',        icon:FileText,  color:'var(--amber)',   dim:'var(--amber-dim)',   border:'var(--amber-border)',  top:'2px solid var(--amber)',   cls:'card-3d-amber' },
  { key:'total_concepts',      label:'Concepts',      icon:Tag,       color:'var(--cyan)',    dim:'var(--cyan-dim)',    border:'var(--cyan-border)',   top:'2px solid var(--cyan)',    cls:'card-3d-cyan' },
  { key:'total_mentions',      label:'Mentions',      icon:Hash,      color:'var(--violet)',  dim:'var(--violet-dim)', border:'var(--violet-border)', top:'2px solid var(--violet)',  cls:'card-3d-violet' },
  { key:'total_relationships', label:'Relationships', icon:GitBranch, color:'var(--emerald)', dim:'var(--emerald-dim)',border:'var(--emerald-border)',top:'2px solid var(--emerald)',cls:'card-3d-emerald' },
]

export default function KnowledgeGraph() {
  const [selectedPaper,setSelectedPaper] = useState(null)
  const { data:stats }    = useQuery('graphStats', graphAPI.stats)
  const { data:papers }   = useQuery('papers',     ()=>graphAPI.papers(20))
  const { data:concepts } = useQuery('concepts',   ()=>graphAPI.concepts(30))

  const statsData    = stats?.data || {}
  const papersList   = papers?.data || []
  const conceptsList = concepts?.data || []

  return (
    <div>
      <div className="mb-8">
        <div className="section-title mb-2">Intelligence</div>
        <h1 className="page-header-title">Knowledge Graph</h1>
        <p className="page-header-sub">Explore relationships between papers and concepts</p>
      </div>

      {/* Stats */}
      <div style={{display:'grid',gridTemplateColumns:'repeat(4,minmax(0,1fr))',gap:'12px',marginBottom:'24px'}}>
        {STATS.map(s=>{
          const Icon=s.icon
          return (
            <div key={s.key} className={`stat-card card-3d ${s.cls}`} style={{borderTop:s.top,cursor:'default'}}>
              <div style={{width:'36px',height:'36px',borderRadius:'10px',display:'flex',alignItems:'center',justifyContent:'center',marginBottom:'14px',background:s.dim,border:`1px solid ${s.border}`}}>
                <Icon size={16} style={{color:s.color}}/>
              </div>
              <div className="stat-card-num mb-1">{statsData[s.key]||0}</div>
              <div className="stat-card-label">{s.label}</div>
              <Icon size={56} style={{position:'absolute',bottom:8,right:8,color:s.color,opacity:0.04}}/>
            </div>
          )
        })}
      </div>

      {/* Two panels */}
      <div style={{display:'grid',gridTemplateColumns:'minmax(0,1fr) minmax(0,1fr)',gap:'16px',marginBottom:'16px'}}>
        {/* Papers */}
        <div className="glass" style={{padding:'20px',border:'1px solid var(--border-subtle)'}}>
          <div style={{display:'flex',alignItems:'center',gap:'10px',marginBottom:'16px'}}>
            <div style={{width:'30px',height:'30px',borderRadius:'8px',display:'flex',alignItems:'center',justifyContent:'center',background:'var(--amber-dim)',border:'1px solid var(--amber-border)'}}>
              <FileText size={14} style={{color:'var(--amber)'}}/>
            </div>
            <span style={{fontSize:'13px',fontWeight:700,color:'var(--text-secondary)',fontFamily:'var(--font-display)'}}>Papers in Graph</span>
          </div>
          {papersList.length===0 ? (
            <div style={{textAlign:'center',padding:'40px 20px'}}>
              <div style={{width:'64px',height:'64px',borderRadius:'50%',border:'2px dashed var(--border-strong)',display:'flex',alignItems:'center',justifyContent:'center',margin:'0 auto 12px'}}>
                <FileText size={24} style={{color:'var(--text-faint)'}}/>
              </div>
              <div style={{fontSize:'13px',color:'var(--text-muted)',fontFamily:'var(--font-display)',fontWeight:600,marginBottom:'4px'}}>No papers in graph yet</div>
              <div style={{fontSize:'11px',color:'var(--text-faint)'}}>Upload and process papers first</div>
            </div>
          ) : (
            <div style={{display:'flex',flexDirection:'column',gap:'6px',maxHeight:'320px',overflowY:'auto'}}>
              {papersList.map(paper=>(
                <div key={paper.id} onClick={()=>setSelectedPaper(paper.id)}
                  style={{
                    padding:'10px 12px',borderRadius:'10px',cursor:'pointer',
                    background:selectedPaper===paper.id?'var(--amber-dim)':'var(--bg-elevated)',
                    border:`1px solid ${selectedPaper===paper.id?'var(--amber-border)':'var(--border-subtle)'}`,
                    transition:'all 0.15s ease'
                  }}
                  onMouseEnter={e=>{if(selectedPaper!==paper.id){e.currentTarget.style.background='var(--bg-hover)'}}}
                  onMouseLeave={e=>{if(selectedPaper!==paper.id){e.currentTarget.style.background='var(--bg-elevated)'}}}>
                  <div style={{fontSize:'12px',fontWeight:700,color:selectedPaper===paper.id?'var(--amber)':'var(--text-primary)',fontFamily:'var(--font-display)',marginBottom:'3px'}}>{paper.title}</div>
                  {paper.authors?.length>0 && <div style={{fontSize:'11px',color:'var(--text-muted)'}}>{paper.authors.join(', ')}</div>}
                  {paper.year && <div style={{fontSize:'10px',color:'var(--text-faint)',fontFamily:'var(--font-mono)',marginTop:'2px'}}>{paper.year}</div>}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Concepts */}
        <div className="glass" style={{padding:'20px',border:'1px solid var(--border-subtle)'}}>
          <div style={{display:'flex',alignItems:'center',gap:'10px',marginBottom:'16px'}}>
            <div style={{width:'30px',height:'30px',borderRadius:'8px',display:'flex',alignItems:'center',justifyContent:'center',background:'var(--violet-dim)',border:'1px solid var(--violet-border)'}}>
              <Tag size={14} style={{color:'var(--violet)'}}/>
            </div>
            <span style={{fontSize:'13px',fontWeight:700,color:'var(--text-secondary)',fontFamily:'var(--font-display)'}}>Top Concepts</span>
          </div>
          {conceptsList.length===0 ? (
            <div style={{textAlign:'center',padding:'40px 20px'}}>
              <div style={{width:'64px',height:'64px',borderRadius:'50%',border:'2px dashed var(--border-strong)',display:'flex',alignItems:'center',justifyContent:'center',margin:'0 auto 12px'}}>
                <Tag size={24} style={{color:'var(--text-faint)'}}/>
              </div>
              <div style={{fontSize:'13px',color:'var(--text-muted)',fontFamily:'var(--font-display)',fontWeight:600,marginBottom:'4px'}}>No concepts extracted yet</div>
              <div style={{fontSize:'11px',color:'var(--text-faint)'}}>Process documents to extract concepts</div>
            </div>
          ) : (
            <div style={{display:'flex',flexWrap:'wrap',gap:'8px',maxHeight:'320px',overflowY:'auto'}}>
              {conceptsList.map(concept=>(
                <button key={concept.name} className="concept-chip"
                  style={{fontSize:`${Math.min(11+(concept.frequency/8),16)}px`}}>
                  {concept.name}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Graph placeholder */}
      {papersList.length===0 && conceptsList.length===0 && (
        <div className="glass text-center" style={{padding:'52px',border:'1px solid var(--border-subtle)'}}>
          <div style={{display:'flex',flexWrap:'wrap',justifyContent:'center',gap:'10px',maxWidth:'180px',margin:'0 auto 20px'}}>
            {Array.from({length:16}).map((_,i)=>(
              <div key={i} style={{
                width:i%5===0?10:6,height:i%5===0?10:6,borderRadius:'50%',
                background:i%5===0?'var(--amber)':'var(--border-default)',
                animation:`glow-pulse ${1.5+(i%4)*0.3}s ease-in-out ${i*0.1}s infinite`,
                boxShadow:i%5===0?'0 0 8px var(--amber-glow)':'none'
              }}/>
            ))}
          </div>
          <div style={{fontFamily:'var(--font-display)',fontSize:'15px',fontWeight:700,color:'var(--text-secondary)',marginBottom:'8px'}}>
            Graph visualization ready
          </div>
          <div style={{fontSize:'13px',color:'var(--text-muted)'}}>
            Upload and process papers to explore concept relationships
          </div>
        </div>
      )}
    </div>
  )
}
