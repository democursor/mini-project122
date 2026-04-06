import { useQuery, useMutation, useQueryClient } from 'react-query'
import { useState } from 'react'
import { FileText, Trash2, Eye, Search as SearchIcon, Upload, Clock, Filter } from 'lucide-react'
import toast from 'react-hot-toast'
import { documentsAPI } from '../api/client'
import { Link, useNavigate } from 'react-router-dom'

function getRelativeTime(d) {
  if (!d) return 'Unknown'
  const diff = Math.floor((new Date()-new Date(d))/1000)
  if (diff<60) return 'Just now'
  if (diff<3600) return `${Math.floor(diff/60)}m ago`
  if (diff<86400) return `${Math.floor(diff/3600)}h ago`
  if (diff<604800) return `${Math.floor(diff/86400)}d ago`
  return new Date(d).toLocaleDateString()
}

const ACCENTS = [
  {dim:'var(--amber-dim)',  border:'var(--amber-border)',  color:'var(--amber)'},
  {dim:'var(--violet-dim)', border:'var(--violet-border)', color:'var(--violet)'},
  {dim:'var(--cyan-dim)',   border:'var(--cyan-border)',   color:'var(--cyan)'},
  {dim:'var(--emerald-dim)',border:'var(--emerald-border)',color:'var(--emerald)'},
]

export default function Documents() {
  const queryClient = useQueryClient()
  const navigate = useNavigate()
  const {data,isLoading} = useQuery('documents', documentsAPI.list)
  const [hovered,setHovered] = useState(null)
  const [search,setSearch] = useState('')

  const deleteMutation = useMutation(documentsAPI.delete, {
    onSuccess: ()=>{ toast.success('Document deleted'); queryClient.invalidateQueries('documents') },
    onError: ()=> toast.error('Failed to delete document'),
  })

  const documents = data?.data?.documents || []
  const filtered = documents.filter(d=>(d.title||d.filename).toLowerCase().includes(search.toLowerCase()))

  if (isLoading) return (
    <div>
      <div className="mb-8">
        <div className="section-title mb-2">Library</div>
        <h1 className="page-header-title">Documents</h1>
        <p className="page-header-sub">Manage your uploaded research papers</p>
      </div>
      <div style={{display:'flex',flexDirection:'column',gap:'10px'}}>
        {[1,2,3].map(i=><div key={i} className="skeleton" style={{height:'86px',borderRadius:'16px'}}/>)}
      </div>
    </div>
  )

  return (
    <div>
      <div className="mb-8" style={{display:'flex',alignItems:'flex-end',justifyContent:'space-between',gap:'16px'}}>
        <div>
          <div className="section-title mb-2">Library</div>
          <h1 className="page-header-title">Documents</h1>
          <p className="page-header-sub">{documents.length} paper{documents.length!==1?'s':''} in your collection</p>
        </div>
        <div style={{display:'flex',alignItems:'center',gap:'10px',flexShrink:0}}>
          <div style={{display:'flex',alignItems:'center',gap:'8px',background:'var(--bg-surface)',border:'1px solid var(--border-default)',borderRadius:'10px',padding:'0 12px',height:'38px',transition:'all 0.2s'}}
            onFocusCapture={e=>e.currentTarget.style.borderColor='var(--violet)'}
            onBlurCapture={e=>e.currentTarget.style.borderColor='var(--border-default)'}>
            <SearchIcon size={13} style={{color:'var(--text-muted)'}}/>
            <input type="text" placeholder="Filter documents..." value={search} onChange={e=>setSearch(e.target.value)}
              style={{background:'transparent',border:'none',outline:'none',fontSize:'13px',color:'var(--text-primary)',width:'160px',fontFamily:'var(--font-body)'}}/>
          </div>
          <Link to="/upload">
            <button className="btn-amber"><Upload size={13}/> Upload New</button>
          </Link>
        </div>
      </div>

      {filtered.length===0 ? (
        <div className="glass text-center" style={{padding:'60px 24px',border:'1px solid var(--border-subtle)'}}>
          <div style={{width:'60px',height:'60px',borderRadius:'18px',background:'var(--bg-elevated)',border:'1px solid var(--border-default)',display:'flex',alignItems:'center',justifyContent:'center',margin:'0 auto 16px'}}>
            <FileText size={26} style={{color:'var(--text-faint)'}}/>
          </div>
          <div style={{fontSize:'15px',fontWeight:700,color:'var(--text-secondary)',fontFamily:'var(--font-display)',marginBottom:'6px'}}>
            {search?'No documents found':'No documents yet'}
          </div>
          <div style={{fontSize:'13px',color:'var(--text-faint)',marginBottom:'20px'}}>
            {search?'Try a different search term':'Upload your first research paper to get started'}
          </div>
          {!search && <Link to="/upload"><button className="btn-amber"><Upload size={14}/> Upload Paper</button></Link>}
        </div>
      ) : (
        <div style={{display:'flex',flexDirection:'column',gap:'10px'}}>
          {filtered.map((doc,i)=>{
            const ac = ACCENTS[i%ACCENTS.length]
            const keywords = doc.keywords?.slice(0,3)||[]
            const isHov = hovered===doc.document_id
            return (
              <div key={doc.document_id} className="card-3d glass"
                style={{padding:'14px 18px',cursor:'pointer',border:`1px solid ${isHov?ac.border:'var(--border-subtle)'}`,borderRadius:'16px',transition:'all 0.3s cubic-bezier(.34,1.56,.64,1)'}}
                onMouseEnter={()=>setHovered(doc.document_id)}
                onMouseLeave={()=>setHovered(null)}>
                <div style={{display:'flex',alignItems:'center',gap:'14px'}}>
                  <div style={{width:'42px',height:'42px',borderRadius:'12px',display:'flex',alignItems:'center',justifyContent:'center',flexShrink:0,background:ac.dim,border:`1px solid ${ac.border}`,boxShadow:isHov?`0 0 16px ${ac.color}33`:'none',transition:'box-shadow 0.3s'}}>
                    <FileText size={18} style={{color:ac.color}}/>
                  </div>
                  <div style={{flex:1,minWidth:0}}>
                    <div style={{fontSize:'13px',fontWeight:700,color:'var(--text-primary)',fontFamily:'var(--font-display)',whiteSpace:'nowrap',overflow:'hidden',textOverflow:'ellipsis',marginBottom:'4px'}}
                      title={doc.title||doc.filename}>
                      {doc.title||doc.filename}
                    </div>
                    <div style={{display:'flex',alignItems:'center',gap:'5px',fontSize:'11px',color:'var(--text-muted)',marginBottom:'6px'}}>
                      <Clock size={10}/> {getRelativeTime(doc.upload_date||doc.created_at)}
                      {doc.pages&&<span>· {doc.pages} pages</span>}
                    </div>
                    {keywords.length>0 && (
                      <div style={{display:'flex',gap:'5px',flexWrap:'wrap'}}>
                        {keywords.map((kw,j)=><span key={j} className="kw-pill">{kw}</span>)}
                      </div>
                    )}
                  </div>
                  <span className={`badge badge-${doc.status==='completed'?'done':doc.status==='processing'?'processing':'done'}`} style={{flexShrink:0}}>
                    {doc.status==='processing'&&<div className="dot-pulse" style={{width:'5px',height:'5px',marginRight:'4px'}}/>}
                    {doc.status||'completed'}
                  </span>
                  <div style={{display:'flex',alignItems:'center',gap:'6px',flexShrink:0,opacity:isHov?1:0,transition:'opacity 0.15s'}}>
                    {[
                      {icon:Eye,  label:'View',   onClick:()=>navigate(`/documents/${doc.document_id}`), hBg:'var(--amber-dim)',   hBdr:'var(--amber-border)'},
                      {icon:SearchIcon,label:'Search',onClick:()=>navigate('/search'), hBg:'var(--violet-dim)',  hBdr:'var(--violet-border)'},
                      {icon:Trash2,label:'Delete', onClick:()=>{if(window.confirm(`Delete "${doc.title||doc.filename}"?`))deleteMutation.mutate(doc.document_id)}, hBg:'var(--rose-dim)', hBdr:'var(--rose-border)'},
                    ].map(({icon:Icon,label,onClick,hBg,hBdr})=>(
                      <button key={label} title={label}
                        onClick={e=>{e.stopPropagation();onClick()}}
                        style={{width:'28px',height:'28px',borderRadius:'8px',display:'flex',alignItems:'center',justifyContent:'center',background:'var(--bg-elevated)',border:'1px solid var(--border-subtle)',cursor:'pointer',transition:'all 0.15s'}}
                        onMouseEnter={e=>{e.currentTarget.style.background=hBg;e.currentTarget.style.borderColor=hBdr}}
                        onMouseLeave={e=>{e.currentTarget.style.background='var(--bg-elevated)';e.currentTarget.style.borderColor='var(--border-subtle)'}}>
                        <Icon size={13} style={{color:'var(--text-muted)'}}/>
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
