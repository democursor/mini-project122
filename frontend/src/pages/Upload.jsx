import { useState, useEffect, useRef } from 'react'
import { useMutation, useQueryClient } from 'react-query'
import { Upload as UploadIcon, FileText, Check, Sparkles, ArrowRight, X, Zap, AlertCircle } from 'lucide-react'
import toast from 'react-hot-toast'
import { documentsAPI } from '../api/client'
import { useNavigate } from 'react-router-dom'

const STAGES = [
  { label:'Validating PDF',        desc:'Checking format and integrity' },
  { label:'Extracting Text',       desc:'Parsing document structure' },
  { label:'Semantic Chunking',     desc:'Splitting into contextual segments' },
  { label:'Extracting Concepts',   desc:'NER and keyphrase detection' },
  { label:'Building Graph',        desc:'Creating Neo4j nodes and relationships' },
  { label:'Generating Embeddings', desc:'Vectorizing with Sentence Transformers' },
]
const STEP_DURATION = 3000

export default function Upload() {
  const [selectedFile,setSelectedFile] = useState(null)
  const [dragActive,setDragActive]     = useState(false)
  const [phase,setPhase]               = useState('idle')
  const [documentId,setDocumentId]     = useState(null)
  const [docStatus,setDocStatus]       = useState(null)
  const [visualStep,setVisualStep]     = useState(0)
  const [allDone,setAllDone]           = useState(false)
  const [backendDone,setBackendDone]   = useState(false)
  const pollRef = useRef(null); const stepRef = useRef(null)
  const queryClient = useQueryClient(); const navigate = useNavigate()

  useEffect(()=>()=>{clearInterval(pollRef.current);clearInterval(stepRef.current)},[])

  useEffect(()=>{
    if(backendDone && visualStep>=STAGES.length-1 && phase==='processing') finishSuccess()
  },[backendDone,visualStep])

  useEffect(()=>{
    if(!documentId||phase!=='processing') return
    
    let retryCount = 0
    const MAX_RETRIES = 10
    const INITIAL_DELAY = 1000  // 1 second
    const MAX_DELAY = 30000      // 30 seconds
    const TIMEOUT = 5 * 60 * 1000 // 5 minutes total timeout
    const startTime = Date.now()
    
    const poll=async()=>{
      try{
        // Check for total timeout
        if (Date.now() - startTime > TIMEOUT) {
          clearInterval(pollRef.current)
          setPhase('failed')
          toast.error('Processing timeout - operation took too long')
          return
        }
        
        const res=await documentsAPI.get(documentId)
        const status=res?.data?.status||res?.data?.processing_status
        setDocStatus(status)
        
        if(status==='completed'){
          clearInterval(pollRef.current)
          setBackendDone(true)
          console.log('Document processing completed')
        }
        else if(status==='failed'){
          clearInterval(pollRef.current)
          clearInterval(stepRef.current)
          setPhase('failed')
          toast.error('Processing failed')
        }
        else {
          // Still processing - schedule next poll with exponential backoff
          retryCount++
          if (retryCount >= MAX_RETRIES) {
            clearInterval(pollRef.current)
            setPhase('failed')
            toast.error('Processing timeout - max retries reached')
            return
          }
          
          // Calculate next delay: double each time, capped at MAX_DELAY
          const nextDelay = Math.min(INITIAL_DELAY * Math.pow(2, retryCount - 1), MAX_DELAY)
          console.log(`Polling retry ${retryCount}/${MAX_RETRIES}, next delay: ${nextDelay}ms`)
          
          clearInterval(pollRef.current)
          pollRef.current = setTimeout(poll, nextDelay)
        }
      }catch(e){
        console.warn('Polling error:', e)
        retryCount++
        if (retryCount >= MAX_RETRIES) {
          clearInterval(pollRef.current)
          setPhase('failed')
          toast.error('Failed to check processing status')
        }
      }
    }
    
    // Start first poll immediately
    poll()
    
    return()=>{
      clearInterval(pollRef.current)
      clearTimeout(pollRef.current)
    }
  },[documentId,phase])

  useEffect(()=>{
    if(phase!=='processing') return
    stepRef.current=setInterval(()=>{
      setVisualStep(prev=>{const n=prev+1;if(n>=STAGES.length){clearInterval(stepRef.current);return prev}return n})
    },STEP_DURATION)
    return()=>clearInterval(stepRef.current)
  },[phase])

  const finishSuccess=()=>{
    clearInterval(pollRef.current);clearInterval(stepRef.current)
    setVisualStep(STAGES.length);setAllDone(true);setPhase('done')
    queryClient.invalidateQueries('documents')
    toast.success('Document processed!',{style:{background:'#0C0C1A',color:'#EEEEFF',border:'1px solid rgba(16,185,129,0.3)'}})
  }

  const uploadMutation = useMutation(documentsAPI.upload,{
    onSuccess:(data)=>{
      const id=data?.data?.document_id||data?.data?.id
      if(!id){toast.error('No document_id returned');setPhase('idle');return}
      setDocumentId(id);setVisualStep(0);setBackendDone(false);setPhase('processing')
    },
    onError:(e)=>{toast.error(e.response?.data?.detail||'Upload failed');setPhase('idle')}
  })

  const handleDrag=(e)=>{e.preventDefault();e.stopPropagation();setDragActive(e.type==='dragenter'||e.type==='dragover')}
  const handleDrop=(e)=>{e.preventDefault();e.stopPropagation();setDragActive(false);const f=e.dataTransfer.files[0];if(f?.type==='application/pdf')setSelectedFile(f);else toast.error('Please upload a PDF file')}
  const handleFileChange=(e)=>{const f=e.target.files[0];if(f?.type==='application/pdf')setSelectedFile(f);else toast.error('Please upload a PDF file')}
  const handleUpload=()=>{if(!selectedFile)return;setPhase('uploading');uploadMutation.mutate(selectedFile)}
  const handleReset=()=>{clearInterval(pollRef.current);clearInterval(stepRef.current);setSelectedFile(null);setDocumentId(null);setDocStatus(null);setVisualStep(0);setAllDone(false);setBackendDone(false);setPhase('idle')}

  const isUploading=phase==='uploading', isProcessing=phase==='processing'

  return (
    <div style={{maxWidth:'640px',margin:'0 auto'}}>
      <div className="mb-8">
        <div className="section-title mb-2">Ingest</div>
        <h1 className="page-header-title">Upload Document</h1>
        <p className="page-header-sub">Upload research papers in PDF format for AI-powered processing</p>
      </div>

      {/* SUCCESS */}
      {phase==='done' && (
        <div style={{borderRadius:'20px',padding:'40px',textAlign:'center',background:'linear-gradient(135deg,rgba(16,185,129,0.08),rgba(16,185,129,0.04))',border:'1px solid var(--emerald-border)',boxShadow:'0 0 40px rgba(16,185,129,0.08)'}}>
          <div style={{width:'64px',height:'64px',borderRadius:'50%',display:'flex',alignItems:'center',justifyContent:'center',margin:'0 auto 20px',background:'var(--emerald-dim)',border:'2px solid var(--emerald)',boxShadow:'0 0 30px var(--emerald-glow)'}}>
            <Check size={28} style={{color:'var(--emerald)'}}/>
          </div>
          <h2 style={{fontFamily:'var(--font-display)',fontSize:'22px',fontWeight:800,color:'var(--text-primary)',marginBottom:'6px'}}>Processing Complete</h2>
          <div style={{fontSize:'13px',color:'var(--text-muted)',marginBottom:'4px'}}>{selectedFile?.name}</div>
          <div style={{fontSize:'11px',color:'var(--text-faint)',fontFamily:'var(--font-mono)',marginBottom:'28px'}}>{(selectedFile?.size/1024/1024).toFixed(2)} MB · PDF Document</div>
          <div style={{display:'flex',alignItems:'center',justifyContent:'center',gap:'10px',flexWrap:'wrap'}}>
            <button className="btn-ghost" onClick={()=>navigate('/documents')}><FileText size={14}/> View Document</button>
            <button className="btn-ghost" onClick={()=>navigate('/search')}><Zap size={14}/> Search Paper</button>
            <button className="btn-amber" onClick={()=>navigate('/chat')}><Sparkles size={14}/> Ask AI</button>
          </div>
          <button onClick={handleReset} style={{display:'block',margin:'20px auto 0',fontSize:'12px',color:'var(--text-faint)',background:'none',border:'none',cursor:'pointer'}}
            onMouseEnter={e=>e.currentTarget.style.color='var(--text-secondary)'}
            onMouseLeave={e=>e.currentTarget.style.color='var(--text-faint)'}>
            Upload another →
          </button>
        </div>
      )}

      {/* FAILED */}
      {phase==='failed' && (
        <div style={{borderRadius:'20px',padding:'40px',textAlign:'center',background:'rgba(244,63,94,0.05)',border:'1px solid var(--rose-border)'}}>
          <AlertCircle size={36} style={{color:'var(--rose)',margin:'0 auto 12px',display:'block'}}/>
          <h2 style={{fontFamily:'var(--font-display)',fontSize:'16px',fontWeight:700,color:'var(--text-primary)',marginBottom:'8px'}}>Processing Failed</h2>
          <p style={{fontSize:'13px',color:'var(--text-muted)',marginBottom:'20px'}}>Backend error — check server logs</p>
          <button className="btn-amber" onClick={handleReset}>Try Again</button>
        </div>
      )}

      {/* STEPPER */}
      {(isUploading||isProcessing) && (
        <div className="glass" style={{padding:'24px',border:'1px solid var(--border-subtle)'}}>
          <div style={{display:'flex',alignItems:'center',justifyContent:'space-between',marginBottom:'6px'}}>
            <div style={{display:'flex',alignItems:'center',gap:'10px'}}>
              <div className="spinner" style={{width:'16px',height:'16px'}}/>
              <span style={{fontFamily:'var(--font-display)',fontSize:'14px',fontWeight:700,color:'var(--text-primary)'}}>
                {isUploading?'Uploading…':'AI Processing Pipeline'}
              </span>
            </div>
            {docStatus && <span style={{fontSize:'10px',fontFamily:'var(--font-mono)',padding:'2px 8px',borderRadius:'20px',background:'var(--amber-dim)',color:'var(--amber)',border:'1px solid var(--amber-border)'}}>{docStatus}</span>}
          </div>
          <div style={{fontSize:'11px',color:'var(--text-faint)',marginBottom:'20px'}}>
            {isUploading?'Sending to server…':backendDone?'Backend done — finalising…':'Processing · polling every 2.5s'}
          </div>
          {STAGES.map((stage,i)=>{
            const done=i<visualStep||allDone, active=i===visualStep&&!allDone&&isProcessing, pend=i>visualStep&&!allDone
            return (
              <div key={i} style={{display:'flex',alignItems:'flex-start',gap:'14px',padding:'12px 0',borderBottom:i<STAGES.length-1?'1px solid var(--border-subtle)':'none',opacity:pend?0.35:1,transition:'opacity 0.5s'}}>
                <div style={{flexShrink:0,marginTop:'2px'}}>
                  {done ? (
                    <div style={{width:'28px',height:'28px',borderRadius:'50%',display:'flex',alignItems:'center',justifyContent:'center',background:'var(--emerald-dim)',border:'1.5px solid var(--emerald)',boxShadow:'0 0 10px var(--emerald-glow)'}}>
                      <Check size={14} style={{color:'var(--emerald)'}}/>
                    </div>
                  ) : active ? (
                    <div style={{position:'relative',width:'28px',height:'28px'}}>
                      <div className="spinner" style={{position:'absolute',inset:0,borderRadius:'50%',borderTopColor:'var(--amber)',borderRightColor:'var(--amber)',borderWidth:'2px'}}/>
                      <div style={{position:'absolute',inset:'4px',borderRadius:'50%',background:'var(--amber-dim)',display:'flex',alignItems:'center',justifyContent:'center'}}>
                        <span style={{fontSize:'9px',fontWeight:800,color:'var(--amber)',fontFamily:'var(--font-display)'}}>{i+1}</span>
                      </div>
                    </div>
                  ) : (
                    <div style={{width:'28px',height:'28px',borderRadius:'50%',display:'flex',alignItems:'center',justifyContent:'center',border:'1.5px solid var(--border-default)'}}>
                      <span style={{fontSize:'11px',color:'var(--text-faint)'}}>{i+1}</span>
                    </div>
                  )}
                </div>
                <div style={{flex:1}}>
                  <div style={{fontSize:'13px',fontWeight:700,fontFamily:'var(--font-display)',color:done?'var(--text-secondary)':active?'var(--text-primary)':'var(--text-muted)',marginBottom:'2px',transition:'color 0.3s'}}>{stage.label}</div>
                  <div style={{fontSize:'11px',color:'var(--text-faint)'}}>{stage.desc}</div>
                </div>
                {done && <span style={{fontSize:'11px',color:'var(--emerald)',opacity:0.7,fontFamily:'var(--font-mono)',paddingTop:'2px'}}>✓</span>}
              </div>
            )
          })}
          {documentId && (
            <div style={{marginTop:'16px',paddingTop:'12px',borderTop:'1px solid var(--border-subtle)',fontSize:'11px',color:'var(--text-faint)'}}>
              ID: <span style={{fontFamily:'var(--font-mono)',color:'var(--text-muted)'}}>{documentId}</span>
            </div>
          )}
        </div>
      )}

      {/* IDLE */}
      {phase==='idle' && (
        <>
          <div className={`drop-zone${dragActive?' active':''}`}
            style={{marginBottom:'14px'}}
            onDragEnter={handleDrag} onDragLeave={handleDrag} onDragOver={handleDrag} onDrop={handleDrop}
            onClick={()=>!selectedFile&&document.getElementById('file-input').click()}>
            <input id="file-input" type="file" accept=".pdf" className="hidden" onChange={handleFileChange}/>
            {selectedFile ? (
              <div>
                <div style={{width:'56px',height:'56px',borderRadius:'16px',display:'flex',alignItems:'center',justifyContent:'center',margin:'0 auto 16px',background:'var(--amber-dim)',border:'1.5px solid var(--amber-border)',boxShadow:'0 0 24px var(--amber-glow)'}}>
                  <FileText size={24} style={{color:'var(--amber)'}}/>
                </div>
                <div style={{fontSize:'14px',fontWeight:600,color:'var(--text-primary)',fontFamily:'var(--font-mono)',marginBottom:'6px'}}>{selectedFile.name}</div>
                <div style={{fontSize:'13px',color:'var(--text-muted)',marginBottom:'14px'}}>{(selectedFile.size/1024/1024).toFixed(2)} MB · PDF ready</div>
                <button onClick={e=>{e.stopPropagation();setSelectedFile(null)}} style={{display:'inline-flex',alignItems:'center',gap:'5px',fontSize:'12px',color:'var(--rose)',background:'none',border:'none',cursor:'pointer'}}>
                  <X size={12}/> Remove file
                </button>
              </div>
            ) : (
              <div>
                <div style={{width:'64px',height:'64px',borderRadius:'18px',display:'flex',alignItems:'center',justifyContent:'center',margin:'0 auto 20px',background:dragActive?'var(--violet-dim)':'var(--bg-elevated)',border:`1.5px solid ${dragActive?'var(--violet-border)':'var(--border-strong)'}`,transition:'all 0.25s',boxShadow:dragActive?'0 0 30px var(--violet-glow)':'none'}}>
                  <UploadIcon size={28} style={{color:dragActive?'var(--violet)':'var(--text-muted)',transition:'color 0.25s'}}/>
                </div>
                <div style={{fontFamily:'var(--font-display)',fontSize:'16px',fontWeight:700,color:'var(--text-primary)',marginBottom:'8px'}}>
                  {dragActive?'Drop it here!':'Drop your PDF here, or click to browse'}
                </div>
                <div style={{fontSize:'13px',color:'var(--text-muted)',marginBottom:'20px'}}>Maximum file size: 50 MB · PDF only</div>
                <label className="btn-amber" style={{cursor:'pointer'}} onClick={e=>e.stopPropagation()}>
                  <UploadIcon size={15}/> Select PDF File
                  <input type="file" accept=".pdf" onChange={handleFileChange} className="hidden"/>
                </label>
              </div>
            )}
          </div>

          {selectedFile && (
            <button onClick={handleUpload} className="btn-amber"
              style={{width:'100%',justifyContent:'center',padding:'13px 24px',fontSize:'14px',borderRadius:'16px',marginBottom:'20px'}}>
              <Sparkles size={16}/> Upload and Process with AI <ArrowRight size={15}/>
            </button>
          )}

          {/* Pipeline preview */}
          <div className="glass" style={{padding:'20px',border:'1px solid var(--border-subtle)'}}>
            <div style={{display:'flex',alignItems:'center',gap:'8px',marginBottom:'16px'}}>
              <Zap size={14} style={{color:'var(--amber)'}}/>
              <span style={{fontFamily:'var(--font-display)',fontSize:'13px',fontWeight:700,color:'var(--text-secondary)'}}>AI Processing Pipeline</span>
            </div>
            <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:'8px'}}>
              {STAGES.map((stage,i)=>(
                <div key={i} style={{display:'flex',alignItems:'flex-start',gap:'10px',padding:'10px 12px',borderRadius:'12px',background:'var(--bg-elevated)',border:'1px solid var(--border-subtle)',transition:'all 0.15s'}}
                  onMouseEnter={e=>{e.currentTarget.style.borderColor='var(--border-default)';e.currentTarget.style.background='var(--bg-hover)'}}
                  onMouseLeave={e=>{e.currentTarget.style.borderColor='var(--border-subtle)';e.currentTarget.style.background='var(--bg-elevated)'}}>
                  <div style={{width:'22px',height:'22px',borderRadius:'7px',display:'flex',alignItems:'center',justifyContent:'center',flexShrink:0,background:'var(--amber-dim)',border:'1px solid var(--amber-border)'}}>
                    <span style={{fontSize:'10px',fontWeight:800,color:'var(--amber)',fontFamily:'var(--font-display)'}}>{i+1}</span>
                  </div>
                  <div>
                    <div style={{fontSize:'12px',fontWeight:700,color:'var(--text-primary)',fontFamily:'var(--font-display)',marginBottom:'2px'}}>{stage.label}</div>
                    <div style={{fontSize:'10px',color:'var(--text-faint)'}}>{stage.desc}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  )
}
