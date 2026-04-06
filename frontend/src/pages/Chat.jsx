import { useState, useRef, useEffect } from 'react'
import { useMutation } from 'react-query'
import { Send, Sparkles, FileText, Trash2, Zap } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import { chatAPI } from '../api/client'

const EXAMPLE_QUESTIONS = [
  { q:'What is the tumor microenvironment?', icon:Sparkles },
  { q:'Summarize the key findings',          icon:FileText },
  { q:'What methods were used?',             icon:Zap },
  { q:'Compare papers in my library',        icon:Sparkles },
]
const CHIPS = ['Summarize findings','What methods?','Key concepts?']

export default function Chat() {
  const [messages,setMessages] = useState(()=>{try{return JSON.parse(localStorage.getItem('chatMessages')||'[]')}catch{return[]}})
  const [input,setInput] = useState('')
  const [sessionId,setSessionId] = useState(null)
  const messagesEndRef = useRef(null)
  const textareaRef = useRef(null)

  useEffect(()=>{localStorage.setItem('chatMessages',JSON.stringify(messages))},[messages])
  useEffect(()=>{messagesEndRef.current?.scrollIntoView({behavior:'smooth'})},[messages])
  useEffect(()=>{
    if(textareaRef.current){
      textareaRef.current.style.height='48px'
      textareaRef.current.style.height=Math.min(textareaRef.current.scrollHeight,120)+'px'
    }
  },[input])

  const chatMutation = useMutation(
    ({question,conversationHistory,sid})=>chatAPI.ask(question,conversationHistory,sid),
    {
      onSuccess:(data)=>{
        const r=data.data
        if(r.session_id&&!sessionId) setSessionId(r.session_id)
        setMessages(prev=>[...prev,{role:'assistant',content:r.answer,citations:r.citations}])
      },
      onError:()=>setMessages(prev=>[...prev,{role:'assistant',content:'Sorry, I encountered an error. Please try again.',error:true}])
    }
  )

  const handleSubmit=(e)=>{
    e?.preventDefault()
    if(!input.trim()||chatMutation.isLoading) return
    const userMsg={role:'user',content:input}
    setMessages(prev=>[...prev,userMsg])
    const history=messages.map(m=>({role:m.role,content:m.content}))
    chatMutation.mutate({question:input,conversationHistory:history,sid:sessionId})
    setInput('')
  }

  const handleClear=()=>{
    if(window.confirm('Clear chat history?')){setMessages([]);setSessionId(null);localStorage.removeItem('chatMessages')}
  }

  return (
    <div style={{height:'calc(100vh - 64px)',display:'flex',flexDirection:'column'}}>
      {/* Header */}
      <div style={{display:'flex',alignItems:'center',justifyContent:'space-between',paddingBottom:'16px',marginBottom:'0',borderBottom:'1px solid var(--border-subtle)',flexShrink:0}}>
        <div style={{display:'flex',alignItems:'center',gap:'12px'}}>
          <div style={{width:'42px',height:'42px',borderRadius:'13px',display:'flex',alignItems:'center',justifyContent:'center',position:'relative',background:'linear-gradient(135deg,rgba(139,92,246,0.2),rgba(139,92,246,0.08))',border:'1px solid var(--violet-border)',boxShadow:'0 0 20px rgba(139,92,246,0.2)'}}>
            <Sparkles size={20} style={{color:'var(--violet-bright)'}}/>
          </div>
          <div>
            <h1 style={{fontFamily:'var(--font-display)',fontSize:'16px',fontWeight:800,color:'var(--text-primary)'}}>AI Research Assistant</h1>
            <div style={{display:'flex',alignItems:'center',gap:'6px',fontSize:'11px',color:'var(--text-faint)'}}>
              <div className="dot-pulse" style={{width:'5px',height:'5px',background:'var(--emerald)'}}/>
              RAG-powered · Gemini Pro
            </div>
          </div>
        </div>
        {messages.length>0 && (
          <button className="btn-ghost" onClick={handleClear}><Trash2 size={13}/> Clear History</button>
        )}
      </div>

      {/* Messages */}
      <div style={{flex:1,overflowY:'auto',padding:'20px 2px',display:'flex',flexDirection:'column'}}>
        {messages.length===0 ? (
          <div style={{maxWidth:'460px',margin:'60px auto 0',textAlign:'center'}}>
            <div style={{position:'relative',width:'76px',height:'76px',margin:'0 auto 20px'}}>
              <div style={{position:'absolute',inset:0,borderRadius:'50%',background:'var(--amber-dim)',animation:'glow-pulse 2.5s ease infinite'}}/>
              <div style={{position:'absolute',inset:'8px',borderRadius:'50%',background:'var(--amber-dim)',border:'1px solid var(--amber-border)',display:'flex',alignItems:'center',justifyContent:'center'}}>
                <Sparkles size={28} style={{color:'var(--amber)'}}/>
              </div>
            </div>
            <h2 style={{fontFamily:'var(--font-display)',fontSize:'20px',fontWeight:800,color:'var(--text-primary)',marginBottom:'8px'}}>Hello, Researcher</h2>
            <p style={{fontSize:'13px',color:'var(--text-muted)',marginBottom:'28px',lineHeight:'1.6'}}>
              Ask me anything about your uploaded papers.<br/>I'll answer using your documents as context.
            </p>
            <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:'10px'}}>
              {EXAMPLE_QUESTIONS.map(({q,icon:Icon},i)=>(
                <button key={i} onClick={()=>setInput(q)} style={{
                  padding:'14px',borderRadius:'14px',textAlign:'left',
                  background:'var(--bg-surface)',border:'1px solid var(--border-subtle)',
                  cursor:'pointer',transition:'all 0.2s ease',fontFamily:'var(--font-body)',
                }}
                onMouseEnter={e=>{e.currentTarget.style.borderColor='var(--amber-border)';e.currentTarget.style.background='var(--bg-elevated)';e.currentTarget.style.transform='translateY(-2px)';e.currentTarget.style.boxShadow='0 4px 16px rgba(0,0,0,0.3)'}}
                onMouseLeave={e=>{e.currentTarget.style.borderColor='var(--border-subtle)';e.currentTarget.style.background='var(--bg-surface)';e.currentTarget.style.transform='none';e.currentTarget.style.boxShadow='none'}}>
                  <Icon size={16} style={{color:'var(--amber)',marginBottom:'8px'}}/>
                  <div style={{fontSize:'12px',color:'var(--text-secondary)',lineHeight:'1.5'}}>{q}</div>
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div style={{display:'flex',flexDirection:'column',gap:'16px'}}>
            {messages.map((msg,i)=>(
              <div key={i} style={{display:'flex',gap:'10px',justifyContent:msg.role==='user'?'flex-end':'flex-start',animation:'slide-up 0.22s ease'}}>
                {msg.role==='assistant' && (
                  <div style={{width:'32px',height:'32px',borderRadius:'10px',display:'flex',alignItems:'center',justifyContent:'center',flexShrink:0,marginTop:'2px',background:'var(--amber-dim)',border:'1px solid var(--amber-border)'}}>
                    <Sparkles size={14} style={{color:'var(--amber)'}}/>
                  </div>
                )}
                <div className={msg.role==='user'?'bubble-user':'bubble-ai'}>
                  {msg.role==='assistant' ? (
                    <>
                      <ReactMarkdown components={{
                        p:({node,...p})=><p style={{color:'var(--text-secondary)',marginBottom:'0.7em'}} {...p}/>,
                        strong:({node,...p})=><strong style={{color:'var(--text-primary)',fontWeight:600}} {...p}/>,
                        code:({node,...p})=><code style={{background:'var(--bg-elevated)',padding:'2px 6px',borderRadius:'4px',fontSize:'0.88em',fontFamily:'var(--font-mono)',color:'var(--amber)'}} {...p}/>,
                        li:({node,...p})=><li style={{color:'var(--text-secondary)',marginBottom:'4px'}} {...p}/>,
                      }}>{msg.content}</ReactMarkdown>
                      {msg.citations?.length>0 && (
                        <div style={{marginTop:'12px',paddingTop:'12px',borderTop:'1px solid var(--border-subtle)'}}>
                          <div style={{fontSize:'10px',textTransform:'uppercase',letterSpacing:'0.8px',color:'var(--text-faint)',fontWeight:700,fontFamily:'var(--font-display)',marginBottom:'8px'}}>Sources</div>
                          <div style={{display:'flex',flexWrap:'wrap',gap:'6px'}}>
                            {msg.citations.map((c,idx)=>(
                              <div key={idx} style={{display:'flex',alignItems:'center',gap:'6px',padding:'4px 10px',borderRadius:'8px',background:'var(--bg-elevated)',border:'1px solid var(--border-subtle)',fontSize:'11px',color:'var(--text-secondary)',cursor:'pointer',transition:'all 0.15s'}}
                                onMouseEnter={e=>{e.currentTarget.style.borderColor='var(--amber-border)';e.currentTarget.style.color='var(--amber)'}}
                                onMouseLeave={e=>{e.currentTarget.style.borderColor='var(--border-subtle)';e.currentTarget.style.color='var(--text-secondary)'}}>
                                <FileText size={11} style={{color:'var(--amber)',flexShrink:0}}/>
                                <span className="mono">{c.title||`Source ${idx+1}`}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </>
                  ) : <p style={{margin:0}}>{msg.content}</p>}
                </div>
              </div>
            ))}
            {chatMutation.isLoading && (
              <div style={{display:'flex',gap:'10px',animation:'slide-up 0.2s ease'}}>
                <div style={{width:'32px',height:'32px',borderRadius:'10px',display:'flex',alignItems:'center',justifyContent:'center',flexShrink:0,marginTop:'2px',background:'var(--amber-dim)',border:'1px solid var(--amber-border)'}}>
                  <Sparkles size={14} style={{color:'var(--amber)'}}/>
                </div>
                <div className="bubble-ai" style={{padding:'14px 18px'}}>
                  <div style={{display:'flex',gap:'5px',alignItems:'center'}}>
                    {[0,0.15,0.3].map((d,i)=>(
                      <div key={i} style={{width:'7px',height:'7px',borderRadius:'50%',background:'var(--amber)',animation:`glow-pulse 0.7s ease-in-out ${d}s infinite`}}/>
                    ))}
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef}/>
          </div>
        )}
      </div>

      {/* Input */}
      <div style={{flexShrink:0,paddingTop:'12px',borderTop:'1px solid var(--border-subtle)'}}>
        {!input && messages.length>0 && (
          <div style={{display:'flex',gap:'8px',marginBottom:'10px',flexWrap:'wrap'}}>
            {CHIPS.map(chip=>(
              <button key={chip} onClick={()=>setInput(chip)} style={{
                fontSize:'11px',padding:'5px 12px',borderRadius:'20px',
                background:'var(--bg-elevated)',border:'1px solid var(--border-subtle)',
                color:'var(--text-secondary)',cursor:'pointer',fontFamily:'var(--font-body)',transition:'all 0.15s'
              }}
              onMouseEnter={e=>{e.currentTarget.style.borderColor='var(--amber-border)';e.currentTarget.style.color='var(--amber)';e.currentTarget.style.background='var(--amber-dim)'}}
              onMouseLeave={e=>{e.currentTarget.style.borderColor='var(--border-subtle)';e.currentTarget.style.color='var(--text-secondary)';e.currentTarget.style.background='var(--bg-elevated)'}}>
                {chip}
              </button>
            ))}
          </div>
        )}
        <form onSubmit={handleSubmit} style={{display:'flex',alignItems:'flex-end',gap:'10px'}}>
          <textarea ref={textareaRef} value={input}
            onChange={e=>setInput(e.target.value)}
            onKeyDown={e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();handleSubmit()}}}
            placeholder="Ask a question about your research papers..."
            disabled={chatMutation.isLoading}
            style={{flex:1,resize:'none',background:'var(--bg-surface)',border:'1px solid var(--border-default)',borderRadius:'14px',padding:'12px 16px',fontSize:'14px',color:'var(--text-primary)',outline:'none',minHeight:'48px',maxHeight:'120px',fontFamily:'var(--font-body)',lineHeight:'1.5',transition:'all 0.2s'}}
            onFocus={e=>{e.currentTarget.style.borderColor='var(--violet)';e.currentTarget.style.boxShadow='0 0 0 3px var(--violet-dim)'}}
            onBlur={e=>{e.currentTarget.style.borderColor='var(--border-default)';e.currentTarget.style.boxShadow='none'}}
          />
          <button type="submit" className="btn-amber" disabled={!input.trim()||chatMutation.isLoading}
            style={{height:'48px',padding:'0 18px',flexShrink:0}}>
            {chatMutation.isLoading
              ? <div className="spinner" style={{width:'16px',height:'16px',borderTopColor:'#06060E',borderColor:'rgba(0,0,0,0.2)'}}/>
              : <Send size={16}/>}
          </button>
        </form>
        <div style={{fontSize:'10px',color:'var(--text-faint)',textAlign:'center',marginTop:'8px',fontFamily:'var(--font-mono)'}}>
          Enter to send · Shift+Enter for new line
        </div>
      </div>
    </div>
  )
}
