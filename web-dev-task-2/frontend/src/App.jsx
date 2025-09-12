import React, {useEffect, useState} from 'react';
import { startOAuth, listRepos, generateReadme } from './api';
import { marked } from 'marked';

function LoginButton() {
  return (
    <button className="gh-btn" onClick={() => startOAuth()}>
      <img src="/src/github-logo.svg" alt="github" style={{height:18, marginRight:8}} />
      Login with GitHub
    </button>
  );
}

export default function App(){
  const [repos, setRepos] = useState([]);
  const [selected, setSelected] = useState(null);
  const [preview, setPreview] = useState('');
  const [loading, setLoading] = useState(false);

  useEffect(()=>{
    async function load(){
      try{
        const r = await listRepos();
        setRepos(r.data);
      }catch(e){
        // not logged in or error
      }
    }
    load();
  },[]);

  async function onGenerate(writeToRepo=false){
    if(!selected) return alert('Select a repo');
    setLoading(true);
    try{
      const resp = await generateReadme({ owner: selected.owner, repo: selected.name, writeToRepo });
      setPreview(resp.data.generated || '');
      if(resp.data.wroteToRepo) alert('README saved to repo');
    }catch(e){
      alert(e?.response?.data?.error || 'Generation failed');
    }
    setLoading(false);
  }

  return (
    <div className="page">
      <div className="card">
        <h1>Private GitHub README generator</h1>
        <LoginButton />

        <div style={{marginTop:20}}>
          <label>Choose repo</label>
          <select onChange={(e)=>{
            const idx = e.target.value;
            setSelected(idx !== '' ? repos[idx] : null);
            setPreview('');
          }}>
            <option value="">-- select repo --</option>
            {repos.map((r,i)=> (
              <option value={i} key={r.id}>{r.full_name}{r.private? ' (private)':''}</option>
            ))}
          </select>
        </div>

        <div style={{marginTop:12}}>
          <button className="action" onClick={()=>onGenerate(false)} disabled={loading}>Generate (preview)</button>
          <button className="action" onClick={()=>onGenerate(true)} disabled={loading}>Generate & Save</button>
        </div>

        {preview && (
          <div className="preview">
            <div dangerouslySetInnerHTML={{__html: marked.parse(preview)}} />
          </div>
        )}

      </div>
    </div>
  );
}
