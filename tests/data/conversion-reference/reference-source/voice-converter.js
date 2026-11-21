const SR=22050,N=1024,H=256,NB=N/2+1,HOP=320,D=12,NM=34,LIF=36;
const $=id=>document.getElementById(id),tick=()=>new Promise(r=>setTimeout(r));
const S={src:null,tgt:null};let ctx,node,out=null,busy=false;
const ac=()=>ctx||(ctx=new(window.AudioContext||window.webkitAudioContext)());

/* ---------- audio io ---------- */
async function decode(blob){
  const buf=await ac().decodeAudioData(await blob.arrayBuffer());
  const off=new OfflineAudioContext(1,Math.ceil(buf.duration*SR),SR),s=off.createBufferSource();
  s.buffer=buf;s.connect(off.destination);s.start();
  return (await off.startRendering()).getChannelData(0).slice();
}
function wav(d){const b=new DataView(new ArrayBuffer(44+d.length*2)),w=(o,s)=>[...s].forEach((c,i)=>b.setUint8(o+i,c.charCodeAt(0)));
  w(0,'RIFF');b.setUint32(4,36+d.length*2,true);w(8,'WAVEfmt ');b.setUint32(16,16,true);b.setUint16(20,1,true);b.setUint16(22,1,true);
  b.setUint32(24,SR,true);b.setUint32(28,SR*2,true);b.setUint16(32,2,true);b.setUint16(34,16,true);w(36,'data');b.setUint32(40,d.length*2,true);
  for(let i=0;i<d.length;i++)b.setInt16(44+i*2,Math.max(-1,Math.min(1,d[i]))*32767,true);return new Blob([b],{type:'audio/wav'})}
function play(d){stop();const a=ac(),b=a.createBuffer(1,d.length,SR);b.copyToChannel(d,0);node=a.createBufferSource();node.buffer=b;node.connect(a.destination);node.start();a.resume()}
function stop(){if(node){try{node.stop()}catch(e){}node=null}}
function wave(cv,d,col){const c=cv.getContext('2d'),w=cv.width,h=cv.height;c.clearRect(0,0,w,h);c.fillStyle=col;
  const step=Math.max(1,Math.floor(d.length/w));for(let x=0;x<w;x++){let m=0;for(let i=x*step,e=Math.min(d.length,i+step);i<e;i++){const v=Math.abs(d[i]);if(v>m)m=v}
  const a=Math.min(1,m*1.6)*h/2;c.fillRect(x,h/2-a,1,Math.max(1,a*2))}}

/* ---------- FFT ---------- */
const cosT=new Float32Array(N/2),sinT=new Float32Array(N/2),rev=new Uint16Array(N),win=new Float32Array(N);
for(let i=0;i<N/2;i++){cosT[i]=Math.cos(2*Math.PI*i/N);sinT[i]=-Math.sin(2*Math.PI*i/N)}
for(let i=0;i<N;i++){let r=0;for(let j=0;j<10;j++)r=(r<<1)|((i>>j)&1);rev[i]=r;win[i]=.5-.5*Math.cos(2*Math.PI*i/N)}
function fft(re,im,inv){
  for(let i=0;i<N;i++){const j=rev[i];if(j>i){let t=re[i];re[i]=re[j];re[j]=t;t=im[i];im[i]=im[j];im[j]=t}}
  for(let len=2;len<=N;len<<=1){const h=len>>1,st=N/len;
    for(let i=0;i<N;i+=len)for(let k=0;k<h;k++){const c=cosT[k*st],s=inv?-sinT[k*st]:sinT[k*st],a=i+k,b=a+h,
      tr=re[b]*c-im[b]*s,ti=re[b]*s+im[b]*c;re[b]=re[a]-tr;im[b]=im[a]-ti;re[a]+=tr;im[a]+=ti}}
  if(inv)for(let i=0;i<N;i++){re[i]/=N;im[i]/=N}
}

/* ---------- pitch tracking (normalised autocorrelation) ---------- */
async function trackPitch(x,fmin,fmax,pg){
  const W=800,lmin=Math.floor(SR/fmax),lmax=Math.ceil(SR/fmin),nf=Math.floor(x.length/HOP)+1,
    f0=new Float32Array(nf),vo=new Uint8Array(nf),r=new Float32Array(lmax+2);
  for(let f=0;f<nf;f++){
    if(f%60==0){pg&&pg(f/nf);await tick()}
    const s=f*HOP-(W>>1);if(s<0||s+W+lmax+2>=x.length)continue;
    let e0=0,el=0;for(let i=0;i<W;i++){e0+=x[s+i]*x[s+i];el+=x[s+lmin+i]*x[s+lmin+i]}
    if(e0/W<2.5e-5)continue;
    let best=0;
    for(let l=lmin;l<=lmax;l++){let c=0;for(let i=0;i<W;i++)c+=x[s+i]*x[s+i+l];
      r[l]=c/Math.sqrt(e0*el+1e-9);if(r[l]>best)best=r[l];el+=x[s+l+W]*x[s+l+W]-x[s+l]*x[s+l]}
    if(best<.45)continue;let bl=0;
    for(let l=lmin+1;l<lmax;l++)if(r[l]>=.85*best&&r[l]>=r[l-1]&&r[l]>=r[l+1]){bl=l;break}
    if(!bl)continue;
    const a=r[bl-1],b=r[bl],c=r[bl+1],dn=a-2*b+c,d=dn?.5*(a-c)/dn:0;
    f0[f]=SR/(bl+Math.max(-1,Math.min(1,d)));vo[f]=1;
  }
  // median smoothing; drops isolated voiced blips
  const o=new Float32Array(nf),ov=new Uint8Array(nf);
  for(let i=0;i<nf;i++){if(!vo[i])continue;const a=[];for(let j=i-2;j<=i+2;j++)if(j>=0&&j<nf&&vo[j])a.push(f0[j]);
    if(a.length<3)continue;a.sort((p,q)=>p-q);o[i]=a[a.length>>1];ov[i]=1}
  return[o,ov];
}
const median=(f,v)=>{const a=[];for(let i=0;i<f.length;i++)if(v[i])a.push(f[i]);a.sort((p,q)=>p-q);return a.length?a[a.length>>1]:0};

/* ---------- TD-PSOLA pitch shift ---------- */
function makeMarks(x,f0,vo){
  const U=110,m=[],fr=t=>Math.min(f0.length-1,Math.round(t/HOP));let pos=0,prev=false;
  const peak=(c,r)=>{let b=Math.max(0,Math.round(c-r)),bv=-9;for(let i=b,e=Math.min(x.length-1,Math.round(c+r));i<=e;i++)if(x[i]>bv){bv=x[i];b=i}return b};
  while(pos<x.length){const fi=fr(pos);
    if(vo[fi]){const T=SR/f0[fi],c=peak(pos,prev?T*.3:T*.5),l=m.length?m[m.length-1].p:-1e9;
      if(c>l+T*.5){m.push({p:c,T,v:1,f:f0[fi]});pos=c+T}else pos+=T*.5;prev=true}
    else{m.push({p:Math.round(pos),T:U,v:0,f:0});pos+=U;prev=false}}
  return m;
}
function psola(x,m,ratio){
  const o=new Float32Array(x.length+4096),ws=new Float32Array(o.length);let ts=0,mi=0;
  while(ts<x.length&&m.length){
    while(mi<m.length-1&&Math.abs(m[mi+1].p-ts)<=Math.abs(m[mi].p-ts))mi++;
    const k=m[mi],L=Math.round(k.T),g=Math.round(ts);
    for(let i=-L;i<=L;i++){const a=k.p+i,b=g+i;if(a<0||a>=x.length||b<0||b>=o.length)continue;
      const w=.5+.5*Math.cos(Math.PI*i/L);o[b]+=x[a]*w;ws[b]+=w}
    ts+=k.v?SR/Math.min(800,Math.max(50,k.f*ratio)):k.T;
  }
  for(let i=0;i<x.length;i++)o[i]/=Math.max(ws[i],.35);
  return o.subarray(0,x.length);
}

/* ---------- spectral envelope, features ---------- */
const CEP=new Float32Array(LIF+1),Re=new Float32Array(N),Im=new Float32Array(N),Cr=new Float32Array(N),Ci=new Float32Array(N);
const mel=f=>2595*Math.log10(1+f/700),imel=m=>700*(10**(m/2595)-1),edges=[];
for(let i=0;i<NM+2;i++){let b=Math.round(imel(mel(100)+(mel(8000)-mel(100))*i/(NM+1))/(SR/N));if(i&&b<=edges[i-1])b=edges[i-1]+1;edges.push(b)}
const dct=[];for(let j=1;j<=D;j++){const r=new Float32Array(NM);for(let b=0;b<NM;b++)r[b]=Math.cos(Math.PI*j*(b+.5)/NM);dct.push(r)}
function analyze(x,pos,le){ // fills Re/Im with spectrum and le with smooth log envelope; returns frame rms
  let e=0;for(let i=0;i<N;i++){const p=pos+i,v=(p>=0&&p<x.length?x[p]:0)*win[i];Re[i]=v;Im[i]=0;e+=v*v}
  fft(Re,Im,0);
  for(let i=0;i<N;i++){Cr[i]=Math.log(Math.hypot(Re[i],Im[i])+1e-5);Ci[i]=0}
  fft(Cr,Ci,1);
  for(let i=LIF+1;i<N-LIF;i++)Cr[i]=0;Ci.fill(0);for(let q=0;q<=LIF;q++)CEP[q]=Cr[q];
  fft(Cr,Ci,0);for(let k=0;k<NB;k++)le[k]=Cr[k];
  return Math.sqrt(e/(N*.375));
}
function feat(le,o,off){const mb=new Float32Array(NM);
  for(let b=0;b<NM;b++){let s=0,lo=edges[b],hi=edges[b+2];for(let k=lo;k<=hi;k++)s+=le[k];mb[b]=s/(hi-lo+1)}
  for(let j=0;j<D;j++){let s=0;for(let b=0;b<NM;b++)s+=mb[b]*dct[j][b];o[off+j]=s}}
function cmn(ft,ok){const m=new Float32Array(D);let n=0;
  for(let f=0;f<ok.length;f++)if(ok[f]){n++;for(let j=0;j<D;j++)m[j]+=ft[f*D+j]}
  for(let j=0;j<D;j++)m[j]/=Math.max(n,1);
  for(let f=0;f<ok.length;f++)for(let j=0;j<D;j++)ft[f*D+j]-=m[j]}
const SIL=.004;

/* ---------- retrieval database from the target voice ---------- */
async function buildDB(t,fmin,fmax,pg){
  const[f0,vo]=await trackPitch(t,fmin,fmax,p=>pg(p*.5)),le0=new Float32Array(NB),les=[],fts=[],vs=[],cs=[],cps=[];
  const nf=Math.floor((t.length-N)/512);
  for(let f=0;f<nf;f++){if(f%80==0){pg(.5+.5*f/nf);await tick()}
    if(analyze(t,f*512,le0)<SIL)continue;cs.push(f*512+N/2);
    const row=new Float32Array(D);feat(le0,row,0);fts.push(row);les.push(le0.slice());cps.push(CEP.slice(1));
    vs.push(vo[Math.min(vo.length-1,Math.round((f*512+N/2)/HOP))])}
  const n=les.length,ft=new Float32Array(n*D),ok=new Uint8Array(n).fill(1),idx=[[],[]];
  fts.forEach((r,i)=>ft.set(r,i*D));cmn(ft,ok);vs.forEach((v,i)=>idx[v].push(i));
  const M=makeMarks(t,f0,vo),mk=new Int32Array(n);let m=0;
  for(let j=0;j<n;j++){while(m<M.length-1&&Math.abs(M[m+1].p-cs[j])<=Math.abs(M[m].p-cs[j]))m++;mk[j]=m}
  const cep=new Float32Array(n*LIF),cmu=new Float64Array(LIF),cvar=new Float64Array(LIF);
  cps.forEach((r,i)=>{cep.set(r,i*LIF);for(let q=0;q<LIF;q++)cmu[q]+=r[q]/n});
  cps.forEach(r=>{for(let q=0;q<LIF;q++)cvar[q]+=(r[q]-cmu[q])**2/n});
  const fm=new Int32Array(M.length);let jj=0;
  for(let q=0;q<M.length;q++){while(jj<n-1&&Math.abs(cs[jj+1]-M[q].p)<=Math.abs(cs[jj]-M[q].p))jj++;fm[q]=jj}
  return{le:les,ft,idx,med:median(f0,vo),n,t,marks:M,mk,fm,cep,cvar};
}
async function knn(sf,sv,sa,db,K,pg){
  const nf=sv.length,res=new Int32Array(nf*K).fill(-1),bd=new Float32Array(K),bi=new Int32Array(K);
  for(let f=0;f<nf;f++){
    if(f%48==0){pg(f/nf);await tick()}
    if(!sa[f])continue;
    const cand=db.idx[sv[f]].length>20?db.idx[sv[f]]:db.idx[1-sv[f]];bd.fill(1e30);bi.fill(-1);
    for(let c=0;c<cand.length;c++){const j=cand[c];let d=0;
      for(let q=0;q<D;q++){const t=sf[f*D+q]-db.ft[j*D+q];d+=t*t}
      if(d<bd[K-1]){let p=K-1;while(p>0&&bd[p-1]>d){bd[p]=bd[p-1];bi[p]=bi[p-1];p--}bd[p]=d;bi[p]=j}}
    res.set(bi,f*K);
  }
  return res;
}

/* ---------- grain retrieval: rebuild speech from the target's own waveform ---------- */
function fillGaps(f0,vo){ // bridge short unvoiced gaps and smooth the contour so pitch glides instead of stepping
  const n=f0.length,o=f0.slice(),v=vo.slice();
  for(let i=0;i<n;){if(v[i]){i++;continue}let j=i;while(j<n&&!v[j])j++;
    if(i>0&&j<n&&j-i<=3)for(let k=i;k<j;k++){const a=(k-i+1)/(j-i+1);o[k]=Math.exp(Math.log(f0[i-1])*(1-a)+Math.log(f0[j])*a);v[k]=1}i=j}
  const s=o.slice();for(let i=1;i<n-1;i++)if(v[i]&&v[i-1]&&v[i+1])o[i]=Math.exp(.25*Math.log(s[i-1])+.5*Math.log(s[i])+.25*Math.log(s[i+1]));
  return[o,v];
}
function grainSynth(len,f0,vo,nb,K,db,ratio,rs,nfr,sf){
  const o=new Float32Array(len+4096),ws=new Float32Array(o.length),t=db.t,M=db.marks;let ts=0,prev=-9,g0=1;
  const fat=u=>{const i=Math.min(f0.length-1,Math.floor(u)),j=Math.min(f0.length-1,i+1),a=u-i;
    return vo[i]&&vo[j]?Math.exp(Math.log(f0[i])*(1-a)+Math.log(f0[j])*a):(vo[i]?f0[i]:f0[j])};
  const dd=(j,f)=>{let d=0;for(let q=0;q<D;q++){const x=sf[f*D+q]-db.ft[j*D+q];d+=x*x}return d};
  while(ts<len){
    const u=ts/HOP,f=Math.min(nfr-1,Math.round(ts/H)),v=vo[Math.min(vo.length-1,Math.round(u))],
      Tn=v?SR/Math.min(800,Math.max(50,fat(u)*ratio)):110,pen=T=>v?1+2*Math.abs(Math.log(T/Tn)):1;
    let best=-1,bs=1e30;
    for(let q=0;q<K;q++){const j=nb[f*K+q];if(j<0)continue;const m=db.mk[j],k=M[m],
      sc=(dd(j,f)+.05)*pen(k.T)*(k.v!==v?2:1);if(sc<bs){bs=sc;best=m}}
    // keep playing the target's own audio when the next grain still fits: natural phase and texture, fewer joins
    if(best>=0&&prev>=0&&prev+1<M.length&&M[prev+1].v===v){
      const sc=(dd(db.fm[prev+1],f)+.05)*pen(M[prev+1].T)*.7;if(sc<bs){bs=sc;best=prev+1}}
    if(best>=0){
      const k=M[best],L=Math.round(k.T),g=Math.round(ts);let e=0,c=0;
      for(let i=-L;i<=L;i++){const a=k.p+i;if(a>=0&&a<t.length){e+=t[a]*t[a];c++}}
      let gn=Math.max(.25,Math.min(4,rs[f]/(Math.sqrt(e/Math.max(c,1))+1e-6)));gn=g0=.7*g0+.3*gn;
      let d=0;
      if(v){let bc=-1e30;const dm=Math.round(k.T/3);
        for(let sh=-dm;sh<=dm;sh++){let cc=0,ee=1e-9;
          for(let i=-L;i<0;i++){const a=k.p+sh+i,b=g+i;if(a<0||a>=t.length||b<0)continue;cc+=o[b]*t[a];ee+=t[a]*t[a]}
          cc/=Math.sqrt(ee);if(cc>bc){bc=cc;d=sh}}}
      for(let i=-L;i<=L;i++){const a=k.p+d+i,b=g+i;if(a<0||a>=t.length||b<0||b>=o.length)continue;
        const w=.5+.5*Math.cos(Math.PI*i/L);o[b]+=t[a]*w*gn;ws[b]+=w}
      prev=best;
    }
    ts+=Tn;
  }
  for(let i=0;i<len;i++)o[i]/=Math.max(ws[i],.35);
  return o.subarray(0,len);
}

/* ---------- GMM voice model + harmonic/noise vocoder ---------- */
function inv(A,n){const I=new Float64Array(n*n);for(let i=0;i<n;i++)I[i*n+i]=1;
  for(let c=0;c<n;c++){let p=c;for(let r=c+1;r<n;r++)if(Math.abs(A[r*n+c])>Math.abs(A[p*n+c]))p=r;
    if(p!==c)for(let k=0;k<n;k++){let t=A[c*n+k];A[c*n+k]=A[p*n+k];A[p*n+k]=t;t=I[c*n+k];I[c*n+k]=I[p*n+k];I[p*n+k]=t}
    const d=A[c*n+c]||1e-9;for(let k=0;k<n;k++){A[c*n+k]/=d;I[c*n+k]/=d}
    for(let r=0;r<n;r++){if(r===c)continue;const f=A[r*n+c];if(!f)continue;for(let k=0;k<n;k++){A[r*n+k]-=f*A[c*n+k];I[r*n+k]-=f*I[c*n+k]}}}
  return I}
async function vocode(len,f0,vo,nb,K,db,ratio,rs,sf,sa,nfr,form,vg){
  const C=LIF,out=new Float32Array(len+N),act=[];
  for(let f=0;f<nfr;f++)if(sa[f]&&nb[f*K]>=0)act.push(f);
  const n=act.length;if(n<30)return out;
  // 1) training pairs (non-parallel): source frame -> mean cepstrum of its K closest target frames
  const X=new Float32Array(n*D),Y=new Float32Array(n*C);
  act.forEach((f,i)=>{for(let q=0;q<D;q++)X[i*D+q]=sf[f*D+q];let c=0;
    for(let k=0;k<K;k++){const j=nb[f*K+k];if(j<0)continue;c++;for(let r=0;r<C;r++)Y[i*C+r]+=db.cep[j*C+r]}
    for(let r=0;r<C;r++)Y[i*C+r]/=Math.max(c,1)});
  // 2) fit a Gaussian mixture on the source features with EM
  const M=Math.max(2,Math.min(8,n/60|0)),mu=new Float64Array(M*D),vr=new Float64Array(M*D),pi=new Float64Array(M).fill(1/M),
    G=new Float64Array(n*M),gv=new Float64Array(D),gm=new Float64Array(D),g=new Float64Array(M);
  for(let i=0;i<n;i++)for(let q=0;q<D;q++)gm[q]+=X[i*D+q]/n;
  for(let i=0;i<n;i++)for(let q=0;q<D;q++)gv[q]+=(X[i*D+q]-gm[q])**2/n;
  for(let m=0;m<M;m++){const i=Math.floor((m+.5)*n/M);for(let q=0;q<D;q++){mu[m*D+q]=X[i*D+q];vr[m*D+q]=gv[q]}}
  const post=(x,o,gg)=>{let mx=-1e300;const lp=new Float64Array(M);
    for(let m=0;m<M;m++){let l=Math.log(pi[m]+1e-12);for(let q=0;q<D;q++){const t=x[o+q]-mu[m*D+q],v=vr[m*D+q];l-=.5*(Math.log(v)+t*t/v)}lp[m]=l;if(l>mx)mx=l}
    let z=0;for(let m=0;m<M;m++){lp[m]=Math.exp(lp[m]-mx);z+=lp[m]}for(let m=0;m<M;m++)gg[m]=lp[m]/z};
  const E=()=>{for(let i=0;i<n;i++){post(X,i*D,g);for(let m=0;m<M;m++)G[i*M+m]=g[m]}};
  for(let it=0;it<15;it++){E();
    for(let m=0;m<M;m++){let w=1e-9;const s1=new Float64Array(D),s2=new Float64Array(D);
      for(let i=0;i<n;i++){const t=G[i*M+m];w+=t;for(let q=0;q<D;q++){const x=X[i*D+q];s1[q]+=t*x;s2[q]+=t*x*x}}
      for(let q=0;q<D;q++){const a=s1[q]/w;mu[m*D+q]=a;vr[m*D+q]=Math.max(s2[q]/w-a*a,gv[q]*.02)}pi[m]=w/n}
    vg(it/15*.3,'Training voice model…');await tick()}
  E();
  // 3) per-component linear regression source features -> target cepstrum (ridge)
  const MX=[],MY=[],AA=[];
  for(let m=0;m<M;m++){let w=1e-9;const mx=new Float64Array(D),my=new Float64Array(C),dx=new Float64Array(D),dy=new Float64Array(C),
      Sxx=new Float64Array(D*D),Syx=new Float64Array(C*D);
    for(let i=0;i<n;i++){const t=G[i*M+m];w+=t;for(let q=0;q<D;q++)mx[q]+=t*X[i*D+q];for(let r=0;r<C;r++)my[r]+=t*Y[i*C+r]}
    for(let q=0;q<D;q++)mx[q]/=w;for(let r=0;r<C;r++)my[r]/=w;
    for(let i=0;i<n;i++){const t=G[i*M+m];if(t<1e-3)continue;
      for(let q=0;q<D;q++)dx[q]=X[i*D+q]-mx[q];for(let r=0;r<C;r++)dy[r]=Y[i*C+r]-my[r];
      for(let a=0;a<D;a++)for(let b=0;b<D;b++)Sxx[a*D+b]+=t*dx[a]*dx[b];
      for(let r=0;r<C;r++)for(let b=0;b<D;b++)Syx[r*D+b]+=t*dy[r]*dx[b]}
    let tr=0;for(let q=0;q<D;q++)tr+=Sxx[q*D+q];for(let q=0;q<D;q++)Sxx[q*D+q]+=.15*tr/D+1e-6;
    const iv=inv(Sxx,D),A=new Float64Array(C*D);
    for(let r=0;r<C;r++)for(let b=0;b<D;b++){let z=0;for(let c=0;c<D;c++)z+=Syx[r*D+c]*iv[c*D+b];A[r*D+b]=z}
    MX.push(mx);MY.push(my);AA.push(A)}
  // 4) convert every frame, smooth in time, restore variance (global-variance postfilter)
  vg(.35,'Converting spectra…');await tick();
  let P=new Float32Array(nfr*C);
  for(let f=0;f<nfr;f++){if(!sa[f])continue;post(sf,f*D,g);
    for(let m=0;m<M;m++)for(let r=0;r<C;r++){let v=MY[m][r];for(let b=0;b<D;b++)v+=AA[m][r*D+b]*(sf[f*D+b]-MX[m][b]);P[f*C+r]+=g[m]*v}}
  const P2=P.slice(),wt=[.1,.2,.4,.2,.1];
  for(let f=2;f<nfr-2;f++){if(!sa[f])continue;for(let r=0;r<C;r++){let v=0,z=0;for(let d=-2;d<=2;d++)if(sa[f+d]){v+=wt[d+2]*P[(f+d)*C+r];z+=wt[d+2]}P2[f*C+r]=v/z}}
  P=P2;
  const mp=new Float64Array(C),vp=new Float64Array(C);
  for(const f of act)for(let r=0;r<C;r++)mp[r]+=P[f*C+r]/n;
  for(const f of act)for(let r=0;r<C;r++)vp[r]+=(P[f*C+r]-mp[r])**2/n;
  const sc=new Float64Array(C);for(let r=0;r<C;r++)sc[r]=1+.7*(Math.max(1,Math.min(2.2,Math.sqrt(db.cvar[r]/(vp[r]+1e-12))))-1);
  for(const f of act)for(let r=0;r<C;r++)P[f*C+r]=mp[r]+(P[f*C+r]-mp[r])*sc[r];
  // 5) synthesis: harmonics at the (smoothed) source pitch + envelope-shaped noise
  const COS=[];for(let q=1;q<=C;q++){const r=new Float32Array(NB);for(let k=0;k<NB;k++)r[k]=2*Math.cos(2*Math.PI*q*k/N);COS.push(r)}
  const NH=220,Am=new Float32Array(nfr*NH),F0=new Float32Array(nfr),HM=new Int32Array(nfr),le=new Float32Array(NB),tmp=new Float32Array(NB),
    Pw=new Float32Array(NB),Nw=new Float32Array(NB),Ph=new Float32Array(NB),cst=2/(N*N*.375),
    ss=(a,b,x)=>{const t=Math.max(0,Math.min(1,(x-a)/(b-a)));return t*t*(3-2*t)},
    fat=u=>{const i=Math.min(f0.length-1,Math.floor(u)),j=Math.min(f0.length-1,i+1),a=u-i;
      return vo[i]&&vo[j]?Math.exp(Math.log(f0[i])*(1-a)+Math.log(f0[j])*a):(vo[i]?f0[i]:f0[j])};
  for(let f=0;f<nfr;f++){
    if(f%40===0){vg(.4+.3*f/nfr,'Building voice…');await tick()}
    if(!sa[f])continue;
    le.fill(0);for(let q=0;q<C;q++){const y=P[f*C+q],cs=COS[q];for(let k=0;k<NB;k++)le[k]+=y*cs[k]}
    if(form!==1){tmp.set(le);for(let k=0;k<NB;k++){const u=k/form,i=Math.min(NB-2,Math.floor(u)),a=Math.min(1,u-i);le[k]=tmp[i]*(1-a)+tmp[i+1]*a}}
    let mxv=-1e9;for(let k=0;k<NB;k++)if(le[k]>mxv)mxv=le[k];
    for(let k=0;k<NB;k++)le[k]=Math.max(le[k]-mxv,-9);
    const u=f*H/HOP,v=vo[Math.min(vo.length-1,Math.round(u))],f0f=v?Math.min(800,Math.max(50,fat(u)*ratio)):0;
    let Eh=0,En=0;
    for(let k=0;k<NB;k++){const nw=v?.03+.55*ss(3500,9000,k*SR/N):1,pw=Math.exp(2*le[k]);Pw[k]=pw;Nw[k]=nw;Ph[k]=v?(1-nw)*pw:0;Eh+=Ph[k];En+=nw*pw}
    Eh*=cst;En*=cst;const gn=Math.sqrt(rs[f]*rs[f]/(Eh+En+1e-20));
    if(v){const Hn=Math.min(NH,Math.floor(10500/f0f));let sp=0;const tp=new Float32Array(Hn);
      for(let h=1;h<=Hn;h++){const b=h*f0f*N/SR,i=Math.min(NB-2,Math.floor(b)),a=b-i;tp[h-1]=Ph[i]*(1-a)+Ph[i+1]*a;sp+=tp[h-1]}
      for(let h=1;h<=Hn;h++)Am[f*NH+h-1]=gn*Math.sqrt(2*Eh*tp[h-1]/(sp+1e-20));
      HM[f]=Hn;F0[f]=f0f}
    for(let k=1;k<N/2;k++){const m=gn*Math.sqrt(Nw[k]*Pw[k]/.375)*Math.sqrt(-Math.log(Math.random()+1e-9)),a=Math.random()*6.2831853;
      Re[k]=m*Math.cos(a);Im[k]=m*Math.sin(a);Re[N-k]=Re[k];Im[N-k]=-Im[k]}
    Re[0]=Im[0]=Re[N/2]=Im[N/2]=0;fft(Re,Im,1);
    const p=f*H-N/2;for(let i=0;i<N;i++){const q=p+i;if(q>=0&&q<out.length)out[q]+=Re[i]*win[i]/1.2247}
  }
  const PH=new Float32Array(NH+1);for(let h=1;h<=NH;h++)PH[h]=2*Math.PI*.6*(((.6180339*h*h)%1)-.5);
  let ph=0;
  for(let n2=0;n2<len;n2++){
    if((n2&16383)===0){vg(.7+.3*n2/len,'Synthesizing voice…');await tick()}
    const u=n2/H,f=Math.min(nfr-1,Math.floor(u)),f1=Math.min(nfr-1,f+1),a=u-f,a0=F0[f],a1=F0[f1],
      fi=a0>0&&a1>0?a0*(1-a)+a1*a:(a0>0?a0:a1);
    if(fi<=0)continue;
    ph+=2*Math.PI*fi/SR;if(ph>6.2831853)ph-=6.2831853;
    const Hn=Math.max(HM[f],HM[f1]),o0=f*NH,o1=f1*NH;let z=0;
    for(let h=1;h<=Hn;h++)z+=(Am[o0+h-1]*(1-a)+Am[o1+h-1]*a)*Math.sin(h*ph+PH[h]);
    out[n2]+=z;
  }
  return out;
}

/* ---------- main conversion ---------- */
let dbKey='',dbVal=null;
async function convert(){
  const x=S.src,P={semi:+val('pitch'),form:+val('form'),ret:+val('ret')/100,K:+val('k')},
    [fmin,fmax]=$('rng').value.split(',').map(Number);
  const pg=(a,b,t)=>p=>{$('bar').firstChild.style.width=((a+(b-a)*p)*100)+'%';if(t)$('msg').textContent=t};
  $('bar').style.display='block';const md=$('mode').value;
  const[f0r,vor]=await trackPitch(x,fmin,fmax,pg(0,.15,'Tracking pitch…')),[f0,vo]=fillGaps(f0r,vor);
  let db=null;
  if(S.tgt&&(P.ret>0||md!=='env')){const key=S.tgt.length+':'+fmin+':'+fmax+':'+S.tgt[S.tgt.length>>1];
    if(key!==dbKey){db=await buildDB(S.tgt,fmin,fmax,pg(.15,.4,'Indexing target voice…'));dbKey=key;dbVal=db}else db=dbVal}
  if(db&&db.n<30){db=null;$('msg').textContent='Target too short or quiet: ignoring retrieval.'}
  const grain=!!db&&md==='grain',voc=!!db&&md==='voc',direct=grain||voc;
  // 1) pitch
  $('msg').textContent='Shifting pitch…';await tick();
  let ratio=2**(P.semi/12);
  if(db&&$('auto').checked&&db.med){const sm=median(f0,vo);if(sm)ratio*=db.med/sm}
  ratio=Math.max(.4,Math.min(3,ratio));
  const y=(Math.abs(ratio-1)>.005&&!direct)?psola(x,makeMarks(x,f0,vo),ratio):x.slice();
  // 2) analyse features of y
  const nfr=Math.floor((y.length+N/2)/H)+1,le=new Float32Array(NB),sv=new Uint8Array(nfr),sa=new Uint8Array(nfr),rs=new Float32Array(nfr);
  let nb=null,sf=null;
  if(db){sf=new Float32Array(nfr*D);
    for(let f=0;f<nfr;f++){if(f%80==0){pg(.4,.5,'Analysing source…')(f/nfr);await tick()}
      const p=f*H-N/2;rs[f]=analyze(y,p,le);sa[f]=rs[f]>=SIL?1:0;feat(le,sf,f*D);
      sv[f]=vo[Math.min(vo.length-1,Math.max(0,Math.round((p+N/2)/HOP)))]}
    cmn(sf,sa);{const sm=sf.slice();for(let f=1;f<nfr-1;f++)for(let q=0;q<D;q++)sf[f*D+q]=.25*sm[(f-1)*D+q]+.5*sm[f*D+q]+.25*sm[(f+1)*D+q]}
    nb=await knn(sf,sv,sa,db,P.K,pg(.5,.7,'Retrieving closest frames…'))}
  else for(let f=0;f<nfr;f++)sv[f]=vo[Math.min(vo.length-1,Math.max(0,Math.round((f*H)/HOP)))];
  // 3) envelope transfer + overlap-add
  const o=grain?grainSynth(y.length,f0,vo,nb,P.K,db,ratio,rs,nfr,sf):voc?await vocode(y.length,f0,vo,nb,P.K,db,ratio,rs,sf,sa,nfr,P.form,(p,t)=>pg(.7,1,t)(p)):new Float32Array(y.length+N),tl=new Float32Array(NB),ret=new Float32Array(NB),prev=new Float32Array(NB);let have=false;
  for(let f=0;f<(direct?0:nfr);f++){
    if(f%60==0){pg(.7,1,'Transforming timbre…')(f/nfr);await tick()}
    const p=f*H-N/2,rms=analyze(y,p,le);
    if(rms>=SIL){
      let mixed=le;
      if(nb&&nb[f*P.K]>=0&&P.ret>0){
        ret.fill(0);let c=0;for(let q=0;q<P.K;q++){const j=nb[f*P.K+q];if(j<0)continue;c++;const e=db.le[j];for(let k=0;k<NB;k++)ret[k]+=e[k]}
        let ms=0,mr=0;for(let k=0;k<NB;k++){ret[k]/=c;ms+=le[k];mr+=ret[k]}const off=(ms-mr)/NB;
        for(let k=0;k<NB;k++){let r=ret[k]+off;if(have)r=.6*r+.4*prev[k];prev[k]=r;tl[k]=le[k]*(1-P.ret)+r*P.ret}
        have=true;mixed=tl}
      for(let k=0;k<NB;k++){
        let t=mixed[k];
        if(P.form!==1){const u=k/P.form,i=Math.min(NB-2,Math.floor(u)),a=Math.min(1,u-i);t=mixed[i]*(1-a)+mixed[i+1]*a}
        let d=Math.max(-3,Math.min(3,t-le[k]));if(!sv[f])d*=.6;
        const g=Math.exp(d);Re[k]*=g;Im[k]*=g;if(k>0&&k<N/2){Re[N-k]*=g;Im[N-k]*=g}}
    }
    fft(Re,Im,1);
    for(let i=0;i<N;i++){const q=p+i;if(q>=0&&q<o.length)o[q]+=Re[i]*win[i]/1.5}
  }
  let pk=0;const res=o.slice(0,y.length);for(let i=0;i<res.length;i++){if(!(res[i]===res[i]))res[i]=0;pk=Math.max(pk,Math.abs(res[i]))}
  const gn=pk>0?.92/pk:1;for(let i=0;i<res.length;i++)res[i]*=gn;
  $('bar').firstChild.style.width='100%';
  return res;
}

