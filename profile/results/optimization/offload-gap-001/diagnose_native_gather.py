"""Fresh-ID native CPU gather prototypes; isolated evidence only."""
import argparse,ctypes,hashlib,json,subprocess,sys,tempfile,time
from pathlib import Path
SOURCE=r'''
#include <cstring>
#include <cstdint>
#include <thread>
extern "C" int gather(const char* src, char* dst, const int64_t* ids, int64_t n,
                       int64_t vocab, int64_t stride, int64_t bytes, int threads) {
 for(int64_t i=0;i<n;i++) if(ids[i]<0 || ids[i]>=vocab) return -1;
 auto copy=[&](int64_t begin,int64_t end){
  for(int64_t i=begin;i<end;i++) std::memcpy(dst+i*bytes,src+ids[i]*stride,bytes);
 };
 if(threads==1) copy(0,n);
 else {
  std::thread workers[4];
  for(int t=0;t<4;t++) workers[t]=std::thread(copy,n*t/4,n*(t+1)/4);
  for(auto& worker:workers) worker.join();
 }
 return 0;
}
'''
def compile_library():
 digest=hashlib.sha256(SOURCE.encode()).hexdigest()
 cache=Path(tempfile.gettempdir())/('offload-native-diagnostic-'+digest);cache.mkdir(exist_ok=True)
 source=cache/'gather.cpp';library=cache/'gather.so';source.write_text(SOURCE)
 command=['g++','-O3','-std=c++17','-shared','-fPIC','-pthread',str(source),'-o',str(library)]
 subprocess.run(command,check=True,capture_output=True)
 fn=ctypes.CDLL(str(library)).gather
 fn.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p]+[ctypes.c_int64]*4+[ctypes.c_int]
 fn.restype=ctypes.c_int
 return fn,command

def main():
 p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 sys.path[:0]=[str(a.source_root),str(a.source_root/'profile')]
 import torch
 from benchmark_telemetry import environment_details,source_state
 fn,compiler=compile_library();results=[]
 with torch.inference_mode():
  torch.manual_seed(1234);table=torch.randn(32000,24,2048,dtype=torch.bfloat16)
  for n in (2048,16384,32768):
   host=torch.empty(n,1,2048,dtype=table.dtype,pin_memory=True)
   samples={name:[] for name in ('torch','native_serial','native_threads4')}
   for trial in range(20):
    ids=torch.randint(0,32000,(n,));layer=trial%24;source=table[:,layer:layer+1];expected=source.index_select(0,ids)
    methods=list(samples);methods=methods[trial%3:]+methods[:trial%3]
    for name in methods:
     t=time.perf_counter_ns()
     if name=='torch':torch.index_select(source,0,ids,out=host)
     else:assert fn(source.data_ptr(),host.data_ptr(),ids.data_ptr(),n,32000,24*2048*2,2048*2,1 if name=='native_serial' else 4)==0
     ms=(time.perf_counter_ns()-t)/1e6
     torch.testing.assert_close(host,expected,rtol=0,atol=0)
     if trial>=10:samples[name].append(ms)
   results.append(dict(tokens=n,samples_ms=samples,mean_ms={k:sum(v)/len(v) for k,v in samples.items()}))
  payload=dict(campaign_id='offload-gap-001',purpose=__doc__,source=source_state(),helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),command=sys.argv,compiler_command=compiler,compiler_version=subprocess.check_output(['g++','--version'],text=True),environment=environment_details(),warmup=10,trials=10,exactness='All fresh-ID outputs exact',results=results)
  a.output.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps([{k:v for k,v in r.items() if k!='samples_ms'} for r in results]))
if __name__=='__main__':main()
