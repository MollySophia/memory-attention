"""Persistent native CPU gather workers, separate from model timing."""
import argparse,ctypes,hashlib,json,subprocess,sys,tempfile,time
from pathlib import Path
SOURCE=r'''
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>
struct Pool {
 std::mutex mutex; std::condition_variable start, done;
 std::vector<std::thread> workers;
 bool stop=false; int generation=0, remaining=0, count;
 const char* src; char* dst; const int64_t* ids;
 int64_t n,stride,bytes;
 explicit Pool(int c):count(c) {
  for(int t=0;t<count;t++) workers.emplace_back([this,t]{
   int seen=0;
   for(;;) {
    std::unique_lock<std::mutex> lock(mutex);
    start.wait(lock,[&]{return stop || generation!=seen;});
    if(stop) return;
    seen=generation; lock.unlock();
    for(int64_t i=n*t/count;i<n*(t+1)/count;i++)
     std::memcpy(dst+i*bytes,src+ids[i]*stride,bytes);
    lock.lock();
    if(--remaining==0) done.notify_one();
   }
  });
 }
 ~Pool(){
  {std::lock_guard<std::mutex> lock(mutex);stop=true;start.notify_all();}
  for(auto& worker:workers) worker.join();
 }
};
extern "C" void* create_pool(int threads) {return new Pool(threads);}
extern "C" void close_pool(void* p) {delete static_cast<Pool*>(p);}
extern "C" int gather_pool(void* p,const char* src,char* dst,const int64_t* ids,
 int64_t n,int64_t vocab,int64_t stride,int64_t bytes) {
 for(int64_t i=0;i<n;i++) if(ids[i]<0 || ids[i]>=vocab) return -1;
 Pool& pool=*static_cast<Pool*>(p);
 std::unique_lock<std::mutex> lock(pool.mutex);
 pool.src=src;pool.dst=dst;pool.ids=ids;pool.n=n;pool.stride=stride;pool.bytes=bytes;
 pool.remaining=pool.count;++pool.generation;pool.start.notify_all();
 pool.done.wait(lock,[&]{return pool.remaining==0;});
 return 0;
}
'''
def library():
 root=Path(tempfile.gettempdir())/('offload-persistent-diagnostic-'+hashlib.sha256(SOURCE.encode()).hexdigest());root.mkdir(exist_ok=True)
 (root/'gather.cpp').write_text(SOURCE)
 command=['g++','-O3','-std=c++17','-shared','-fPIC','-pthread',str(root/'gather.cpp'),'-o',str(root/'gather.so')]
 subprocess.run(command,check=True,capture_output=True)
 lib=ctypes.CDLL(str(root/'gather.so'));lib.create_pool.argtypes=[ctypes.c_int];lib.create_pool.restype=ctypes.c_void_p
 lib.close_pool.argtypes=[ctypes.c_void_p];lib.close_pool.restype=None
 lib.gather_pool.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int64]*4;lib.gather_pool.restype=ctypes.c_int
 return lib,command

def main():
 p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 sys.path[:0]=[str(a.source_root),str(a.source_root/'profile')]
 import torch
 from benchmark_telemetry import environment_details,source_state
 lib,compiler=library();pools={f'pool{n}':lib.create_pool(n) for n in (4,8,16)};results=[]
 try:
  with torch.inference_mode():
   torch.manual_seed(1234);table=torch.randn(32000,24,2048,dtype=torch.bfloat16)
   for n in (2048,16384,32768):
    host=torch.empty(n,1,2048,dtype=table.dtype,pin_memory=True);samples={name:[] for name in ['torch',*pools]}
    for trial in range(20):
     ids=torch.randint(0,32000,(n,));layer=trial%24;source=table[:,layer:layer+1];expected=source.index_select(0,ids)
     names=list(samples);names=names[trial%4:]+names[:trial%4]
     for name in names:
      t=time.perf_counter_ns()
      if name=='torch':torch.index_select(source,0,ids,out=host)
      else:assert lib.gather_pool(pools[name],source.data_ptr(),host.data_ptr(),ids.data_ptr(),n,32000,24*2048*2,2048*2)==0
      ms=(time.perf_counter_ns()-t)/1e6
      torch.testing.assert_close(host,expected,rtol=0,atol=0)
      if trial>=10:samples[name].append(ms)
    results.append(dict(tokens=n,samples_ms=samples,mean_ms={k:sum(v)/len(v) for k,v in samples.items()}))
   payload=dict(campaign_id='offload-gap-001',purpose=__doc__,source=source_state(),helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),command=sys.argv,compiler_command=compiler,compiler_version=subprocess.check_output(['g++','--version'],text=True),environment=environment_details(),warmup=10,trials=10,exactness='All changing-ID outputs exact',results=results,limitations='Persistent pools are separate prototype workers; Torch thread count stays unchanged. Only one method runs at a time. No model overlap or speedup claim.')
   a.output.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps([{k:v for k,v in r.items() if k!='samples_ms'} for r in results]))
 finally:
  for pool in pools.values():lib.close_pool(pool)
if __name__=='__main__':main()
