use std::collections::HashMap;
use std::any::Any;

use std::cell::RefCell;

use std::time::Instant;

use futures::executor::block_on;

use rand::prelude::*;

use switchyard::Switchyard;

use switchyard::threads::{thread_info, single_pool_one_to_one};

struct ThreadLocalData {
	rng: StdRng,
	count: u64,
}
impl ThreadLocalData {
	fn new() -> ThreadLocalData {
		ThreadLocalData{ rng: StdRng::from_entropy(), count: 0 }
	}
}

fn main() {
    println!();

	let start = Instant::now();
	let mut yard = Switchyard::new(1, single_pool_one_to_one(thread_info(), None), || RefCell::new(ThreadLocalData::new())).unwrap();
	const N_JOBS: u64  = 32;
	const N_ITERS_PER_JOB: u64  = 1_000_000;
	for _i in (0 .. N_JOBS) {
		yard.spawn_local(0, 0, move |data| {
						 async move { 
						let mut borrowed_data = data.borrow_mut();
						let mut local_count = 0;
						for _j in (0 .. N_ITERS_PER_JOB) {
							
			                    let (x, y) = borrowed_data.rng.gen::<(f32, f32)>();//::<(f32, f32)>();
			                
							if ((x * x) + (y * y)) <= 1.0 {
								local_count = (local_count + 1);
							}
						}
						borrowed_data.count = (borrowed_data.count + local_count);
						 } 
					});
    }
    println!("{:#?}", "CPU and Threads"); 
	block_on(yard.wait_for_idle());
	let total_hits = yard.access_per_thread_data().unwrap().iter().map(|d| d.borrow_mut().count).sum::<u64>();
	println!("{:#?}", (4.0 * ((total_hits as f64) / (((N_JOBS * N_ITERS_PER_JOB)) as f64))));
    yard.finish();
    println!("{:#?}s for {} cycles", start.elapsed().as_secs_f64(), N_ITERS_PER_JOB * N_JOBS); 
    
    // ==========================
    println!();

    use std::fs;
    extern crate ocl;
    use ocl::ProQue;
    
    
    use std::time::Instant;
    
    const N_JOBS_GPU: u64  = 50_000;
    const N_ITERS_PER_JOB_GPU: u64  = 1_000_000;
    let start = Instant::now();
    
    let shader_contents = fs::read_to_string("shaders/main.comp").unwrap();
    
    let pro_que0368 = ProQue::builder()
        .src(shader_contents)
        .dims([N_JOBS_GPU])
        .build().unwrap();
    
    let count_buffer = ocl::Buffer::<u32>::builder()
        .queue(pro_que0368.queue().clone())
        .flags(ocl::flags::MEM_READ_WRITE)
        .len([N_JOBS_GPU])
        .build().unwrap();
    
    
    let count_kernel = pro_que0368.kernel_builder("main")
        .arg(&count_buffer)
        .arg(&(N_ITERS_PER_JOB_GPU as u32))
        .build().unwrap();
    
    let range = (0 .. N_JOBS_GPU);
    let mut counts: Vec<u32> = range.map(|_i| 0).collect();
    
    
    count_buffer.write(&counts).enq().unwrap();
    
    unsafe { count_kernel.enq().unwrap(); }
    
    count_buffer.read(&mut counts).enq().unwrap();
    let total_hits: u64 = counts.iter().map(|x| *x as u64).sum();
    println!("{:#?}", "GPU and Shaders");
    println!("{:#?}", (4.0 * ((total_hits as f64) / (((N_JOBS_GPU * N_ITERS_PER_JOB_GPU)) as f64))));
    println!("{:#?}s for {} cycles", start.elapsed().as_secs_f64(), N_ITERS_PER_JOB_GPU * N_JOBS_GPU);
    println!();
}

