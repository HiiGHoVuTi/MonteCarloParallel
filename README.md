
# Short showcase of how easy it is to get started with fast computations and Glace

There are two versions of the same problem (which is Monte Carlo approximation of pi), dealt with using either safe threads or shaders.

In `main.rs` are the two versions once transpiled to Rust, ready to be ran using `cargo run`.
Note that the results will differ greatly depending on the hardware, but the spirit is the same.

My results (spaced out the big numbers): 
```rs
"CPU and Threads"
3.141550125
10.7173975s for 32_000_000 cycles

"GPU and Shaders"
3.1415915264
8.7319167s for 50_000_000_000 cycles
```