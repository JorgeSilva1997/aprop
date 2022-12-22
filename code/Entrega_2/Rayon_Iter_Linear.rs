const NPOINTS: usize = 1000;
const MAXITER: usize = 1000;

use std::time::Instant;
use rayon;
use rayon::prelude::*;
use std::sync::{Arc ,Mutex};

 struct DComplex {
     r: f64,
     i: f64,
 }

fn testpoint(i: i32, j:i32) -> i32{

    let eps:f64 = 1.0e-5;

    let c = DComplex {
        r: -2.0+2.5 * i as f64 / ( NPOINTS as f64 + eps),
        i: 1.125 * j as f64 / (NPOINTS as f64 + eps),
    };

    let mut z = DComplex {
        r: c.r,
        i: c.i,
    };

    (0..MAXITER).into_iter().find(|_| {
        let temp = (z.r*z.r)-(z.i*z.i)+c.r;
        z.i = z.r*z.i*2.0+c.i;
        z.r = temp;
        z.r*z.r+z.i*z.i>4.0
    }).map_or(0, |_| 1)
}


fn main() {

    let area:f64;
    let error:f64;
    let numoutside = Arc::new(Mutex::new(0));

    let now = Instant::now();

    (0..NPOINTS).into_par_iter().for_each(|i| {
        (0..NPOINTS).into_par_iter().for_each(|j| {
            let test = testpoint(i as i32, j as i32);
            if test == 1 {
                let mut numoutside = numoutside.lock().unwrap();
                *numoutside+= test;
            }
        });
    });

    let numoutside = *numoutside.lock().unwrap();
    let end = now.elapsed();
    println!("numoutside = {}", numoutside);

    area = 2.0*2.5*1.125* (NPOINTS as f64 *NPOINTS as f64 - numoutside as f64)/ (NPOINTS as f64 *NPOINTS as f64);
    error = area/NPOINTS as f64;

    println!("Area of Mandlebrot set  = {} +/- {}", area, error);
    println!("Time = {:?}", end);

}
