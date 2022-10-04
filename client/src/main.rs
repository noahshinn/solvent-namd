use std::fs;
use std::str;
use tokio::task;
use std::unimplemented;

use std::{thread, time::Duration};


const WRITE_FILE: &str = "out.log";
const NTRAJ: u16 = 500;

#[tokio::main]
async fn main() {
    let mut handles = vec![];

    // FIXME: executes in batches of size N where N is the number of cores
    for i in 0..NTRAJ {
        let h = task::spawn(async move {
            unimplemented!("call python microservice");
        });
        handles.push(h);
    }

    let mut results = vec![];
    for h in handles {
        results.push(h.await.unwrap());
    }

    let mut res_string: String = "".to_owned();
    for res in results {
        res_string.push_str(res);
        res_string.push_str("\n")
    }

    fs::write(WRITE_FILE, res_string).expect("Unable to write to file!");

}
