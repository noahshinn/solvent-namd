use std::fs;
use std::str;
//use tokio::task;
//use std::unimplemented;


const WRITE_FILE: &str = "out.log";

fn main() {
    let mut handles = vec![];

    for _ in 0..500 {
        let h = task::spawn(async {
            unimplemented!("call python microservice")
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
