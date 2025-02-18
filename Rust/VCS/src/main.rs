use std::{error::Error, *};

fn main() {
    println!("Hello, world!");
    
    let mut id = String::new();

    init_repository("dir");

    


}

fn init_repository(dir : &str)-> Result<error> {
    dr = fs::create_dir(dir)?;
    Ok(());
    return dr;
}
