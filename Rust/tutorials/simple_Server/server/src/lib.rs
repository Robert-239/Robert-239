use std::{
    sync::{mpsc, Arc , Mutex },
    thread
};

pub struct ThreadPool {
    workers : Vec<Worker>,
    sender : mpsc::Sender<Job>,
}


type Job = Box< dyn FnOnce() + Send + 'static>;

impl ThreadPool {
    ///Creates new ThreadPool
    ///
    ///size is the number of threads in the pool
    ///
    /// # panics
    ///
    /// new function will panic if size is zero

    pub fn new(size : usize) -> ThreadPool {
    assert!(size > 0);

    let (sender,reciever) = mpsc::channel();
    let reciever = Arc::new(Mutex::new(reciever));
    let mut workers = Vec::with_capacity(size);

    for id in 0..size {
            workers.push(Worker::new(id,Arc::clone(&reciever)));
        }

    ThreadPool {workers , sender}
    }
    pub fn execute<F> (&self, f: F) where F: FnOnce() + Send + 'static,
    {
        let job = Box::new(f);
        self.sender.send(job).unwrap();
    }
}
    // pub fn build(size : usize) -> Result<ThreadPool , PoolCreationError >{}
struct Worker {
    id: usize,
    thread: thread::JoinHandle<()>,
}

impl Worker {
    fn new(id: usize , reciever: Arc<Mutex<mpsc::Receiver<Job>>>) -> Worker {
        let thread = thread::spawn(move || loop {
            let job = reciever.lock().unwrap().recv().unwrap();
            println!("Worker {id} got a job; executing");
            job();
        });
        Worker {id , thread}
    }
}

