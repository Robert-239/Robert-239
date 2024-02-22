package main

import( 
	"fmt"
	"net"
//	"log"
	"time"
)

type Server struct{
	listenAddr string
	ln net.Listener
}

func NewServer(listenAddr string) *Server {
	return &Server{
		listenAddr: listenAddr,
	}
}

func main()  {
	now := time.Now()
	fmt.Printf("Ho Ho Ho")
	fmt.Println("hello world, the time is: ",now)	
}
