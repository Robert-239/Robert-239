package main

import (
	"errors"
	"fmt"
)

func main() {
	var printValue string = "I was passed to a custom function"
	printMe(printValue)


	var numerator int = 13
	var denominator int  = 5
	var result , remainder , err = intDivision(numerator,denominator)
	if err != nil{	
		printMe(err.Error())
	}else if remainder == 0{
		
		fmt.Printf("The result of the integer division  %v",result)
	}
	fmt.Printf("The result of the integer division %v/%v is %v with remander %v",numerator,denominator,result,remainder)
}

func printMe(printValue string) {
	fmt.Println(printValue)
}

func intDivision(numerator int, denominator int) (int , int, error) {
	var err error
	if denominator == 0 {
		err = errors.New("Cannot devide by zero")
		return 0, 0 ,err
	}
	var result int = numerator / denominator
	var remainder  int  = numerator % denominator

	return result, remainder , err
}
